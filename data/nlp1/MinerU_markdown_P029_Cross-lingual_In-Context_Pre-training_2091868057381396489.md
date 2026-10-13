# Enhancing LLM Language Adaption through Cross-lingual In-Context Pre-training

Linjuan Wu<sup>1</sup>*, Haoran Wei<sup>2†</sup>, Huan Lin<sup>2</sup>, Tianhao Li<sup>2</sup>, Baosong Yang<sup>2</sup>, Fei Huang<sup>2§</sup>, Weiming Lu<sup>1,‡</sup> 

<sup>1</sup>Zhejiang University 

<sup>2</sup>Tongyi Lab, Alibaba Group 

<sup>1</sup>{wulinjuan525,luwm}@zju.edu.cn 

<sup>2</sup>{funan.whr, yangbaosong.ybs}@alibaba-inc.com 

## Abstract

Large language models (LLMs) exhibit remarkable multilingual capabilities despite Englishdominated pre-training, attributed to crosslingual mechanisms during pre-training. Existing methods for enhancing cross-lingual transfer remain constrained by parallel resources, suffering from limited linguistic and domain coverage. We propose Cross-lingual In-context Pre-training (CrossIC-PT), a simple and scalable approach that enhances cross-lingual transfer by leveraging semantically related bilingual texts via simple next-word prediction. We construct CrossIC-PT samples by interleaving semantic-related bilingual Wikipedia documents into a single context window. To access window size constraints, we implement a systematic segmentation policy to split long bilingual document pairs into chunks while adjusting the sliding window mechanism to preserve contextual coherence. We further extend data availability through a semantic retrieval framework to construct CrossIC-PT samples from web-crawled corpus. Experimental results demonstrate that CrossIC-PT improves multilingual performance on three models (Llama-3.1-8B, Qwen2.5-7B, and Qwen2.5- 1.5B) across six target languages, yielding performance gains of 3.79%, 3.99%, and 1.95%, respectively, with additional improvements after data augmentation. 

## 1 Introduction

Recent state-of-the-art (SOTA) large language models (LLMs) (Achiam et al., 2023; Anthropic; Reid et al., 2024) have demonstrated remarkable multilingual capabilities. These models are typically pretrained on massive web-crawled corpora, where English text overwhelmingly dominates in the quantity (Brown et al., 2020; Dubey et al., 2024). However, current LLMs exhibit unexpectedly strong performance on non-English languages that cannot be fully explained by their relative data proportions during pre-training. Researchers have attributed this phenomenon to cross-lingual transfer in LLM training, where linguistic patterns and knowledge acquired from high-resource languages (particularly English) appear to transfer effectively to enhance performance on the other languages (Artetxe et al., 2020; Scao et al., 2022; Wang et al., 2024). 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/22a9270d-058b-4880-9777-03ba84ffaf64/e59a6d83cbf304ebab59cb98a342c17f09700a440ea218e028e2959d946314b7.jpg)



(b) Semantically Related Multilingual In-Context Data



Figure 1: Existing works randomly mix multilingual texts (a) in an input window. Our approach groups semantically related texts (b) to enhance cross-lingual transfer.


A series of works have explored methods for interpreting and enhancing cross-lingual transfer during language model pre-training. Blevins and Zettlemoyer (2022) revealed that even in English-dominated pre-training data, millions of non-English tokens can be identified, which are crucial for multilingual capabilities. Some studies have attempted to analyze cross-lingual transfer abilities from perspectives of shared vocabulary and representation similarity(Patil et al., 2022; Lin et al., 2023), though their conclusions primarily apply to specific language groups. The predominant research paradigm has focused on explicitly enhancing cross-lingual transfer through exploiting supervision signals, such as parallel corpora(Zhang et al., 2024b; Ming et al., 2024; Ji et al., 2024; Gosal et al., 2024; Gilabert et al., 2024), codeswitching datasets(Singh et al., 2024; Yoo et al., 2024), or fine-grained signals like cross lingual entity links(Yamada and Ri, 2024). These approaches, however, remain constrained by the limited quantity, domain coverage, and morphological diversity of available bilingual resources (e.g., dictionaries, and parallel sentence pairs). 

Our approach builds upon the fundamental principle of LLM pre-training: contextual modeling through next-word prediction (NWP) loss optimization within fixed-length text windows. Since LLMs could effectively learn monolingual semantics through this mechanism, we hypothesize that extending NWP optimization on semantically related cross-lingual content - using source language context to predict target language sequences - could enhance cross-lingual transfer capabilities. As illustrated in Fig.1(b), our method constructs Crosslingual In-context samples by interleaving semantically related bilingual text pairs. Subsequently, we optimize LLMs through standard NWP loss computation on these composite samples. The proposed Cross-lingual In-Context Pre-Training (CrossIC-PT) eliminates the reliance on parallel corpora, and could be applied to different types of text, providing a simple and scalable paradigm for cross-lingual transfer learning. 

To validate our method, we implement the proposed CrossIC-PT method through continued pretraining (CPT) on existing LLMs (Dubey et al., 2024; Yang et al., 2024). This strategy converges faster than training from scratch, providing a costeffective solution for multilingual experimentation (Zheng et al., 2024). Leveraging the readily available multilingual Wikipedia data, we construct a cross-lingual in-context corpus by concatenating two bilingual Wikipedia articles on the same entity, as illustrated in Fig.2. To mitigate context window length constraints, we segment article pairs into bilingual sub-pairs, using a dedicated [SPLIT] token as delimiters (Fig.2(b)). We further optimize the sliding window mechanism, ensuring that the next window starts from the token after the last [SPLIT] of the current window, thereby maintaining context coherence and enhancing cross-lingual alignment learning. To further assess the generalizability of our method, we develop a cross-lingual semantic retrieval framework build upon that extends beyond Wikipedia data by incorporating webcrawled text. As shown in Fig.3, this framework retrieves semantically related paragraphs from the English Fineweb_edu (Lozhkov et al., 2024) dataset using title and partial content keywords from the target-language Wikipedia articles as query. 

We conducted experiments in six languages based on three LLMs (Llama-3.1-8B, Qwen2.5- 7B, Qwen2.5-1.5B) and tested them on seven tasks. The CrossIC-PT model, built on Wikipedia, improved average performance by 3.79%, 3.99%, and 1.95% compared to the base models, respectively. The expansion of the data further boosted performance by 0.73% for Llama-3.1-8B. 

Our contributions can be summarized as follows: 

• We propose CrossIC-PT, a novel method that enhances LLMs’ cross-lingual transfer by leveraging semantically related in-context data. 

• To address input window length limitations, we design a window-split strategy with a [SPLIT] token and an optimized sliding window mechanism to maintain cross-lingual contextual coherence. 

• We also design a cross-lingual semantic retrieval framework to augment training data, which further enhances model performance, proving the robustness and scalability of our approach. 

## 2 Related Work

Many existing works focus on collecting multilingual data to enhance LLMs’ cross-lingual capabilities (Yang et al., 2024; Dubey et al., 2024; Ming et al., 2024; Ji et al., 2024). Samples from different languages are randomly packed into fixed window sizes (e.g., 4096) without cross-contamination in self-attention. Even so, these models already demonstrate multilingual ability. Based on this, we hypothesize that concatenating semantically related English and target language data (Fig.1(b)) could enhance cross-lingual transfer by leveraging implicit supervision signals. 

Cross-lingual supervision signals have been proven effective in enhancing LLMs’ cross-lingual transfer abilities (Singh et al., 2024; Yamada and Ri, 2024; Zhang et al., 2024c). Most methods rely on bilingual corpora as explicit supervision signals (Zhang et al., 2024b; Ming et al., 2024; Ji et al., 2024; Gosal et al., 2024; Gilabert et al., 2024). Some works, like (Zhang et al., 2024b) distills translation pairs from LLMs through backtranslation to create supervision signals. Others, such as (Singh et al., 2024; Yamada and Ri, 2024), apply code-switching techniques to replace or augment words with English translations. (Yoo et al., 2024) also explores code-switching at various levels using curriculum learning. However, parallel corpora have restricted types, domains (most bilingual corpora are short sentence level bitexts, and usually extracted from news websites), and quantity. Synthetic parallel documents built by backtranslation, however, are limited in text Quality. In contrast, our method constructs semantically related document pairs from the authentic data on the Internet, which is more scalable and less problematic. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/22a9270d-058b-4880-9777-03ba84ffaf64/56e0f9b317f3ea37a11d4cd26a45b0ffec12c3b59789b72e6af89249c0456c15.jpg)



Figure 2: The implementation process of our method, CrossIC-PT, which constructs cross-lingual in-contexts based on Wikipedia data and performs continued pre-training (CPT) on existing multilingual models. Here, N represents the input window length of the model. The T indicates the title of the articles, and L indicates the target language.


## 3 Method

Multilingual LLM pre-training typically packs documents from different languages randomly into the fixed-size context window. We hypothesize that concatenating semantically related English and target language corpora, predicting the next words based upon not only monolingual and cross-lingual context could enhance cross-lingual transfer ability. We call this concatenated sample Cross-lingual In-context data, where English serves as the guiding context for learning the target language. Based on this, we propose CrossIC-PT, a pre-training method leveraging cross-lingual in-context data. 

As LLMs are pre-trained with a fixed tokens window size (e.g. 4096 tokens), cross-lingual incontext data, which are usually two times longer than the vanilla monolingual documents, may exceed the size limit. Simplifying the packing by length may break the cross-lingual relationship. To address this problem, we carefully design a bilingual-aware window-split strategy to construct cross-lingual in-context data. Additionally, to avoid the traditional sliding window mechanism from splitting the concatenated context, we further optimize the sliding window mechanism to ensure context coherence. 

We take advantage of Wikipedia data to implement our method, as shown in Fig.2, consisting of three key steps: (1) Data preparation, where we extract and align bilingual article pairs from Wikipedia (Sec. 3.1); (2) Window-split crosslingual in-context construction, where we split multilingual contexts to match the length of the input window (Sec. 3.2); and (3) training with an optimized sliding window mechanism to enhance cross-lingual representation learning (Sec. 3.3). In order to test the generalization of our approach, we propose a cross-lingual semantic retrieval framework to augment the training data (Sec. 3.4). 

## 3.1 Data Preparation

To obtain aligned article pairs in English and the target language (denoted L), we utilize three key tables from Wikimedia with three steps: 

1. Langlinks Table for Language L: It contains article ID mappings between language L and other languages with matching titles, along with the corresponding title names T. This table helps identify English article IDs and title names that match those in language L, mapping as $( I D ^ { L } , ( I D ^ { e n } , T ^ { e n } ) )$ ). 

2. English Pages Table: The ‘pages‘ table of English provides article IDs and their correspond ing title. We use it to remove English articles with blank or invalid titles from the initial mappings in step (1), yielding the final ID pairs $( I D ^ { L } , I D ^ { e n } )$ 

3. Articles Tables for English and Language L: The ‘articles‘ tables for both languages contain the article ID and full information on the web page, which includes the article content. Using the bilingual article ID pairs $( I D ^ { L } , I D ^ { e n } )$ ), we extract the corresponding article pairs with matching titles. 

To ensure completeness, we also perform the reverse mapping $( I D ^ { e n } , I D ^ { L } )$ ), and combine the results with the forward mappings to obtain a comprehensive set of bilingual article pairs. This process ensures that we capture all possible title-matched articles between English and the target language. 

## 3.2 Window-split Cross-lingual In-Context Construction

To fit within the context size N, we set a strategy for processing long article pairs by segmenting them into paragraphs and aligning them sequentially. Specifically, for each bilingual article pair $( A _ { e n } , A _ { L } )$ , we extract the title T and split the articles into paragraphs by signal $" \setminus \setminus "$ 

$$
A _ {e n} = [ p _ {1} ^ {e n}, p _ {2} ^ {e n}, \dots , p _ {n} ^ {e n} ], A _ {L} = [ p _ {1} ^ {L}, p _ {2} ^ {L}, \dots , p _ {m} ^ {L} ].
$$

We iteratively select paragraph pairs $( p _ { i } ^ { e n } , p _ { i } ^ { L } )$ ) until adding the k-th pair would exceed the length N, and then concat the paragraphs as follows: 

$$
(T ^ {e n}, p _ {1} ^ {e n}; p _ {2} ^ {e n}; \dots ; p _ {k - 1} ^ {e n}; T ^ {L}, p _ {1} ^ {L}; p _ {2} ^ {L}; \dots ; p _ {k - 1} ^ {L}),
$$

with all English paragraphs preceding the target language L paragraphs, and the delimiter as $" \setminus \setminus "$ Each concatenated sequence is terminated with a special [SPLIT] token to mark the end of the context window. If the paragraphs of one language are exhausted before the other, we continue concatenating paragraphs from the remaining language until the length limit N is reached or all paragraphs are used. This process converts each bilingual article pair into one or more window-split multilingual in-contexts, each fitting within the length limit N. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/22a9270d-058b-4880-9777-03ba84ffaf64/8b9cdac3cd1ce1ca21af9269d8b3f1acebcfd76dda4b95ee5904884b6c0f6470.jpg)



Figure 3: The framework of cross-lingual semantic retrieval based on FAISS similarity search tool.


## 3.3 Pre-training Method

## 3.3.1 Sliding Window Mechanism

In standard pre-training, the sliding window mechanism concatenates all training data and sliding with a fixed window size. However, this can randomly break down our cross-lingual in-contexts, disrupting coherence. To address this, we optimize the sliding window by the introduced tag "[SPLIT]". Specifically, all the windows set the start boundary after the last "[SPLIT]" token, as shown in Fig. 2. The tokens remain between the end boundary and the latest "[SPLIT]" token will be dropped. In this way we could try best to preserve the cross-lingual coherence within the window. 

## 3.3.2 Training Strategy

As discussed earlier, continual pre-training (CPT) is cost-effective for cross-lingual transfer. So we adopt it in all our experiments. Recent studies, such as (Whitehouse et al., 2024), show that Low-Rank Adaptation (LoRA) is highly competitive with full fine-tuning, especially in low-data and cross-lingual transfer scenarios. In our experiments, we also adopt LoRA during continual pretraining and results reveal that LoRA consistently provides better and more stable performance. 

## 3.4 Data Augmentation via Retrieval

To validate our approach, we use the Wikipedia corpus, which includes data in nearly 200 languages linked by matched titles. While the content across languages is not strictly parallel, it covers the same topics, making it suitable for our needs. To enhance the generalization of our method, we introduce a cross-lingual semantic retrieval framework based on the FAISS similarity search tool (Johnson et al., 2019)*, as shown in Fig.3. This framework augments the training data by incorporating relevant English articles from the Fineweb_edu (Lozhkov et al., 2024) dataset, retrieved using title and content keywords (up to 10 per article) extracted from the Wikipedia data. 

First, keywords are extracted from the targetlanguage Wikipedia page and mapped to English via the langlinks table. Fineweb_edu is then indexed using FAISS for similarity calculations. We employ a two-step retrieval process using FAISS: (1) retrieval based on title keywords, and (2) retrieval based on both title and content keywords. The final similarity score is the average of these two steps, balancing the importance of the titles (which may be ambiguous) and content keywords. Based on empirical observations, we set a similarity threshold of 0.75 and retrieved up to three relevant samples per target-language article to construct window-split cross-lingual in-context data. These samples are combined with the original Wikipedia data to form an augmented dataset. 

## 4 Experiments

## 4.1 Training Data

Our training data is primarily sourced from Wikipedia<sup>†</sup> (denoted as W), with token counts for English and each target language listed in Table 1. We selected six target languages L: Arabic (ar), Spanish (es), Japanese (ja), Korean (ko), Portuguese (pt), and Thai (th). To further expand the dataset, we retrieved relevant English data from a subset of Fineweb_edu (denoted as F), which has a file size of 17.44GB. The token counts for the augmented data are also provided in Table 1. 

<table><tr><td>data</td><td>language</td><td>ar</td><td>es</td><td>ja</td><td>ko</td><td>pt</td><td>th</td></tr><tr><td rowspan="2">W</td><td>en</td><td>1.53B</td><td>1.88B</td><td>1.32B</td><td>1.01B</td><td>1.48B</td><td>0.42B</td></tr><tr><td>L</td><td>0.67B</td><td>1.57B</td><td>1.28B</td><td>0.37B</td><td>0.81B</td><td>0.18B</td></tr><tr><td rowspan="2">F</td><td>en</td><td>0.12B</td><td>0.10B</td><td>0.06B</td><td>0.04B</td><td>0.05B</td><td>0.10B</td></tr><tr><td>L</td><td>0.12B</td><td>0.13B</td><td>0.08B</td><td>0.03B</td><td>0.05B</td><td>0.06B</td></tr></table>


Table 1: The token counts for the data from Wikipedia (W) and augmented data from Wikipedia and Fineweb_edu (F).


## 4.2 Training Settings

We conducted experiments on three base models: Llama-3.1-8B (Dubey et al., 2024), Qwen2.5- 7B (Yang et al., 2024), and Qwen2.5-1.5B (Yang et al., 2024). For LoRA, we set the rank to 64, alpha to 128, and dropout to 0.05. The input window length N was set to 4096, with a batch size of 128. All models were trained for one epoch, using a warmup ratio of 0.05, a cosine learning rate scheduler, and the AdamW optimizer. We randomly selected 0.1% of the data as the validation set, with a seed number of 32. For Llama-3.1- 8B and Qwen2.5-7B, the models after one epoch of training were used as the final models. For Qwen2.5-1.5B, we validated the model every 100 steps and saved the checkpoint with the lowest validation loss as the final model. The training was performed on 8 A100 GPUs. 

## 4.3 Benchmark

We evaluated our models on several tasks from the latest multilingual and multitask benchmark, P-MMEVAL (Zhang et al., 2024a), which includes: generation (FLORES-200 (Costa-jussà et al., 2022)), understanding (XNLI (Conneau et al., 2018), MHELLASWAG <sup>‡</sup>), knowledge (MMMLU<sup>§</sup>), logical reasoning (MLOGIQA), and mathematical reasoning (MGSM (Shi et al., 2023)). To further assess the models’ paragraph comprehension abilities, we incorporated a reading comprehension task (MRC). The MRC test data includes TydiQA-GoldP (Clark et al., 2020) for Arabic (ar) and Korean (ko), XQuAD (Artetxe et al., 2020) for Spanish (es), Portuguese (pt), and Thai (th), and 1,200 samples from JaQuAD (So et al., 2022) for Japanese (ja). Details of the evaluation setting can be found in Appendix A. 

## 4.4 Baselines

In addition to the base models (Llama-3.1-8B, Qwen2.5-7B, and Qwen2.5-1.5B), we included the most relevant baseline Mix-PT, which uses titlematched article pairs (Fig. 2(a)) for pre-training but without cross-lingual text concatenation. 

Based on Llama-3.1-8B we also included two other baselines:(1)LEIA (Yamada and Ri, 2024): A method that randomly adds English translations of entities to target-language Wikipedia data for pre-training, leveraging cross-lingual entity supervision. We reproduced this method using the provided code to construct the data and perform CPT on Llama-3.1-8B, ensuring the target-language token count matched ours. We conducted experiments with three random seeds (32, 111, 222) and reported the mean and variance of the results. (2)X-MONO-PT: A method that uses the target language data from our title-matched article pairs in Fig.2(a) for pre-training. 

<table><tr><td rowspan="2" colspan="2">Model</td><td colspan="6">Languages</td><td rowspan="2">AVG</td></tr><tr><td>ar</td><td>es</td><td>ja</td><td>ko</td><td>pt</td><td>th</td></tr><tr><td rowspan="9">Llama-3.1-8B</td><td>base</td><td>37.96</td><td>42.11</td><td>43.02</td><td>43.82</td><td>44.36</td><td>38.79</td><td>41.68</td></tr><tr><td>LEIA</td><td><eq>37.04 \pm 0.49</eq></td><td><eq>44.03 \pm 0.24</eq></td><td><eq>44.86 \pm 0.89</eq></td><td><eq>44.11 \pm 0.48</eq></td><td><eq>44.48 \pm 0.58</eq></td><td><eq>42.90 \pm 0.82</eq></td><td><eq>42.98 \pm 0.25</eq></td></tr><tr><td>X-MONO-PT</td><td>39.11</td><td>44.35</td><td>43.57</td><td>44.24</td><td>44.04</td><td>42.09</td><td>42.90</td></tr><tr><td>Mix-PT</td><td>38.24</td><td>44.09</td><td>44.09</td><td>44.12</td><td>45.65</td><td>41.44</td><td>42.94</td></tr><tr><td>CrossIC-PT</td><td>39.66</td><td>44.57</td><td>45.61</td><td>46.02</td><td>47.32</td><td>42.30</td><td>44.24</td></tr><tr><td>Mix-PT</td><td>38.09</td><td>43.46</td><td>44.81</td><td>44.75</td><td>46.45</td><td>42.38</td><td>43.32</td></tr><tr><td>CrossIC-PT</td><td>40.57</td><td>45.49</td><td>47.27</td><td>46.87</td><td>49.09</td><td>43.51</td><td>45.47</td></tr><tr><td>Mix-PT</td><td>40.19</td><td>44.58</td><td>44.75</td><td>44.48</td><td>46.62</td><td>42.05</td><td>43.78</td></tr><tr><td>CrossIC-PT</td><td>41.18</td><td>46.93</td><td>48.10</td><td>47.32</td><td>49.97</td><td>43.72</td><td>46.20</td></tr><tr><td rowspan="3">Qwen-2.5-7B</td><td>base</td><td>50.91</td><td>54.71</td><td>56.95</td><td>55.52</td><td>56.49</td><td>53.81</td><td>54.73</td></tr><tr><td>Mix-PT</td><td>54.48</td><td>58.71</td><td>57.69</td><td>57.39</td><td>60.30</td><td>56.19</td><td>57.46</td></tr><tr><td>CrossIC-PT</td><td>55.97</td><td>59.44</td><td>59.00</td><td>59.03</td><td>61.59</td><td>57.33</td><td>58.73</td></tr><tr><td rowspan="3">Qwen-2.5-1.5B</td><td>base</td><td>37.83</td><td>43.90</td><td>42.26</td><td>39.75</td><td>44.35</td><td>41.40</td><td>41.58</td></tr><tr><td>Mix-PT</td><td>38.14</td><td>44.37</td><td>41.85</td><td>39.48</td><td>45.63</td><td>40.92</td><td>41.73</td></tr><tr><td>CrossIC-PT</td><td>40.21</td><td>45.09</td><td>43.96</td><td>41.47</td><td>48.25</td><td>42.23</td><td>43.54</td></tr></table>


Table 2: The average results of our CrossIC-PT model, based on three base LLMs (Llama-3.1-8B, Qwen2.5-7B, and Qwen2.5-1.5B), are compared with corresponding baselines across six target languages. The cross-lingual in-context datasets used in methods based on Qwen2.5-7B and Qwen2.5-1.5B are sourced from Wikipedia(W).


## 4.5 Results

## 4.5.1 Base Results

ALl average six languages results of the baselines and our method, based on different training data volume from Wikipedia (W) and Fineweb_edu (F) are shown in Table 2. Detailed results for each languages can be found in Appendix B. CrossIC-PT consistently improves the performance of the base LLMs and outperforms other baselines, demonstrating the effectiveness of using semantically related cross-lingual in-context corpora for pre-training. 

Compared to the base LLMs, our CrossIC-PT method trained with only Wikipedia(W) data improves performance by 3.79%, 3.99%, and 1.95% on Llama-3.1-8B, Qwen2.5-7B, and Qwen2.5- 1.5B, respectively, across six languages. Notably, in Portuguese (pt), CrossIC-PT improves performance by 4.73% on Llama-3.1-8B, surpassing the strongest baseline by 2.64%. The performance gains for Qwen2.5 models are more pronounced as model size increases, which may be attributed to the fact that CPT performance is influenced by the initial capabilities of the model. 

Our method consistently improves performance across all languages. The improvement in Thai is less noticeable on Qwen2.5-1.5B, likely due to the smaller dataset size. The LEIA method shows significant gains in some languages (Spanish, Japanese, and Thai), but its performance is unstable and data-dependent. For instance, the standard deviation for Japanese and Thai exceeds 0.8. This suggests that the implicit supervision signals from our cross-lingual in-context data are more robust and adaptable across languages compared to the entity-alignment signals used by LEIA. 

The Mix-PT model is a strong baseline, trained on non-concatenated title-matched article pairs from Wikipedia, and improves performance across all six languages compared to the three base LLMs. However, our method improves the average performance by 2.15% over the Mix-PT model on Llama-3.1-8B. Our method further enhances Mix-PT by concatenating cross-lingual data and designing an optimized sliding window mechanism. 

## 4.5.2 Results of Data Augmentation

To explore the generalization of our method, we propose a cross-lingual semantic retrieval framework (Fig. 3) to augment the training data by expanding Wikipedia (W) with FineWeb-edu (F). After retrieval, the data volume increased by 0.06B–0.23B tokens. Although this is a relatively small increase, it improved the average performance of our method by 0.73%. Even when using only the augmented data (F), our method still achieved a 1.26% improvement over the strong baseline LEIA. This demonstrates that even if English data is not perfectly aligned with target languages, semantic similarity still facilitates crosslingual transfer. Moreover, the simplicity of our retrieval process allows easy extension to various data sources. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/22a9270d-058b-4880-9777-03ba84ffaf64/a716a7167b8b56337e1318e551d01a1f7620a35bad5263c5e904c1020f7e9493.jpg)



Figure 4: Performance progression of CrossIC-PT across intermediate checkpoints based on Llama-3.1-8B. Our method outperforms the baseline LLM early on, indicating quick acquisition of cross-lingual transfer capabilities,: Response n maintaining a slow upward trend as data volume increases.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/22a9270d-058b-4880-9777-03ba84ffaf64/e0b08db4dd5f8d81041211510d00265158daf422bc84f444cf2a02fc6c8479cb.jpg)



Figure 5: Ablation results of CrossIC-PT.


Additionally, we saved several intermediate checkpoints to assess the impact of data volume on performance. As shown in Fig. 4, at earlier checkpoints, our method outperformed the baseline LLM in all six languages and surpassed the strong baseline Mix-PT in four languages. This suggests that CrossIC-PT can quickly acquire useful crosslingual transfer capabilities from the cross-lingual in-context data. Although performance improvements became slower as data volume increased, a consistent upward trend was still observed. 

## 4.6 Ablation Study

We conduct ablation studies to evaluate two key components of our approach: (1) the optimized sliding window mechanism (Opt-SWM) and (2) the semantic related of cross-lingual contexts. First, comparing CrossIC-PT with and without Opt-SWM (denoted as CrossIC-PT w/o Opt-SWM), Fig.5 shows that while window-split cross-lingual contexts alone improve performance across all languages, adding Opt-SWM further enhances results by maintaining context coherence. Second, when replacing semantically similar pairs with randomly paired bilingual documents (CrossIC-PT_random), performance degrades significantly - though still surpassing Mix-PT baselines - confirming the importance of our semantic similarity strategy. These results collectively validate the effectiveness of each design component in CrossIC-PT. 

## 5 Analysis

## 5.1 Concatenation Direction

We believe that placing semantically related English text before target-language content helps models better learn from cross-lingual contexts. Thus, we set the order as English first, followed by the target language. To verify if this direction is more beneficial, we analyze the concatenation order. 

To evaluate the impact of concatenation direction on performance, we compare the original direction (English first, target language second) with the reverse direction (target language first, English second), as well as a 1:1 random mix of both directions. Previously, we only reported results for the en-xx direction in the translation task. In this experiment, we also provide results for the xx-en direction on FLORES-200. 

The average results of six languages across tasks are presented in Table 3. The effect of data concatenation order on translation tasks is most pronounced and fits the intuition. The best translation performance occurs when the concatenation direction matches the translation direction. When combining both directions, CrossIC-PT consistently outperforms the Mix-PT method in translation tasks, showing that even non-parallel bilingual data improves translation. Overall, the English-first, target language-second concatenation gives the best results, aligning with our intention of using English as context to guide the target language learning. 

<table><tr><td></td><td>pt</td><td>ko</td><td>ja</td><td>es</td><td>th</td><td>ar</td></tr><tr><td>W</td><td>p &gt; 0.1</td><td>p &gt; 0.1</td><td>p &gt; 0.1</td><td>p &gt; 0.1</td><td>p &gt; 0.1</td><td>p &gt; 0.05</td></tr><tr><td>W+F</td><td>p &gt; 0.1</td><td>p &gt; 0.1</td><td>p &gt; 0.1</td><td>p &gt; 0.1</td><td>p &lt; 0.05</td><td>p &gt; 0.05</td></tr></table>

<table><tr><td rowspan="2">Model</td><td rowspan="2">XLOGIQA</td><td rowspan="2">XHELLASWAG</td><td rowspan="2">MMMLU</td><td rowspan="2">XNLI</td><td rowspan="2">MRC</td><td colspan="2">FLORESE-200</td><td rowspan="2">MGSM</td><td rowspan="2">AVG.</td></tr><tr><td>en-xx</td><td>xx-en</td></tr><tr><td>Llama-3.1-8B</td><td>33.25</td><td>35.33</td><td>40.10</td><td>56.17</td><td>57.49</td><td>38.56</td><td>29.41</td><td>38.00</td><td>41.04</td></tr><tr><td>Mix-PT</td><td>34.75</td><td>36.68</td><td>43.05</td><td>59.17</td><td>60.36</td><td>39.63</td><td>32.72</td><td>36.96</td><td>42.92</td></tr><tr><td>CrossIC-PT</td><td>36.00</td><td>39.71</td><td>43.15</td><td>62.17</td><td>63.02</td><td>41.39</td><td>30.44</td><td>39.68</td><td>44.44</td></tr><tr><td>CrossIC-PTmix</td><td>34.75</td><td>32.33</td><td>43.55</td><td>58.33</td><td>62.20</td><td>40.75</td><td>33.53</td><td>36.96</td><td>42.80</td></tr><tr><td>CrossIC-PTreverse</td><td>35.50</td><td>33.69</td><td>43.00</td><td>57.67</td><td>62.41</td><td>39.51</td><td>34.12</td><td>36.40</td><td>42.79</td></tr></table>


Table 3: The average task results of CrossIC-PT with mix two directions $( \mathrm { C r o s s I C - P T } _ { m i x } )$ and the reverse direction $( \mathrm { C r o s s I C - P T } _ { r e v e r s e } )$ of cross-lingual in-context data.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/22a9270d-058b-4880-9777-03ba84ffaf64/86ca9176eb52509d6e2fd370af23f9de69adf50baf9f986548654dd312d0f6e4.jpg)



Figure 6: The average results of each target language model in English tasks. The p is the significant score between the CrossIC-PT model and Llama-3.1-8B.


## 5.2 Performance on English Tasks

To prevent catastrophic forgetting during continual pre-training, it’s important to ensure English per formance is maintained. To verify this, we tested the performance of six target language models on English tasks, using the same tasks as before. The results are shown in Fig.6. 

The upper part of Fig.6 shows the average performance of each target language model on English tasks, with the x-axis ordered by the performance gap between Llama-3.1-8B’s performance on the target language and English. The trend suggests that a larger performance gap corresponds to a greater impact on English performance after training. For example, the English performance of Thai (th) and Arabic (ar) is lower. However, it is primarily due to a significant drop in one task. To further investigate, the lower part of Fig.6 presents the statistical significance ("p") of the performance differences between target language models and the base model, Llama-3.1-8B, across seven tasks. The results show that, except for the Thai model trained with data augmentation (which exhibits a significant drop in English performance), there are no significant differences for other target language models. This suggests that CrossIC-PT improves performance in target languages while effectively preserving English capabilities. We believe this is likely due to the inclusion of at least 50% of English tokens in the cross-lingual in-context corpus, which helps mitigate severe forgetting. This result further validates the robustness and practicality of CrossIC-PT for cross-lingual transfer. 

## 6 Conclusion

Our work explores a special angle by focusing on semantically related multilingual in-context to enhance the cross-lingual transfer capability of LLMs. We hypothesize that concatenating semantically related English and target language corpora as Crosslingual In-context data is easily accessible and provides an implicitly cross-lingual supervision signal. Building on this hypothesis, we propose CrossIC-PT, a pre-training method based on cross-lingual in-context data. We implement our method using Wikipedia data and employ continual pre-training of existing LLMs on this data. To address the limitations posed by input window length during model training, we design a window-split strategy coupled with an optimized window sliding mechanism. Experimental results demonstrate that CrossIC-PT enhances multilingual performance across three models—Llama-3.1-8B, Qwen2.5-7B, and Qwen2.5-1.5B—across six target languages, achieving performance gains of 3.79%, 3.99%, and 1.95%, respectively, compared to base models. Further improvements are observed after data augmentation using a semantic retrieval framework. Our approach is simple to scale for multilingual LLM pre-training and offers an efficient way to expand data volume. 

## Limitations

To our knowledge, this work has the following limitations: 

• Due to resource constraints, our experiments were limited to a context window length of 4096 tokens. Longer windows could better preserve the completeness of articles and enable the concatenation of similar multilingual data from more than two languages, potentially further enhancing cross-lingual transfer. 

• Our experiments focused on validating the effectiveness of concatenated cross-lingual incontext data, so we performed continued pretraining on monolingual data rather than mixing multilingual data. While this choice aligns with our research goals, our approach also provides valuable insights for developers working on multilingual LLMs. 

• Our data expansion method, based on retrieval, currently demonstrates how to retrieve additional English data from external sources using target-language Wikipedia data. However, this approach can be easily extended to retrieve more diverse data. Wikipedia’s broad domain coverage makes it an ideal hub for retrieving both target language and English data from other sources. By controlling the retrieval process with appropriate similarity thresholds, the retrieved bilingual data can be used to construct high-quality cross-lingual in-context data. 

## Acknowledgements

This work is supported by the National Natural Science Foundation of China (No. 62376245), the Key Research and Development Program of Zhejiang Province, China (No. 2024C01034), the Fundamental Research Funds for the Central Universities (226-2024-00170), National Key Research and Development Project of China (No. 2018AAA0101900), and MOE Engineering Research Center of Digital Library, and the Alibaba Research Intern Program. 

## References



OpenAI Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, 





Shyamal Anadkat, Red Avila, Igor Babuschkin, et al. 2023. Gpt-4 technical report. 





Anthropic. The claude 3 model family: Opus, sonnet, haiku. 





Mikel Artetxe, Sebastian Ruder, and Dani Yogatama. 2020. On the cross-lingual transferability of monolingual representations. In Proceedings of the 58th Annual Meeting ofthe Associationfor Computational Linguistics, ACL 2020, pages 4623–4637. 





Terra Blevins and Luke Zettlemoyer. 2022. Language contamination helps explains the cross-lingual capabilities of english pretrained models. In Proceedings ofthe 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022, pages 3563–3574. Association for Computational Linguistics. 





Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeff Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Ma teusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. ArXiv, abs/2005.14165. 





Jonathan H. Clark, Jennimaria Palomaki, Vitaly Nikolaev, Eunsol Choi, Dan Garrette, Michael Collins, and Tom Kwiatkowski. 2020. Tydi QA: A benchmark for information-seeking question answering in typologically diverse languages. Trans. Assoc. Comput. Linguistics, 8:454–470. 





Alexis Conneau, Ruty Rinott, Guillaume Lample, Adina Williams, Samuel R. Bowman, Holger Schwenk, and Veselin Stoyanov. 2018. XNLI: evaluating crosslingual sentence representations. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, Brussels, Belgium, October 31 - November 4, 2018, pages 2475–2485. Association for Computational Linguistics. 





Marta R. Costa-jussà, James Cross, Onur Çelebi, Maha Elbayad, Kenneth Heafield, Kevin Heffernan, Elahe Kalbassi, Janice Lam, Daniel Licht, Jean Maillard, Anna Y. Sun, Skyler Wang, Guillaume Wenzek, Al Youngblood, Bapi Akula, Loïc Barrault, Gabriel Mejia Gonzalez, Prangthip Hansanti, John Hoffman, Semarley Jarrett, Kaushik Ram Sadagopan, Dirk Rowe, Shannon Spruit, Chau Tran, Pierre Andrews, Necip Fazil Ayan, Shruti Bhosale, Sergey Edunov, Angela Fan, Cynthia Gao, Vedanuj Goswami, Francisco Guzmán, Philipp Koehn, Alexandre Mourachko, Christophe Ropers, Safiyyah Saleem, Holger Schwenk, and Jeff Wang. 2022. No language left behind: Scaling human-centered machine translation. CoRR, abs/2207.04672. 





Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurélien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Rozière, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Graeme Nail, Grégoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel M. Kloumann, Ishan Misra, Ivan Evtimov, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, and et al. 2024. The llama 3 herd of models. CoRR, abs/2407.21783. 





Javier García Gilabert, Carlos Escolano, Aleix Sant Savall, Francesca de Luca Fornaciari, Audrey Mash, Xixian Liao, and Maite Melero. 2024. Investigating the translation capabilities of large language models trained on parallel data only. CoRR, abs/2406.09140. 





Gurpreet Gosal, Yishi Xu, Gokul Ramakrishnan, Rituraj Joshi, Avraham Sheinin, Zhiming Chen, Biswajit Mishra, Natalia Vassilieva, Joel Hestness, Neha Sengupta, Sunil Kumar Sahu, Bokang Jia, Onkar Pandit, Satheesh Katipomu, Samta Kamboj, Samujjwal Ghosh, Rahul Pal, Parvez Mullah, Soundar Doraiswamy, Mohamed El Karim Chami, and Preslav Nakov. 2024. Bilingual adaptation of monolingual foundation models. CoRR, abs/2407.12869. 





Shaoxiong Ji, Zihao Li, Indraneil Paul, Jaakko Paavola, Peiqin Lin, Pinzhen Chen, Dayyán O’Brien, Hengyu Luo, Hinrich Schütze, Jörg Tiedemann, et al. 2024. Emma-500: Enhancing massively multilingual adaptation of large language models. CoRR, abs/2409.17892. 





Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2019. Billion-scale similarity search with GPUs. IEEE Transactions on Big Data, 7(3):535–547. 





Peiqin Lin, Chengzhi Hu, Zheyu Zhang, André F. T. Martins, and Hinrich Schütze. 2023. mplm-sim: Bet ter cross-lingual similarity and transfer in multilin gual pretrained language models. In Findings. 





Anton Lozhkov, Loubna Ben Allal, Leandro von Werra, and Thomas Wolf. 2024. Fineweb-edu: the finest collection of educational content. 





Lingfeng Ming, Bo Zeng, Chenyang Lyu, Tianqi Shi, Yu Zhao, Xue Yang, Yefeng Liu, Yiyu Wang, Linlong Xu, Yangyang Liu, Xiaohu Zhao, Hao Wang, Heng Liu, Hao Zhou, Huifeng Yin, Zifu Shang, Haijun Li, Longyue Wang, Weihua Luo, and Kaifu Zhang. 2024. Marco-llm: Bridging languages via massive multilingual training for cross-lingual enhancement. CoRR, abs/2412.04003. 





Vaidehi Patil, Partha Pratim Talukdar, and Sunita Sarawagi. 2022. Overlap-based vocabulary generation improves cross-lingual transfer among related languages. ArXiv, abs/2203.01976. 





Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy P. Lillicrap, Jean-Baptiste Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Firat, Julian Schrittwieser, Ioannis Antonoglou, Rohan Anil, Sebastian Borgeaud, et al. 2024. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. ArXiv, abs/2403.05530. 





Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilic, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, Jonathan Tow, Alexander M. Rush, Stella Biderman, Albert Webson, Pawan Sasanka Ammanamanchi, Thomas Wang, Benoît Sagot, Niklas Muennighoff, Albert Villanova del Moral, Olatunji Ruwase, Rachel Bawden, Stas Bekman, Angelina McMillan-Major, Iz Beltagy, Huu Nguyen, Lucile Saulnier, Samson Tan, Pedro Ortiz Suarez, Victor Sanh, Hugo Laurençon, Yacine Jernite, Julien Launay, Margaret Mitchell, Colin Raffel, Aaron Gokaslan, Adi Simhi, Aitor Soroa, Alham Fikri Aji, Amit Alfassy, Anna Rogers, Ariel Kreisberg Nitzav, Canwen Xu, Chenghao Mou, Chris Emezue, Christopher Klamm, Colin Leong, Daniel van Strien, David Ifeoluwa Adelani, and et al. 2022. BLOOM: A 176b-parameter open-access multilingual language model. CoRR, abs/2211.05100. 





Freda Shi, Mirac Suzgun, Markus Freitag, Xuezhi Wang, Suraj Srivats, Soroush Vosoughi, Hyung Won Chung, Yi Tay, Sebastian Ruder, Denny Zhou, Dipanjan Das, and Jason Wei. 2023. Language models are multilingual chain-of-thought reasoners. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net. 





Vaibhav Singh, Amrith Krishna, Karthika NJ, and Ganesh Ramakrishnan. 2024. A three-pronged approach to cross-lingual adaptation with multilingual llms. CoRR, abs/2406.17377. 





ByungHoon So, Kyuhong Byun, Kyungwon Kang, and Seongjin Cho. 2022. Jaquad: Japanese question answering dataset for machine reading comprehension. CoRR, abs/2202.01764. 





Hetong Wang, Pasquale Minervini, and E. Ponti. 2024. Probing the emergence of cross-lingual alignment during llm training. ArXiv, abs/2406.13229. 





Chenxi Whitehouse, Fantine Huot, Jasmijn Bastings, Mostafa Dehghani, Chu-Cheng Lin, and Mirella Lapata. 2024. Low-rank adaptation for multilingual summarization: An empirical study. In Findings of the Associationfor Computational Linguistics: NAACL 2024, Mexico City, Mexico, June 16-21, 2024, pages 1202–1228. Association for Computational Linguistics. 





Ikuya Yamada and Ryokan Ri. 2024. LEIA: facilitating cross-lingual knowledge transfer in language models with entity-based data augmentation. In Findings of the Associationfor Computational Linguistics, ACL 2024, Bangkok, Thailand and virtual meeting, August 11-16, 2024, pages 7029–7039. Association for Computational Linguistics. 





An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. 2024. Qwen2.5 technical report. CoRR, abs/2412.15115. 





Haneul Yoo, Cheonbok Park, Sangdoo Yun, Alice Oh, and Hwaran Lee. 2024. Code-switching curriculum learning for multilingual transfer in llms. CoRR, abs/2411.02460. 





Yidan Zhang, Boyi Deng, Yu Wan, Baosong Yang, Haoran Wei, Fei Huang, Bowen Yu, Junyang Lin, and Jingren Zhou. 2024a. P-mmeval: A parallel multilingual multitask benchmark for consistent evaluation of llms. CoRR, abs/2411.09116. 





Yuanchi Zhang, Yile Wang, Zijun Liu, Shuo Wang, Xiaolong Wang, Peng Li, Maosong Sun, and Yang Liu. 2024b. Enhancing multilingual capabilities of large language models through self-distillation from resource-rich languages. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024, pages 11189–11204. Association for Computational Linguistics. 





Zhihan Zhang, Dong-Ho Lee, Yuwei Fang, Wenhao Yu, Mengzhao Jia, Meng Jiang, and Francesco Barbieri. 2024c. PLUG: leveraging pivot language in crosslingual instruction tuning. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), ACL 2024. 





Wenzhen Zheng, Wenbo Pan, Xu Xu, Libo Qin, Li Yue, and Ming Zhou. 2024. Breaking language barriers: 





Cross-lingual continual pre-training at scale. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, EMNLP 2024, Miami, FL, USA, November 12-16, 2024, pages 7725– 7738. Association for Computational Linguistics. 



## A The Setting for Evaluation

The prompts of each task we used are shown in Table 4. Since our method aims to transfer English capabilities to target languages, the prompts are primarily designed in English, and the demonstrations are also selected from English data. For the mathematical reasoning task (MGSM), we conducted an 8-shot test; for the reading comprehension task (MRC), we adopted a zero-shot setting to evaluate the model’s understanding of the target language; for other tasks, we set up a 5-shot test. For multiple-choice tasks (e.g., XNLI, MMLU, XHELLASWAG, XLOGIQA), we directly obtain answers by predicting the next logits. For other tasks, we use greedy search to generate answers and extract the final answer through regular expression matching. 

<table><tr><td>Task</td><td>Prompt</td></tr><tr><td>XLOGIQA</td><td>Passage: {context}\nQuestion: {question}\nChoices:\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD.{option_d}\nAnswer:</td></tr><tr><td>XHELLASWAG</td><td>{premise}\nOptions: \nA. {option_1}\nB. {option_2}\nC. {option_3}\nD. {option_4}\nQuestion: Which is the correct ending for the sentence from A, B, C, and D?\nAnswer:</td></tr><tr><td>MMMLU</td><td>The following is a multiple-choice question.\n\n{question}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD.{option_d}\n\nAnswer:</td></tr><tr><td>XNLI</td><td>Take the following as truth: {premise}\nThen the following statement: &quot;{hypothesis}&quot; is\nOptions:\nA. true\nB.inconclusive\nC. false\nAnswer:</td></tr><tr><td>MRC</td><td>Refer to the passage below and answer the following question:\nPassage: {context}\nQuestion:{question}\nAnswer: Based on the passage, the answer to the question is &quot;</td></tr><tr><td>FLORES-200</td><td>Translate from [source] to [target].\n[source]:</td></tr><tr><td>MGSM</td><td>Solve this math problem. Give the reasoning steps before giving the final answer on the last line by itself in the format of &quot;[The answer is ]&quot;. Do not add anything other than the integer answer after &quot;The answer is&quot;.\n{question}</td></tr></table>


Table 4: Task Prompts."[]" represents optional content. For the FLORESE task, the "[source]" indicates the source language, and "[target]" indicates the target language of translation. For MGSM, "[The answer is ]" is the translation of "The answer is " according to the test language.


For the FLORES-200 generation tasks, we employed SacreBLEU<sup>¶</sup> with the default configuration: nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.0.0. For languages lacking whitespace-based word boundaries (Japanese [ja], Korean [ko], and Thai [th]), we introduced a pre-tokenization step prior to BLEU computation, implemented as follows: 

```python
class NonASCIITokenizer(object):
    def __init__(self):
    self.is_cjk = re.compile(
    "([\u2e80-\u9fff]|"  # zh,ja,ko
    "[\ua960-\ua97f]|"  # Hangul Extended A
    "[\uac00-\ud7ff]|"  # Hangeul Syllables + Hangeul Letters Extension B
    "[\u0E00-\u0E7F]|"  # th")
    )
    def __call__(self, sent):
    sent = sent.strip()
    chs = list(sent)
    line_chtok = []
    for ch in chs:
    if self.is_cjk.match(ch):
    line_chtok.append('_')
    line_chtok.append(ch)
    line_chtok.append('-')
    else:
    line_chtok.append(ch)
    line_chtok = trim_multiple_space(line_chtok)
    return '-'.join(line_chtok) 
```

## B Results of Per-Languages


The results of our method, ablation study, and the baselines across six languages in each task are shown in Table 5.


<table><tr><td rowspan="2" colspan="2">Model</td><td colspan="8">Arabic Tasks</td></tr><tr><td>XLOGIQA</td><td>XHELLASWAG</td><td>MMMLU</td><td>XNLI</td><td>MRC</td><td>FLORES-200</td><td>MGSM</td><td>AVG</td></tr><tr><td rowspan="11">Llama-3.1-8B</td><td>base</td><td>37.50</td><td>35.34</td><td>33.25</td><td>54.17</td><td>52.84</td><td>16.20</td><td>34.40</td><td>37.67</td></tr><tr><td>LEIA</td><td><eq>33.75 \pm 1.77</eq></td><td><eq>32.76 \pm 0.70</eq></td><td><eq>34.33 \pm 0.24</eq></td><td><eq>51.39 \pm 1.04</eq></td><td><eq>53.02 \pm 1.56</eq></td><td><eq>18.55 \pm 0.17</eq></td><td><eq>35.47 \pm 0.75</eq></td><td><eq>37.04 \pm 0.49</eq></td></tr><tr><td>X-MONO-PT</td><td>40.00</td><td>29.31</td><td>36.75</td><td>54.17</td><td>61.24</td><td>17.91</td><td>34.40</td><td>39.11</td></tr><tr><td>Mix-PT</td><td>33.75</td><td>33.62</td><td>33.75</td><td>57.50</td><td>58.41</td><td>17.04</td><td>33.60</td><td>38.24</td></tr><tr><td>CrossIC-PT</td><td>40.00</td><td>32.76</td><td>33.25</td><td>60.83</td><td>59.50</td><td>17.26</td><td>34.00</td><td>39.66</td></tr><tr><td>Mix-PT</td><td>36.25</td><td>29.31</td><td>36.50</td><td>50.00</td><td>63.43</td><td>17.92</td><td>33.20</td><td>38.09</td></tr><tr><td rowspan="3">W</td><td>33.75</td><td>34.48</td><td>36.00</td><td>55.83</td><td>63.65</td><td>21.03</td><td>34.80</td><td>39.93</td></tr><tr><td>37.50</td><td>33.62</td><td>36.75</td><td>51.67</td><td>60.34</td><td>18.04</td><td>36.80</td><td>39.25</td></tr><tr><td>35.00</td><td>35.34</td><td>37.25</td><td>55.83</td><td>62.83</td><td>22.95</td><td>34.80</td><td>40.57</td></tr><tr><td>Mix-PT</td><td>33.75</td><td>33.62</td><td>36.50</td><td>55.83</td><td>68.56</td><td>17.85</td><td>35.20</td><td>40.19</td></tr><tr><td>W+F</td><td>36.25</td><td>35.34</td><td>37.25</td><td>54.17</td><td>66.81</td><td>22.82</td><td>35.60</td><td>41.18</td></tr><tr><td rowspan="3">Qwen-2.5-7B</td><td>base</td><td>50.00</td><td>54.31</td><td>45.50</td><td>57.50</td><td>67.25</td><td>16.61</td><td>65.20</td><td>50.91</td></tr><tr><td>Cross-CPT</td><td>47.50</td><td>57.76</td><td>45.25</td><td>76.67</td><td>73.47</td><td>17.09</td><td>63.60</td><td>54.48</td></tr><tr><td>CrossIC-CPT</td><td>50.00</td><td>61.21</td><td>46.00</td><td>71.67</td><td>73.58</td><td>24.15</td><td>65.20</td><td>55.97</td></tr><tr><td rowspan="3">Qwen-2.5-1.5B</td><td>base</td><td>36.25</td><td>31.03</td><td>36.00</td><td>45.83</td><td>73.25</td><td>6.87</td><td>35.60</td><td>37.83</td></tr><tr><td>Cross-CPT</td><td>40.00</td><td>30.17</td><td>38.25</td><td>49.17</td><td>72.71</td><td>5.89</td><td>30.80</td><td>38.14</td></tr><tr><td>CrossIC-CPT</td><td>41.25</td><td>35.34</td><td>37.25</td><td>53.33</td><td>76.09</td><td>7.44</td><td>30.80</td><td>40.21</td></tr></table>

<table><tr><td rowspan="2" colspan="2">Model</td><td colspan="8">Spanish Tasks</td></tr><tr><td>XLOGIQA</td><td>XHELLASWAG</td><td>MMMLU</td><td>XNLI</td><td>MRC</td><td>FLORES-200</td><td>MGSM</td><td>AVG</td></tr><tr><td rowspan="11">Llama-3.1-8B</td><td>base</td><td>36.25</td><td>36.97</td><td>41.25</td><td>60.00</td><td>54.51</td><td>25.86</td><td>43.20</td><td>42.58</td></tr><tr><td>LEIA</td><td><eq>38.75 \pm 1.77</eq></td><td><eq>43.70 \pm 0.00</eq></td><td><eq>39.08 \pm 0.51</eq></td><td><eq>63.06 \pm 2.83</eq></td><td><eq>52.69 \pm 0.34</eq></td><td><eq>25.60 \pm 0.09</eq></td><td><eq>45.33 \pm 0.82</eq></td><td><eq>44.03 \pm 0.24</eq></td></tr><tr><td>X-MONO-PT</td><td>42.50</td><td>33.61</td><td>44.00</td><td>63.33</td><td>54.60</td><td>26.04</td><td>46.40</td><td>44.35</td></tr><tr><td>Mix-PT</td><td>41.50</td><td>37.82</td><td>43.25</td><td>62.50</td><td>51.48</td><td>25.69</td><td>46.40</td><td>44.09</td></tr><tr><td>CrossIC-PT</td><td>42.50</td><td>37.82</td><td>43.25</td><td>63.33</td><td>52.49</td><td>25.80</td><td>46.80</td><td>44.57</td></tr><tr><td>Mix-PT</td><td>37.50</td><td>39.50</td><td>44.25</td><td>57.50</td><td>56.71</td><td>25.98</td><td>42.80</td><td>43.46</td></tr><tr><td rowspan="3">W</td><td>CrossIC-PT w/o Opt-SWM</td><td>40.00</td><td>42.86</td><td>43.25</td><td>57.50</td><td>59.32</td><td>27.33</td><td>42.80</td></tr><tr><td>CrossIC-PT_random</td><td>43.75</td><td>35.29</td><td>44.00</td><td>55.00</td><td>56.96</td><td>25.99</td><td>44.00</td></tr><tr><td>CrossIC-PT</td><td>36.25</td><td>43.70</td><td>44.75</td><td>60.83</td><td>58.65</td><td>27.82</td><td>46.40</td></tr><tr><td>Mix-PT</td><td>36.25</td><td>42.86</td><td>45.00</td><td>59.17</td><td>56.71</td><td>25.68</td><td>46.40</td><td>44.58</td></tr><tr><td>W+F</td><td>CrossIC-PT</td><td>38.75</td><td>47.90</td><td>43.50</td><td>62.50</td><td>63.71</td><td>27.77</td><td>44.40</td></tr><tr><td rowspan="3">Qwen-2.5-7B</td><td>base</td><td>42.50</td><td>65.55</td><td>51.50</td><td>66.67</td><td>55.02</td><td>25.35</td><td>76.40</td><td>54.71</td></tr><tr><td>Cross-CPT</td><td>47.50</td><td>63.87</td><td>52.75</td><td>80.00</td><td>66.16</td><td>25.11</td><td>75.60</td><td>58.71</td></tr><tr><td>CrossIC-CPT</td><td>42.50</td><td>68.07</td><td>51.50</td><td>82.50</td><td>69.45</td><td>27.66</td><td>74.40</td><td>59.44</td></tr><tr><td rowspan="3">Qwen-2.5-1.5B</td><td>base</td><td>38.75</td><td>46.22</td><td>48.50</td><td>50.83</td><td>51.31</td><td>20.06</td><td>51.60</td><td>43.90</td></tr><tr><td>Cross-CPT</td><td>35.00</td><td>42.02</td><td>47.00</td><td>56.67</td><td>61.18</td><td>19.49</td><td>49.20</td><td>44.37</td></tr><tr><td>CrossIC-CPT</td><td>40.00</td><td>43.70</td><td>47.50</td><td>55.83</td><td>56.46</td><td>22.12</td><td>50.00</td><td>45.09</td></tr></table>

<table><tr><td colspan="2">Model</td><td>XLOGIQA</td><td>XHELLASWAG</td><td>MMMLU</td><td>XMLI</td><td>MRC</td><td>FLORES-200</td><td>MGSM</td><td>AVG</td></tr><tr><td rowspan="11">Llama-3.1-8B</td><td>base</td><td>32.50</td><td>35.00</td><td>36.50</td><td>54.17</td><td>68.95</td><td>39.81</td><td>33.20</td><td>42.88</td></tr><tr><td>LEIA</td><td><eq>36.08 \pm 1.56</eq></td><td><eq>38.78 \pm 2.58</eq></td><td><eq>37.83 \pm 1.23</eq></td><td><eq>60.56 \pm 2.75</eq></td><td><eq>71.67 \pm 0.51</eq></td><td><eq>39.22 \pm 0.16</eq></td><td><eq>29.87 \pm 0.19</eq></td><td><eq>44.86 \pm 0.89</eq></td></tr><tr><td>X-MONO-PT</td><td>32.50</td><td>33.33</td><td>39.50</td><td>54.17</td><td>73.25</td><td>40.64</td><td>31.60</td><td>43.57</td></tr><tr><td>Mix-PT</td><td>32.50</td><td>36.67</td><td>38.75</td><td>55.00</td><td>70.50</td><td>40.04</td><td>35.20</td><td>44.09</td></tr><tr><td>CrossIC-PT</td><td>40.00</td><td>35.83</td><td>39.75</td><td>57.50</td><td>70.83</td><td>40.13</td><td>35.20</td><td>45.61</td></tr><tr><td>Mix-PT</td><td>33.75</td><td>37.50</td><td>41.00</td><td>55.83</td><td>72.57</td><td>40.63</td><td>32.40</td><td>44.81</td></tr><tr><td rowspan="3">W</td><td>35.00</td><td>40.00</td><td>39.25</td><td>59.17</td><td>75.37</td><td>41.76</td><td>35.20</td><td>46.54</td></tr><tr><td>40.00</td><td>32.50</td><td>40.00</td><td>57.50</td><td>75.17</td><td>40.65</td><td>33.60</td><td>45.63</td></tr><tr><td>35.00</td><td>40.00</td><td>39.25</td><td>61.67</td><td>77.08</td><td>42.29</td><td>35.60</td><td>47.27</td></tr><tr><td>Mix-PT</td><td>31.25</td><td>40.00</td><td>42.00</td><td>51.67</td><td>75.92</td><td>40.78</td><td>31.60</td><td>44.75</td></tr><tr><td>W+F</td><td>33.75</td><td>47.50</td><td>39.75</td><td>65.00</td><td>75.58</td><td>41.90</td><td>33.20</td><td>48.10</td></tr><tr><td rowspan="3">Qwen-2.5-7B</td><td>base</td><td>48.75</td><td>58.33</td><td>49.50</td><td>65.00</td><td>75.75</td><td>40.91</td><td>60.40</td><td>56.95</td></tr><tr><td>Cross-CPT</td><td>47.50</td><td>60.83</td><td>51.75</td><td>75.83</td><td>71.00</td><td>40.11</td><td>56.80</td><td>57.69</td></tr><tr><td>CrossIC-CPT</td><td>50.00</td><td>58.33</td><td>53.25</td><td>76.67</td><td>75.37</td><td>42.98</td><td>56.40</td><td>59.00</td></tr><tr><td rowspan="3">Qwen-2.5-1.5B</td><td>base</td><td>38.75</td><td>33.33</td><td>41.50</td><td>52.50</td><td>69.00</td><td>27.91</td><td>32.80</td><td>42.26</td></tr><tr><td>Cross-CPT</td><td>38.75</td><td>35.83</td><td>42.25</td><td>55.83</td><td>69.83</td><td>23.65</td><td>26.80</td><td>41.85</td></tr><tr><td>CrossIC-CPT</td><td>45.00</td><td>38.33</td><td>41.50</td><td>57.50</td><td>70.75</td><td>26.24</td><td>28.40</td><td>43.96</td></tr></table>


Table 5: The results of our method, ablation study, and the baselines in Arabic, Spanish and Japanese.


## C Results in Other Non-Latin Languages

To further validate the generalization capability of CrossIC-PT, we conducted additional experiments on Chinese and Vietnamese. As shown in Table 7, our method demonstrates consistent effectiveness across these non-Latin script languages. 

<table><tr><td rowspan="2" colspan="2">Model</td><td colspan="8">Korean Tasks</td></tr><tr><td>XLOGIQA</td><td>XHELLASWAG</td><td>MMMLU</td><td>XNLI</td><td>MRC</td><td>FLORES-200</td><td>MGSM</td><td>AVG</td></tr><tr><td rowspan="11">Llama-3.1-8B</td><td>base</td><td>32.50</td><td>32.50</td><td>41.50</td><td>60.00</td><td>74.91</td><td>33.60</td><td>34.00</td><td>44.14</td></tr><tr><td>LEIA</td><td><eq>36.00 \pm 2.70</eq></td><td><eq>32.28 \pm 1.04</eq></td><td><eq>41.03 \pm 0.24</eq></td><td><eq>63.83 \pm 0.68</eq></td><td><eq>74.31 \pm 0.63</eq></td><td><eq>28.98 \pm 0.34</eq></td><td><eq>35.60 \pm 0.65</eq></td><td><eq>44.58 \pm 0.46</eq></td></tr><tr><td>X-MONO-PT</td><td>35.00</td><td>28.33</td><td>42.25</td><td>63.33</td><td>74.17</td><td>33.40</td><td>33.20</td><td>44.24</td></tr><tr><td>Mix-PT</td><td>36.25</td><td>28.33</td><td>39.75</td><td>60.83</td><td>75.65</td><td>33.20</td><td>34.80</td><td>44.12</td></tr><tr><td>CrossIC-PT</td><td>38.75</td><td>32.50</td><td>40.25</td><td>65.00</td><td>75.65</td><td>34.00</td><td>36.00</td><td>46.02</td></tr><tr><td>Mix-PT</td><td>36.25</td><td>29.17</td><td>43.00</td><td>65.00</td><td>75.65</td><td>33.40</td><td>30.80</td><td>44.75</td></tr><tr><td rowspan="3">W</td><td>37.50</td><td>34.17</td><td>41.75</td><td>67.50</td><td>77.12</td><td>34.80</td><td>33.20</td><td>46.58</td></tr><tr><td>CrossIC-PT_random</td><td>38.75</td><td>32.50</td><td>42.00</td><td>65.00</td><td>76.38</td><td>34.26</td><td>31.60</td></tr><tr><td>CrossIC-PT</td><td>38.75</td><td>33.33</td><td>43.50</td><td>66.67</td><td>76.94</td><td>34.89</td><td>34.00</td></tr><tr><td>Mix-PT</td><td>36.25</td><td>27.50</td><td>42.50</td><td>65.00</td><td>76.38</td><td>33.70</td><td>30.00</td><td>44.48</td></tr><tr><td>W+F</td><td>40.00</td><td>34.17</td><td>43.00</td><td>67.50</td><td>77.12</td><td>35.05</td><td>34.40</td><td>47.32</td></tr><tr><td rowspan="3">Qwen-2.5-7B</td><td>base</td><td>48.75</td><td>59.17</td><td>48.50</td><td>61.67</td><td>79.34</td><td>30.42</td><td>60.80</td><td>55.52</td></tr><tr><td>Cross-CPT</td><td>46.25</td><td>62.50</td><td>50.00</td><td>73.33</td><td>81.55</td><td>29.67</td><td>58.40</td><td>57.39</td></tr><tr><td>CrossIC-CPT</td><td>51.25</td><td>62.50</td><td>49.00</td><td>75.00</td><td>81.55</td><td>35.51</td><td>58.40</td><td>59.03</td></tr><tr><td rowspan="3">Qwen-2.5-1.5B</td><td>base</td><td>32.50</td><td>31.67</td><td>40.00</td><td>54.17</td><td>70.48</td><td>15.84</td><td>33.60</td><td>39.75</td></tr><tr><td>Cross-CPT</td><td>35.00</td><td>27.50</td><td>39.75</td><td>63.33</td><td>72.32</td><td>12.83</td><td>25.60</td><td>39.48</td></tr><tr><td>CrossIC-CPT</td><td>35.00</td><td>31.67</td><td>41.00</td><td>66.67</td><td>72.32</td><td>16.40</td><td>27.20</td><td>41.47</td></tr></table>

<table><tr><td rowspan="2" colspan="2">Model</td><td colspan="8">Portugues Tasks</td></tr><tr><td>XLOGIQA</td><td>XHELLASWAG</td><td>MMMLU</td><td>XNLI</td><td>MRC</td><td>FLORES-200</td><td>MGSM</td><td>AVG</td></tr><tr><td rowspan="11">Llama-3.1-8B</td><td>base</td><td>33.75</td><td>40.52</td><td>42.75</td><td>56.67</td><td>49.42</td><td>44.37</td><td>46.80</td><td>44.90</td></tr><tr><td>LEIA</td><td>32.50±1.77</td><td>42.09±2.67</td><td>40.67±0.51</td><td>61.61±1.42</td><td>45.47±1.03</td><td>44.35±0.08</td><td>44.67±1.47</td><td>44.48±0.58</td></tr><tr><td>X-MONO-PT</td><td>35.00</td><td>36.21</td><td>45.25</td><td>53.33</td><td>50.25</td><td>45.07</td><td>43.20</td><td>44.04</td></tr><tr><td>Mix-PT</td><td>40.00</td><td>37.07</td><td>41.75</td><td>60.00</td><td>49.08</td><td>44.47</td><td>47.20</td><td>45.65</td></tr><tr><td>CrossIC-PT</td><td>40.00</td><td>37.93</td><td>43.25</td><td>60.00</td><td>59.83</td><td>44.62</td><td>45.60</td><td>47.32</td></tr><tr><td>Mix-PT</td><td>35.00</td><td>41.38</td><td>46.50</td><td>59.17</td><td>52.67</td><td>45.26</td><td>45.20</td><td>46.45</td></tr><tr><td rowspan="3">W</td><td>38.75</td><td>46.55</td><td>45.75</td><td>65.00</td><td>53.67</td><td>47.33</td><td>47.20</td><td>49.18</td></tr><tr><td>38.75</td><td>37.93</td><td>44.75</td><td>62.50</td><td>57.67</td><td>45.06</td><td>43.60</td><td>47.18</td></tr><tr><td>37.50</td><td>44.83</td><td>47.00</td><td>62.50</td><td>56.67</td><td>47.94</td><td>47.20</td><td>49.09</td></tr><tr><td>Mix-PT</td><td>36.25</td><td>39.66</td><td>44.00</td><td>60.00</td><td>53.92</td><td>44.84</td><td>43.60</td><td>46.04</td></tr><tr><td>W+F</td><td>37.50</td><td>44.83</td><td>47.50</td><td>65.83</td><td>56.33</td><td>47.81</td><td>46.80</td><td>49.51</td></tr><tr><td rowspan="3">Qwen-2.5-7B</td><td>base</td><td>43.75</td><td>65.52</td><td>50.50</td><td>65.00</td><td>52.92</td><td>43.36</td><td>74.40</td><td>56.49</td></tr><tr><td>Cross-CPT</td><td>47.50</td><td>64.66</td><td>52.00</td><td>80.00</td><td>59.75</td><td>42.96</td><td>75.20</td><td>60.30</td></tr><tr><td>CrossIC-CPT</td><td>50.00</td><td>67.24</td><td>52.00</td><td>80.83</td><td>60.33</td><td>48.33</td><td>72.40</td><td>61.59</td></tr><tr><td rowspan="3">Qwen-2.5-1.5B</td><td>base</td><td>40.00</td><td>42.24</td><td>46.25</td><td>48.33</td><td>49.75</td><td>32.70</td><td>51.20</td><td>44.35</td></tr><tr><td>Cross-CPT</td><td>38.75</td><td>41.38</td><td>47.50</td><td>55.83</td><td>55.25</td><td>30.73</td><td>50.00</td><td>45.63</td></tr><tr><td>CrossIC-CPT</td><td>36.25</td><td>45.69</td><td>48.75</td><td>62.50</td><td>55.67</td><td>38.86</td><td>50.00</td><td>48.25</td></tr></table>

<table><tr><td rowspan="2" colspan="2">Model</td><td colspan="8">Thai Tasks</td></tr><tr><td>XLOGIQA</td><td>XHELLASWAG</td><td>MMMLU</td><td>XNLI</td><td>MRC</td><td>FLORES-200</td><td>MGSM</td><td>AVG</td></tr><tr><td rowspan="11">Llama-3.1-8B</td><td>base</td><td>31.25</td><td>31.67</td><td>38.50</td><td>50.00</td><td>39.66</td><td>49.14</td><td>32.80</td><td>39.00</td></tr><tr><td>LEIA</td><td><eq>32.50 \pm 1.02</eq></td><td><eq>36.83 \pm 2.04</eq></td><td><eq>37.92 \pm 0.51</eq></td><td><eq>59.56 \pm 1.57</eq></td><td><eq>50.95 \pm 0.39</eq></td><td><eq>49.37 \pm 0.36</eq></td><td><eq>33.20 \pm 2.04</eq></td><td><eq>42.90 \pm 0.82</eq></td></tr><tr><td>X-MONO-PT</td><td>30.00</td><td>34.17</td><td>40.75</td><td>57.50</td><td>47.43</td><td>52.76</td><td>32.00</td><td>42.09</td></tr><tr><td>Mix-PT</td><td>31.25</td><td>29.17</td><td>39.50</td><td>51.67</td><td>51.39</td><td>51.90</td><td>35.20</td><td>41.44</td></tr><tr><td>CrossIC-PT</td><td>31.25</td><td>30.00</td><td>39.50</td><td>55.00</td><td>52.67</td><td>51.65</td><td>36.00</td><td>42.30</td></tr><tr><td>Mix-PT</td><td>31.25</td><td>35.83</td><td>40.50</td><td>58.33</td><td>44.22</td><td>52.90</td><td>33.60</td><td>42.38</td></tr><tr><td rowspan="3">W</td><td>CrossIC-PT w/o Opt-SWM</td><td>32.50</td><td>36.67</td><td>40.25</td><td>59.17</td><td>45.32</td><td>53.80</td><td>43.04</td></tr><tr><td>CrossIC-PT_random</td><td>35.00</td><td>35.83</td><td>42.00</td><td>54.17</td><td>43.29</td><td>52.92</td><td>42.29</td></tr><tr><td>CrossIC-PT</td><td>32.50</td><td>36.67</td><td>41.25</td><td>59.17</td><td>45.74</td><td>54.01</td><td>43.51</td></tr><tr><td>Mix-PT</td><td>28.75</td><td>35.00</td><td>41.25</td><td>54.17</td><td>49.79</td><td>52.97</td><td>32.40</td><td>42.05</td></tr><tr><td>W+F</td><td>CrossIC-PT</td><td>31.25</td><td>38.33</td><td>40.75</td><td>58.33</td><td>49.87</td><td>53.90</td><td>43.72</td></tr><tr><td rowspan="3">Qwen-2.5-7B</td><td>base</td><td>35.00</td><td>61.67</td><td>46.50</td><td>60.83</td><td>60.14</td><td>53.73</td><td>58.80</td><td>53.81</td></tr><tr><td>Cross-CPT</td><td>36.25</td><td>60.83</td><td>47.25</td><td>70.83</td><td>64.37</td><td>53.77</td><td>60.00</td><td>56.19</td></tr><tr><td>CrossIC-CPT</td><td>35.00</td><td>62.50</td><td>48.25</td><td>70.00</td><td>69.87</td><td>55.66</td><td>60.00</td><td>57.33</td></tr><tr><td rowspan="3">Qwen-2.5-1.5B</td><td>base</td><td>32.50</td><td>35.83</td><td>35.00</td><td>49.17</td><td>65.49</td><td>42.22</td><td>29.60</td><td>41.40</td></tr><tr><td>Cross-CPT</td><td>36.25</td><td>35.83</td><td>38.50</td><td>51.67</td><td>61.52</td><td>37.84</td><td>24.80</td><td>40.92</td></tr><tr><td>CrossIC-CPT</td><td>37.50</td><td>33.33</td><td>38.75</td><td>54.17</td><td>64.05</td><td>41.83</td><td>26.00</td><td>42.23</td></tr></table>


Table 6: The results of our method, ablation study, and the baselines in Korean, Porturguese and Thai.


<table><tr><td rowspan="2">Chinese</td><td colspan="7">Tasks</td><td rowspan="2">AVG</td></tr><tr><td>XLOGIQA</td><td>XHELLASWAG</td><td>MMMLU</td><td>XNLI</td><td>MRC</td><td>FLORES-200</td><td>MGSM</td></tr><tr><td>Llama-3.1-8B</td><td>48.75</td><td>35.09</td><td>35.75</td><td>56.67</td><td>54.60</td><td>35.67</td><td>38.00</td><td>43.50</td></tr><tr><td>Mix-PT</td><td>43.75</td><td>32.46</td><td>40.25</td><td>60.83</td><td>60.34</td><td>36.17</td><td>36.00</td><td>44.26</td></tr><tr><td>CrossIC-PT</td><td>45.00</td><td>36.97</td><td>41.00</td><td>60.83</td><td>61.77</td><td>37.37</td><td>39.60</td><td>46.08</td></tr></table>

<table><tr><td rowspan="2">Vietnamese</td><td colspan="7">Tasks</td><td rowspan="2">AVG</td></tr><tr><td>XLOGIQA</td><td>XHELLASWAG</td><td>MMMLU</td><td>XNLI</td><td>MRC</td><td>FLORES-200</td><td>MGSM</td></tr><tr><td>Llama-3.1-8B</td><td>46.25</td><td>37.50</td><td>41.75</td><td>55.83</td><td>51.31</td><td>36.69</td><td>36.80</td><td>43.73</td></tr><tr><td>Mix-PT</td><td>43.75</td><td>39.17</td><td>42.75</td><td>60.00</td><td>47.43</td><td>36.75</td><td>39.60</td><td>44.21</td></tr><tr><td>CrossIC-PT</td><td>43.75</td><td>41.38</td><td>43.50</td><td>62.50</td><td>58.73</td><td>40.09</td><td>39.20</td><td>47.02</td></tr></table>


Table 7: The results of CrossIC-PT in Chinese and Vietnamese.


## D Results in Instruction-Tuned LLMs

To comprehensively evaluate our approach, we extended experiments to instruction-tuned models using Llama-3-8B-Instruct for Thai. We maintained all default parameters except for setting the learning rate to 5e-6. As shown in Table 8, CrossIC-PT demonstrates performance limitations on Thai XLOGIQA, XHELLASWAG, and XNLI tasks, while maintaining advantages in knowledge-based (MMMLU), comprehension (MRC), and translation (FLORES-200) benchmarks. 

These results suggest that while CrossIC-PT remains effective for certain capabilities, its performance on instruction-tuned models reveals limitations potentially attributable to the divergence between finetuning and pre-training objectives. This finding indicates the need for further investigation into optimal methods for leveraging semantically related contextual corpora in instruction-tuned settings. 

<table><tr><td rowspan="2">Thai</td><td colspan="7">Tasks</td><td rowspan="2">AVG</td></tr><tr><td>XLOGIQA</td><td>XHELLASWAG</td><td>MMMLU</td><td>XNLI</td><td>MRC</td><td>FLORES-200</td><td>MGSM</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>37.50</td><td>35.00</td><td>42.25</td><td>59.17</td><td>64.47</td><td>50.86</td><td>47.60</td><td>48.12</td></tr><tr><td>Mix-PT</td><td>30.00</td><td>37.50</td><td>44.75</td><td>54.17</td><td>68.78</td><td>53.46</td><td>46.40</td><td>47.87</td></tr><tr><td>CrossIC-PT</td><td>36.25</td><td>35.00</td><td>45.00</td><td>58.33</td><td>69.96</td><td>54.25</td><td>47.20</td><td>49.43</td></tr></table>


Table 8: The results of CrossIC-PT in Thai based on Llama-3.1-8B-Instruct.


## E Extended Context Window Analysis

To thoroughly investigate the impact of context window size on model performance, we experimented with an extended context window. Specifically, we evaluated the Qwen-2.5-1.5B model on the Thai language by increasing the context window length to 8192 tokens. The results, presented in Table ??, indicate that employing an 8192-token window yielded further performance improvements. This configuration achieved an additional 1.33% gain compared to the 4096-token CrossIC-PT model, underscoring the benefits of a larger context window for processing extensive bilingual input. 

<table><tr><td>Model</td><td>XLOGIQA</td><td>XHELLASWAG</td><td>MMMLU</td><td>XNLI</td><td>MRC</td><td>FLORES-200</td><td>MGSM</td><td>AVG.</td></tr><tr><td>Qwen2.5-1.5B</td><td>32.50</td><td>35.83</td><td>35.00</td><td>49.17</td><td>65.49</td><td>42.22</td><td>29.60</td><td>41.40</td></tr><tr><td>Mix-PT</td><td>36.25</td><td>35.83</td><td>38.50</td><td>51.67</td><td>61.52</td><td>37.84</td><td>24.80</td><td>40.92</td></tr><tr><td>CrossIC-PT window_length=4096</td><td>37.50</td><td>33.33</td><td>38.75</td><td>54.17</td><td>64.05</td><td>41.83</td><td>26.00</td><td>42.23</td></tr><tr><td>CrossIC-PT window_length=8192</td><td>38.75</td><td>32.50</td><td>41.00</td><td>51.67</td><td>66.84</td><td>41.77</td><td>32.40</td><td>43.56</td></tr></table>


Table 9: Performance on Thai with Extended Context Window


This expanded experimental setup provides valuable insights into the architectural considerations for handling long-range dependencies in multilingual contexts. 