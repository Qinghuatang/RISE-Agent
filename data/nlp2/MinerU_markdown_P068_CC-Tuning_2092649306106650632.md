# CC-TUNING: A Cross-Lingual Connection Mechanism for Improving Joint Multilingual Supervised Fine-Tuning

Yangfan Ye<sup>1</sup>, Xiaocheng Feng<sup>1,2</sup>*, Zekun Yuan<sup>1</sup>, Xiachong Feng<sup>3</sup>, Libo Qin<sup>4</sup>, Lei Huang<sup>1</sup> Weitao Ma<sup>1</sup>, Yichong Huang<sup>1</sup>, Zhirui Zhang, Yunfei Lu<sup>5</sup>, Xiaohui Yan<sup>5</sup>, Duyu Tang<sup>5</sup>, Dandan Tu<sup>5</sup>, Bing Qin<sup>1,2</sup> 

<sup>1</sup>Harbin Institute of Technology <sup>2</sup>Peng Cheng Laboratory <sup>3</sup>The University of Hong Kong <sup>4</sup>Central South University <sup>5</sup>Huawei Technologies Co., Ltd {yfye,xcfeng,qinb}@ir.hit.edu.cn 

## Abstract

Current large language models (LLMs) often exhibit imbalanced multilingual capabilities due to their English-centric training corpora. To address this, existing fine-tuning approaches operating at the data-level (e.g., through data augmentation or distillation) typically introduce implicit cross-lingual alignment, overlooking the potential for more profound, latentlevel<sup>1</sup> cross-lingual interactions. In this work, we propose CC-TUNING, a novel multilingual fine-tuning paradigm that explicitly establishes a cross-lingual connection mechanism at the latent level. During training, CC-TUNING fuses the feed forward activations from both English and non-English inputs, enabling the model to benefit from both linguistic resources. This process is facilitated with a trainable Decision Maker that identifies beneficial activations. Furthermore, during inference, a Transform Matrix is utilized to simulate the cross-lingual connection under monolingual setting through representation transformation. Our experiments on six benchmarks covering 22 languages show that CC-TUNING outperforms vanilla SFT and offers a strong latent-level alternative to datalevel augmentation methods. Further analysis also highlights the practicality of CC-TUNING and the potential of latent-level cross-lingual interactions in advancing the multilingual performance of LLMs. (Code link: CC-Tuning) 

## 1 Introduction

Recent advancements in large language models (LLMs) have demonstrated exceptional capabilities in handling diverse tasks (Dong et al., 2023; Wei et al., 2022a,b; Shanahan, 2022; Zhao et al., 2023; Liu et al., 2023; Huang et al., 2025) while exhibiting promising generalizability across diverse languages (Ye et al., 2023; Qin et al., 2024; Huo et al., 2025). However, significant performance disparities persist across languages due to the overwhelming dominance of English in training corpora, making balanced multilingual proficiency an ongoing research challenge (Touvron et al., 2023; Zhang et al., 2023; Ye et al., 2024a). 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/311c80d8-b7f1-4fbe-9825-38f28559cdba/b1332390f2fed25bc244bf997bbd4128430ca593c65d55998cd8b982a804a6c5.jpg)



Figure 1: Comparison between vanilla supervised finetuning with data augmentation at data level (implicit) and our method at latent activation level (explicit).


One of the prevailing approaches towards these challenges focuses on joint multilingual supervised fine-tuning (SFT) (Ouyang et al., 2022), which refers to fine-tuning the model with supervised data spanning multiple languages. While effective in principle, these methods encounter the “curse of multilinguality” – a paradoxical phenomenon where expanding language coverage during joint training leads to performance degradation across both high- and low-resource languages (Conneau et al., 2020; Wang et al., 2020). 

To address this, current studies primarily focus on data-level interventions through parallel corpus utilization. Common strategies include: multilingual data augmentation with English-aligned parallel examples (Aharoni et al., 2019; Shaham et al., 

2024), explicit translation task formulation (Johnson et al., 2017; Tang et al., 2020), and response distillation from resource-rich languages (Zhang et al., 2024). While these methods demonstrate partial success, their reliance on implicitly introducing data-level text alignment overlooks the potential for deeper, latent-level cross-lingual interactions. 

We propose CC-TUNING, a novel multilingual fine-tuning paradigm that introduces explicit crosslingual connections at the latent activation level by fusing feed-forward activations from English and non-English languages (Figure 1). This approach is grounded in recent empirical findings highlighting the significant potential of feed-forward activations in improving model’s multilingual performance (Ye et al., 2024b). During training, our method leverages parallel bilingual inputs and incorporates a trainable Decision Maker to identify linguistically beneficial signals from auxiliary English activations, integrating them into the forward propagation of non-English inputs. Additionally, during inference, an “easy-to-learn” Transform Matrix is utilized to simulate the cross-lingual connection without the parallel bilingual inputs, ensuring the practicality of our approach. This latent-level interaction mechanism fundamentally differs from conventional data-level approaches, as it establishes direct interlingual activation connections rather than relying on statistical correlations in training data. 

To validate our approach, we conduct extensive experiments across six benchmarks encompassing both natural language understanding and generation tasks, spanning 22 languages using two representative LLMs. Our results highlight the superiority of CC-TUNING over vanilla SFT in multilingual joint learning scenarios. Besides, compared to data-level augmentation or distillation methods that leverage parallel data, CC-TUNING offers a highly effective alternative for facilitating cross-lingual interaction. Additionally, our further ablation studies and analysis also provide strong evidence of the practicality and robustness of CC-TUNING. 

## 2 Related Work

Multilingual Large Language Models. Recently, larger models such as Bloom (Scao et al., 2022), Mala-500 (Lin et al., 2024) and Aya Model (Üstün et al., 2024) have pushed multilingual performance further by leveraging the benefits of greater scale. Generally, multilingual pretraining and fine-tuning are now the two mainstream methods for improving multilingual capabilities. Models such as Sabia (Pires et al., 2023), ChineseLLaMA (Cui et al., 2023), ChineseMixtral (HIT-SCIR, 2024), PolyLM (Wei et al., 2023) and PaLM2 (Anil et al., 2023) have been developed through (continuous) pretraining with large multilingual corpora or language-specific data. Other models like BLOOMz (Muennighoff et al., 2022), m-LLaMA (Zhu et al., 2023), Camoscio (Santilli and Rodolà, 2023), Phoenix (Chen et al., 2023) and Bode (Garcia et al., 2024) have opted for leveraging multilingual or language-specific data directly during SFT stage to foster cross-lingual alignment. 

Multilingual Supervised Fine-Tuning. Multilingual SFT is an effective way to enhance the multilingual performance of LLMs. Current research often focuses on data augmentation or distillation techniques to enrich training data and improve model generalization across multiple languages. For instance, Pan et al. (2024) highlighted the importance of diverse, high-quality data for machine translation fine-tuning, while Li et al. (2023) addressed "translationese" by using Google Translate and ChatGPT for multilingual response generation. In terms of instruction tuning, Shaham et al. (2024) showed that adding multilingual examples to English-centric fine-tuning significantly boosts multilingual instruction-following, while Chen et al. (2024) demonstrated the superiority of multilingual tuning over language-specific training. Translation-based fine-tuning has been shown to enhance semantic alignment, as argued by Ranaldi et al. (2024). Similarly, Zhu et al. (2023) combined translation data, cross-lingual tasks, and scaling laws to optimize multilingual performance. Additionally, Zhang et al. (2024) proposed a selfdistillation approach leveraging LLMs’ internal capabilities in resource-rich languages to enhance multilingual performance. 

The above methods primarily focus on enriching training data with parallel data to foster implicit cross-lingual alignment. In contrast, our CC-TUNING emphasizes improving the training paradigm by explicitly incorporating cross-lingual latent interactions into the training process. 

## 3 Method

In this section, we first revisit the vanilla multilingual supervised fine-tuning paradigm, then present the training implementation of CC-TUNING and its specialized configurations during inference stage. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/311c80d8-b7f1-4fbe-9825-38f28559cdba/d0a413d22670b53869c6ba5289a4bb8362ad56683bc48f7d4db6e3a8fb452f25.jpg)



Figure 2: Overview of the cross-lingual connection mechanism in CC-TUNING. In the training stage, CC-TUNING leverages an auxiliary English input alongside the non-English input, while retaining the vanilla loss computation without introducing additional training objectives. In the inference stage, a transform matrix is used to simulate cross-lingual connection in monolingual input scenarios, eliminating the dependence on bilingual parallel input.


## 3.1 Multilingual Supervised Fine-Tuning

Multilingual supervised fine-tuning enables pretrained models to better perform downstream tasks across diverse languages through training on annotated multilingual instruction dataset $D =$ $\{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ , where N represents the size of the dataset, x<sub>i</sub> denotes the input question or instruction, and $y _ { i }$ is the corresponding expected output or response. The training process is required to minimize the following objective of negative loglikelihood of the predicted output with respect to the ground-truth response. θ denotes the parameters of the model. 

$$
\mathcal {L} _ {S F T} (\theta) = \frac {1}{N} \sum_ {i = 1} ^ {N} - \log P (y _ {i} | x _ {i}, \theta)\tag{1}
$$

Data Augmentation with Parallel Data. For the multilingual instruction dataset D, we define its corresponding English parallel data as $D _ { e n } .$ Several previous studies have explored enriching the original training data by merging these two datasets, incorporating additional translation task form data constructed from parallel pairs, or utilizing techniques such as distillation. We collectively refer to these augmented datasets as $D _ { a u g } =$ $\{ ( x _ { i } ^ { a u g } , y _ { i } ^ { a u g } ) \} _ { j = 1 } ^ { M }$ . These approaches, in essence, do not alter the SFT process; rather, they introduce additional supervised data, as illustrated below: 

$$
\begin{array}{l} \mathcal {L} _ {S F T _ {a u g}} (\theta) = \frac {1}{N} \sum_ {i = 1} ^ {N} - \log P (y _ {i} | x _ {i}, \theta) \\ \qquad + \frac {1}{M} \sum_ {j = 1} ^ {M} - \log P (y _ {i} ^ {a u g} | x _ {i} ^ {a u g}, \theta) \end{array}\tag{2}
$$

## 3.2 CC-TUNING

We will introduce cross-lingual connection mechanism in CC-TUNING in detail, focusing on its implementation during training and inference stages. 

## 3.2.1 Training with Cross-lingual Connection

Motivated by the findings in Ye et al. (2024b), which empirically demonstrate that feed-forward activations from English hold the potential to significantly enhance a model’s performance in non-English languages. The cross-lingual connection mechanism in CC-TUNING aims to incorporate the above latent interactions into the multilingual fine-tuning process, enabling the model to benefit from both English and non-English resources as the parameters are updated. 

We denote $D = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ as a multilingual supervised instruction dataset, where $x _ { i }$ represents the input question for the i-th data point and $y _ { i }$ denotes the corresponding ground-truth response. Besides, CC-TUNING requires auxiliary parallel data, $D ^ { e n } \ : = \ : \{ ( x _ { i } , x _ { i } ^ { e n } , y _ { i } ) \} _ { i = 1 } ^ { N }$ , where $\boldsymbol { x } _ { i } ^ { e n }$ is the English translation of $x _ { i }$ . Generally, the cross-lingual connection mechanism consists of two key operations: (1) adaptive decision maker and (2) latent feed forward connection. Notably, these operations are executed just before the Response Start Token (RST), which marks the beginning of the model’s response in the training template. This ensures that our operations can smoothly introduce the intervention into the response generation process. Assuming the training template is structured as “ [Input] {question} [output] {answer} ”, these operations are executed at the position that is right before the [output] token. 

Adaptive Decision Maker. Given an auxiliary input $x _ { i } ^ { e n }$ , we first pass it through the model to extract its feed-forward activations $F _ { i } ^ { e n } \in \mathbb { R } ^ { L \times d }$ $\{ f _ { i , l } ^ { e n } \} _ { l = 1 } ^ { L }$ from L decoder layers, where d is the dimensionality of the hidden states. Notably, prior research has shown that not all feed-forward activations contribute equally and some may degrade performance (Ye et al., 2024b). To mitigate this issue, we introduce a trainable linear layer $W _ { D M } \in \mathbb { R } ^ { d \times L }$ , referred to as the Decision Maker, which adaptively selects the most beneficial layer. By combining $F _ { i } ^ { e n }$ with the embedding activations $e _ { i } \in \mathbb { R } ^ { d }$ of $x _ { i } .$ , we integrate features from both English and non-English inputs. The resulting combined features are then fed into the Decision Maker along with Gumbel-Softmax (Jang et al., 2016) to achieve the identification as follows: 

$$
H _ {i} = \frac {1}{L} \sum_ {l = 1} ^ {L} (f _ {i, l} ^ {e n} + e _ {i}) \cdot W _ {D M}\tag{3}
$$

$$
f _ {i, s} ^ {e n} = \sum_ {l = 1} ^ {L} \operatorname{Gumbel-Softmax} (H _ {i}) _ {l} \cdot f _ {i, l} ^ {e n}\tag{4}
$$

where $f _ { i , s } ^ { e n } \in \mathbb { R } ^ { d }$ represents the selected activation from the s-th layer among the L decoder layers. 

Latent Feed Forward Connection. The second step aims to transfer the beneficial activation $f _ { i , s } ^ { e n }$ identified in the previous step into the forward prop agation process of non-English input. When the input $x _ { i }$ is fed into the model, let the output of all L decoders be denoted as $O _ { i } = \{ o _ { i , l } \} _ { l = 1 } ^ { L }$ , where each $o _ { i , l }$ should have been obtained by combining the feed-forward activations $f _ { i , l }$ and self-attention activations $a _ { i , l }$ through a residual connection. However, the incorporation of $f _ { i , s } ^ { e n }$ refines this process by connecting itself with the feed-forward activation $f _ { i , 1 }$ from the first decoder layer. Formally, this modification can be expressed as: 

$$
\tilde {f} _ {i, 1} = f _ {i, 1} + f _ {i, s} ^ {e n}\tag{5}
$$

The forward propagation of the input $x _ { i }$ then continues with this modification. Consequently, the original decoder outputs $\{ o _ { i , l } \} _ { l = j } ^ { L }$ will be altered to $\{ \tilde { o } _ { i , l } \} _ { l = j } ^ { L }$ due to the update of $f _ { i , 1 } \to \tilde { f } _ { i , 1 }$ , leading to new final prediction outcomes $\tilde { o } _ { i , L }$ 

And within CC-TUNING, the training objective remains the same as the vanilla loss objective in Equation 1. During the tuning process, the model itself, along with the Decision Maker, learns to leverage the benefits of both English and non-English resources, improving its multilingual capabilities. 

## 3.2.2 Inference with Transform Matrix

Unlike the training stage, our inference process is conducted without the need for parallel inputs. Instead, we leverages a training-free Transform Matrix to simulate the cross-lingual connection. 

The role of the Transform Matrix $W _ { T }$ here is to achieve the transformation of $F _ { i } = \{ f _ { i , l } \} _ { l = 1 } ^ { L } $ $F _ { i } ^ { e n } = \{ f _ { i , l } ^ { e n } \} _ { l = 1 } ^ { L }$ in the absence of parallel English input $\boldsymbol { x } _ { i } ^ { e n }$ . Specifically, after training, we firstly sample 1,000 parallel pairs $( x _ { i } , x _ { i } ^ { e n } )$ from the datasets D and $D ^ { e n }$ , and collect their feed-forward activations, $F _ { i }$ and $F _ { i } ^ { e n }$ , respectively. These activations are then stacked and denoted as $A \ =$ $\{ f _ { i , l } \ | \ i = 1 , . . . , N ; l = 1 , . . . , L \}$ and $\boldsymbol { B } = \{ f _ { i , l } ^ { e n } \ \vert$ $i = 1 , . . . , N ; l = 1 , . . . , L \}$ . Therefore, A can be mapped into $B$ as follows through $W _ { T } { : }$ 

$$
A \cdot W _ {T} = B\tag{6}
$$

To minimize the difference A and B, our objective is defined as follows (Least-Squares optimization): 

$$
\boldsymbol {W} _ {T} ^ {*} = \underset {\boldsymbol {W} _ {T}} {\operatorname{argmin}} \sum_ {i = 1} ^ {N} \sum_ {l = 1} ^ {L} \left\| f _ {i, l} \boldsymbol {W} _ {T} - f _ {i, l} ^ {e n} \right\| ^ {2}\tag{7}
$$

This problem seeks the optimal $W _ { T } ^ { * }$ that minimizes the distance between the source and target representations. Hence, the closed-form solution to this optimization problem is: 

$$
\boldsymbol {W} _ {T} ^ {*} = \left(\sum_ {i = 1} ^ {N} \sum_ {i = l} ^ {L} (f _ {i, l}) ^ {T} f _ {i, l}\right) ^ {- 1} \left(\sum_ {i = 1} ^ {N} \sum_ {i = l} ^ {L} (f _ {i, l}) ^ {T} f _ {i, l} ^ {e n}\right)\tag{8}
$$

Once the optimal $W _ { T }$ has been learned, it can be applied to the non-English representation to map it to the corresponding English representation. This resulting mapped representation $F _ { i } \cdot W _ { T }$ then substitutes $F _ { i } ^ { e n } = \{ f _ { i , l } ^ { e n } \} _ { l = 1 } ^ { L }$ in equations 3, 4, 5, thereby simulating the cross-lingual connection. This alignment effectively eliminates the dependence for bilingual parallel data and enables the simulation of cross-lingual connection in a monolingual scenario. 

## 4 Experiments

## 4.1 Setup

Models. We selected two representative LLMs: (1) LLaMA-3.1-8B (Dubey et al., 2024) and (2) Qwen2.5-7B (Yang et al., 2024). 

Training Corpus. We totally select 20,236 multilingual instruction pairs from aya dataset (Singh et al., 2024) as our training corpus and the multilingual training corpus covers more than 60 languages, ensuring extensive multilingual coverage. Our training processes are conducted on 8 * A800- SXM4-80GB with the following settings: batch size=16, epochs=3, learning rate=1.0e-5, warmup ratio=0.1, and bf16=true. The implementation is based on LLaMA-Factory (Zheng et al., 2024). 

<table><tr><td rowspan="3">Method</td><td colspan="6">Multilingual Understanding</td><td colspan="6">Multilingual Generation</td></tr><tr><td colspan="2">XNLI</td><td colspan="2">XStoryCloze</td><td colspan="2">MMMLU</td><td colspan="2">MKQA</td><td colspan="2">XQuAD</td><td colspan="2">XLSum</td></tr><tr><td>LLaMA.</td><td>Qwen.</td><td>LLaMA.</td><td>Qwen.</td><td>LLaMA.</td><td>Qwen.</td><td>LLaMA.</td><td>Qwen.</td><td>LLaMA.</td><td>Qwen.</td><td>LLaMA.</td><td>Qwen.</td></tr><tr><td colspan="13">Baselines</td></tr><tr><td>ML-SFT</td><td>31.88</td><td>48.23</td><td>65.23</td><td>70.06</td><td>40.20</td><td>50.05</td><td>14.64</td><td>14.73</td><td>60.42</td><td>63.61</td><td>12.27</td><td>12.40</td></tr><tr><td>+EN</td><td>35.02</td><td>50.76</td><td>65.13</td><td>71.63</td><td>39.62</td><td>48.80</td><td>13.28</td><td>13.05</td><td>57.40</td><td>62.34</td><td>12.04</td><td>12.20</td></tr><tr><td>+MT</td><td>35.90</td><td>47.05</td><td>69.90</td><td>70.50</td><td>40.68</td><td>47.49</td><td>13.56</td><td>13.54</td><td>58.40</td><td>64.03</td><td>12.89</td><td>12.48</td></tr><tr><td>+SDRRL</td><td>29.74</td><td>52.36</td><td>55.82</td><td>80.67</td><td>28.06</td><td>47.28</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td colspan="13">Ours</td></tr><tr><td>CC-TUNING</td><td>38.42(+6.54)</td><td>51.00(+2.77)</td><td>70.60(+5.37)</td><td>71.43(+1.37)</td><td>40.74(+0.54)</td><td>49.65(-0.40)</td><td>15.94(+1.30)</td><td>14.84(+0.11)</td><td>61.85(+1.21)</td><td>63.72(+0.11)</td><td>12.88(+0.61)</td><td>12.50(+0.10)</td></tr><tr><td>+EN</td><td>32.72(-2.30)</td><td>49.48(-1.28)</td><td>60.94(-4.19)</td><td>64.69(-6.94)</td><td>38.73(-0.89)</td><td>47.35(-1.45)</td><td>14.61(+1.33)</td><td>13.56(+0.51)</td><td>60.89(+3.40)</td><td>62.69(+0.35)</td><td>12.78(+0.74)</td><td>12.63(+0.43)</td></tr><tr><td>+MT</td><td>36.44(+0.54)</td><td>48.13(+1.08)</td><td>73.54(+3.64)</td><td>71.39(+0.89)</td><td>38.87(-1.81)</td><td>49.39(+1.90)</td><td>15.59(+2.03)</td><td>13.77(+0.23)</td><td>61.55(+3.10)</td><td>64.26(+0.23)</td><td>13.05(+0.16)</td><td>12.87(+0.39)</td></tr><tr><td>+SDRRL</td><td>29.84(+0.10)</td><td>53.06(+0.70)</td><td>69.19(+13.37)</td><td>80.93(+0.26)</td><td>37.77(+9.71)</td><td>47.87(+0.59)</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr></table>


Table 1: Main results that are the averages of the performance across all languages involved for each dataset. Blue cell indicates better performance than the vanilla ML-SFT under the same training data setting, while Gray cell indicates the opposite. Bold numbers indicate the best performance. LLaMA. and Qwen. respectively represent LLaMA-3.1-8B and Qwen2.5-7B.


Baselines. More details are in Appendix A.1. 

• ML-SFT represents vanilla supervised instruction tuning (Ouyang et al., 2022) with original multilingual instruction dataset (data size=N). 

• ML-SFT+EN incorporates the full parallel English version of the dataset for training, followed by vanilla supervised fine-tuning (data size=2N). 

• ML-SFT+MT constructs additional translation task form data by pairing the original multilingual instruction dataset with its parallel English version and then applies supervised instruction tuning (data size=2N). 

• ML-SFT+SDRRL (Zhang et al., 2024) is a self-distillation-based method that integrates English instruction tuning data and its multilingual code-switched extensions. Additionally, it incorporates partially translated data and completion data for fine-tuning (LLaMA-3.1-8B: data size 1.2N, Qwen2.5-7B: data size 1.6N). 

And CC-TUNING (+EN, +MT, +SDRRL) refers to our method applying the cross-lingual connection mechanism and its combination with different above mentioned training data settings. 

Evaluation Datasets. We conduct experiments on 6 benchmarks, which can be categorized into: 

• Multilingual Understanding: (1) XNLI (Conneau et al., 2018), a multilingual natural language inference (NLI) dataset, (2) XStoryCloze (Lin et al., 2022), a multilingual commonsense reasoning dataset for evaluating story understanding and (3) MMMLU, the multilingual version of MMLU (Hendrycks et al., 2020), designed to evaluate models’ general knowledge. 

• Multilingual Generation: (1) MKQA (Longpre et al., 2021), an open-domain multilingual question answering evaluation dataset, (2) XQuAD (Artetxe et al., 2020), a question answering dataset and (3) XLSum (Hasan et al., 2021), a multilingual abstractive summarization benchmark comprising professionally annotated articlesummary pairs. 

For each of the above datasets, we conduct experiments on 10 language subsets, covering a total of 22 languages. For XNLI, XStoryCloze, MMMLU, MKQA and XQuAD datasets, Accuracy metric is used for evaluation. And for XLSum dataset, ROUGE-L scores are reported. We use greedy decoding with a max of 40 new tokens for each model. Detailed information on the datasets and evaluations can be found in Appendix A.2. 

## 4.2 Main Results

The average results across the different languages involved in each dataset are presented in Table 1. The detailed results for different languages can be found in Table 7, 8. Note that the results of applying +SDRRL to NLG tasks are not reported, as it may lead to deviations from the prompt language in model responses, as shown in Appendix A.3. 

(1) CC-TUNING outperforms vanilla SFT in joint multilingual learning scenarios. The results in Table 1 demonstrate that under the same multilingual training data settings of original data, +MT and +SDRRL, CC-TUNING significantly outperforms vanilla SFT in both multilingual understanding and multilingual generation tasks. However, under the +EN setting, where more than half of the training data is in English, the cross-lingual connection becomes an EN2EN connection. This shift undermines the core goal of CC-TUNING—to promote cross-lingual latent interaction—leading to a notable decline in performance, which also emphasizes CC-TUNING’s alignment with its motivation and use case in joint multilingual learning scenarios. 

(2) CC-TUNING with original training data outperforms data augmentation and distillation methods on LLaMA-3.1-8B. As observed on LLaMA-3.1-8B, CC-TUNING, even when trained solely with the original dataset (data size = N), outperforms the data augmentation and distillation approaches of ML-SFT+EN (data size = 2N), +MT (data size = 2N), and +SDRRL (data size 1.2N), which utilize larger training set. This suggests that, compared to implicitly introducing crosslingual alignment information at the data level, the explicit latent-level cross-lingual connection mechanism in CC-TUNING provides a compelling alternative for facilitating cross-lingual interaction. 

## 4.3 Ablation Studies

We perform ablation studies to assess the following aspects: (1) the effectiveness of the Transform Matrix, (2) the necessity of the Decision Maker, and (3) the advantages of feed-forward activations in facilitating cross-lingual interactions. 

(1) The Transform Matrix aligns well with the effect of using parallel bilingual inputs. We verify whether the Transform Matrix $W _ { T }$ can effectively achieve the alignment by evaluating the mean squared error (MSE) between $f _ { i , l } \cdot W _ { T }$ and $f _ { i , l } ^ { e n }$ as well as the performance difference $| \Delta |$ between using parallel bilingual inputs during inference and applying the Transform Matrix. The results in Table 2 show that the MSE value reaches the order of magnitude as low as $1 0 ^ { - 2 }$ , indicating that the Transform Matrix effectively transforms $f _ { i , l }$ into $f _ { i , l } ^ { e n }$ . Additionally, the small performance difference $| \Delta |$ further suggests that the Transform Matrix serves as an effective substitute for parallel bilingual inputs, achieving great alignment. 

<table><tr><td rowspan="2">Method (|M|=1000)</td><td>XNLI</td><td>XStoryCloze</td><td>MMMLU</td><td>MKQA</td><td>XQuAD</td><td>XLSum</td></tr><tr><td colspan="6">Model: LLaMA-3.1-8B</td></tr><tr><td>MSE value</td><td></td><td colspan="5"><eq>MSE = \frac{1}{N\times L} \sum_{i=1}^{N} \sum_{l=1}^{L} \left( \frac{1}{d} \| f_{i,l} \cdot W_T - f_{i,i}^{en} \|_2^2 \right)</eq></td></tr><tr><td>CC-TUNING</td><td>0.012427</td><td>0.013256</td><td>0.013196</td><td>0.021572</td><td>0.015428</td><td>0.014314</td></tr><tr><td>+EN</td><td>0.014215</td><td>0.012734</td><td>0.012917</td><td>0.018137</td><td>0.015919</td><td>0.014046</td></tr><tr><td>+MT</td><td>0.020413</td><td>0.021251</td><td>0.023016</td><td>0.027770</td><td>0.021769</td><td>0.025639</td></tr><tr><td>+SDRRL</td><td>0.017896</td><td>0.019859</td><td>0.017098</td><td>-</td><td>-</td><td>-</td></tr><tr><td>AVG.MSE</td><td>0.016238</td><td>0.016775</td><td>0.016557</td><td>0.022493</td><td>0.017705</td><td>0.018000</td></tr><tr><td>|Δ| value</td><td colspan="6">|Δ| = |Result(Parallel Bilingual Input) - Result(Transform Matrix)|</td></tr><tr><td>CC-TUNING</td><td>0.16</td><td>0.60</td><td>0.03</td><td>0.01</td><td>0.21</td><td>0.28</td></tr><tr><td>+EN</td><td>0.08</td><td>0.31</td><td>0.25</td><td>0.01</td><td>0.08</td><td>0.16</td></tr><tr><td>+MT</td><td>0.01</td><td>0.23</td><td>0.36</td><td>0.12</td><td>0.07</td><td>0.10</td></tr><tr><td>+SDRRL</td><td>0.06</td><td>0.56</td><td>0.11</td><td>-</td><td>-</td><td>-</td></tr><tr><td>AVG.|Δ|</td><td>0.08</td><td>0.43</td><td>0.19</td><td>0.05</td><td>0.12</td><td>0.18</td></tr></table>


Table 2: The results of mean squared error between feedforward representations in English and the transformed representations after applying the Transform Matrix, as well as the performance difference $| \Delta |$ between using parallel bilingual inputs and applying Transform Matrix.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/311c80d8-b7f1-4fbe-9825-38f28559cdba/6fa84a1c0dead014676e1cf892f900e3a59b306381c44d9a3851be09d3c7fa8a.jpg)



Figure 3: Performance comparisons of using Decision Maker, Mean Pooling and Random Pooling strategy on XNLI and MKQA datasets.


(2) The Decision Maker plays a crucial role. To verify the necessity of the Decision Maker, we replaced it with two alternative strategies—Mean Pooling and Random Pooling—during both training and inference, and compared their performance in Figure 3. In Mean Pooling, the feed-forward activations from all layers are averaged, while in Random Pooling, a single activation is randomly selected from the set of feed-forward activations across all layers. The results demonstrate that the performance with the Decision Maker significantly outperforms the other two strategies, confirming that the Decision Maker effectively serves its role in beneficial activation identification and contributes to the overall training paradigm of CC-TUNING. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/311c80d8-b7f1-4fbe-9825-38f28559cdba/70dfe515c7c4b17476a534a7ff0499dc2e2960873e0ddc169e615c07050db645.jpg)



Figure 4: Performance comparisons of utilizing feed forward activations, self-attention activations and whole decoder block activations for cross-lingual connection on XNLI and MKQA datasets.


(3) Feed-forward activations contribute the most in cross-lingual connection. In addition to investigating cross-lingual connections at the feedforward activation level, we also explored the potential contributions of self-attention activations and whole decoder block activations. Our results, as shown in Figure 4, indicate that feed-forward activations have the most pronounced impact on cross-lingual connections within the CC-Tuning paradigm. This finding highlights the crucial role of feed-forward activations in facilitating crosslingual latent interactions, which well match the findings presented in Dai et al. (2022), where FFN stores factual knowledge, as well as the motivation of cross-lingual feed forward transplantation operation in Ye et al. (2024b). 

## 5 Further Analysis

## 5.1 Practicality Analysis

(1) Is the Transform Matrix difficult to learn? Figure 5 presents the variation in MSE values between $f _ { i , l } \cdot W _ { T }$ and $f _ { i , l } ^ { e n }$ as the amount of parallel data, M , used to acquire the Transform Matrix increases. We observe that when $| M | = 1 0 0 0$ , the MSE value starts to converge between 0.01 and 0.02, and subsequently exhibits a stable trend. This indicates that only a thousand of parallel data are sufficient to effectively align $f _ { i , l }$ with $f _ { i , l } ^ { e n }$ through the Transform Matrix, suggesting that the Transform Matrix is relatively easy to learn. 

(2) Does incorporating cross-lingual connection substantially interfere with model training and model inference? During training, as shown in 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/311c80d8-b7f1-4fbe-9825-38f28559cdba/bea04d9cfeac217e2257ecfe2f8d7f9e6f09b8728b0a76dfaae8ea2f9642b7a6.jpg)



Figure 5: The curves of mean squared error between feed-forward representations in English and the transformed representations after applying the Transform Matrix, as the amount of parallel data used to acquire the Transform Matrix increases.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/311c80d8-b7f1-4fbe-9825-38f28559cdba/f9f0bb65586b30883977df4f97dc592263e785fafcb30100c8ffb3b73d9b0ce6.jpg)



Figure 6: The training loss curves of vanilla supervised fine-tuning and CC-TUNING under different training settings (models and training data).


Figure 6, the loss curves of vanilla SFT and CC-TUNING are closely aligned, suggesting that the incorporation of cross-lingual connection on top of vanilla SFT introduces only negligible interference to the overall training process. This is primarily because no additional training objectives are introduced. In terms of training overhead, our statistics show that the training time for CC-TUNING is approximately 1.12 1.16 times that of vanilla SFT (Table 4). Moreover, the additional linear layer Decision Maker accounts for only 0.0016% and 0.0013% of the total parameter count in LLaMA-3.1-8B and Qwen2.5-7B, respectively—proportions so small that they are practically negligible. During inference, the time cost for inference with the Transform Matrix is also approximately 1.1 times that of vanilla inference (Table 5). 


LLaMA-3.1-8B + NLG tasks (MKQA, XQuAD, XLSum)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/311c80d8-b7f1-4fbe-9825-38f28559cdba/5e75be3f275ab7040ddaa90f1978c5218c9f8bf7ce9a222750a55f77cd5ddea5.jpg)



Figure 7: t-SNE visualizations of output representations by LLaMA-3.1-8B before fine-tuning, after vanilla supervised fine-tuning and after CC-TUNING.


## 5.2 Multilingual Representation Analysis

To analyze the impact of CC-TUNING on multilingual representations, we employ t-SNE (Van der Maaten and Hinton, 2008) to visualize the representations of 200 sentences sampled from XNLI in parallel across English, Arabic, and Chinese. 

As depicted in Figure 7 (c), after applying CC-TUNING, the multilingual representations show a significantly more compact clustering. This indicates that CC-TUNING has already facilitated a certain level of cross-lingual interaction through the cross-lingual connection mechanism, allowing the multilingual representations after CC-TUNING require less extensive sharing with representations from other languages in high-dimensional space. And the boundaries between different language representations become more distinct, suggesting that CC-TUNING alleviates the mutual dependency between representations of different languages, enabling the model to exhibit clearer and more distinct multilingual modeling capabilities. 

## 5.3 Beneficial Layer Distribution Analysis

In this section, we present the distribution of the layer with the highest probability of being selected by the Decision Maker across NLU and NLG tasks, as shown in Figure 8. This analysis explores layerwise effectiveness within the cross-lingual connection. The distribution results indicate that LLMs tend to predominantly utilize the middle layers for both NLU and NLG tasks (LLaMA-3.1-8B: 19; Qwen2.5-7B: 17), which suggests that the middle layers may capture more valuable and generalized knowledge, potentially acting as a bridge between representations in different languages. Additionally, we observe that the beneficial layers identified in NLG tasks are more diverse, likely due to the inherent complexity of generation tasks. In contrast, NLU tasks—primarily focused on selecting from predefined options (e.g., A, B, C, or D)—are less complex, and thus, the layer distribution tend to be more concentrated 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/311c80d8-b7f1-4fbe-9825-38f28559cdba/5f47b07eef2acc5afc5d5ccd75135ccc80b69197d1f90c12897067f1dd3b6ed3.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/311c80d8-b7f1-4fbe-9825-38f28559cdba/cbd2884836589b38b3fb7b772126cc264d2bfe84c7e87a9c710bc385edd67e20.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/311c80d8-b7f1-4fbe-9825-38f28559cdba/6e26889b98b0b944900d24b716c1aaefa418afe8ec05dd412ff9cb7712b75be2.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/311c80d8-b7f1-4fbe-9825-38f28559cdba/c3da0ab5e18b53831f30f291582af17041e1c86601f625df29c71d9d47abd5aa.jpg)



Figure 8: The distribution of the layer with the highest probability of being selected after the Decision Maker over NLU and NLG tasks.


## 5.4 Performance on Cross-lingual Task

We further conduct additional experiments evaluating the performance of CC-TUNING in crosslingual scenarios on XQuAD under following settings: (1) “X-to-English”: the question is given in language X, and the model is explicitly prompted to respond in English. (2) “English-to-X”: the question is given in English, and the model is explicitly prompted to respond in language X. 

The results in Table 3 show that CC-TUNING outperforms vanilla ML-SFT in both settings, highlighting its effectiveness in cross-lingual scenarios. The advantage is more pronounced in the X-to-English” setting, where the model is given non-English questions. This aligns with the motivation behind CC-TUNING: the models can benefit more when processing non-English inputs by leveraging the latent activations from English. Moreover, the performance gains on English questions under “English-to-X” setting are relatively smaller, which is also consistent with the observations under +EN setting in Table 1. 

<table><tr><td>XQuAD (Ask in X, Answer in English)</td><td>en</td><td>ar</td><td>de</td><td>el</td><td>hi</td><td>ru</td><td>th</td><td>tr</td><td>vi</td><td>zh</td><td>AVG</td></tr><tr><td>ML-SFT</td><td>72.61</td><td>15.29</td><td>30.84</td><td>21.60</td><td>12.18</td><td>15.21</td><td>18.82</td><td>25.38</td><td>33.87</td><td>13.95</td><td>25.97</td></tr><tr><td>CC-TUNING</td><td>74.62</td><td>18.82</td><td>35.71</td><td>24.54</td><td>12.27</td><td>17.14</td><td>25.29</td><td>27.23</td><td>35.21</td><td>17.90</td><td>28.87</td></tr><tr><td>XQuAD (Ask in English, Answer in X)</td><td>en</td><td>ar</td><td>de</td><td>el</td><td>hi</td><td>ru</td><td>th</td><td>tr</td><td>vi</td><td>zh</td><td>AVG</td></tr><tr><td>ML-SFT</td><td>72.61</td><td>20.34</td><td>40.50</td><td>18.91</td><td>21.09</td><td>20.08</td><td>17.48</td><td>29.24</td><td>31.93</td><td>27.48</td><td>29.97</td></tr><tr><td>CC-TUNING</td><td>75.29</td><td>18.99</td><td>40.34</td><td>20.67</td><td>23.36</td><td>20.34</td><td>19.41</td><td>29.33</td><td>33.45</td><td>28.49</td><td>30.97</td></tr></table>


Table 3: Results on the cross-lingual QA task with LLaMA-3.1-8B. The symbol X refers to either the input prompt language or the required response language, as specified by the corresponding configuration


## 5.5 Language Confusion Analysis

Language confusion refers to the cases where the model fails to consistently response in the user’s desired language, or the appropriate language given the context. Here we employ the lid.176.bin model from fastText, which can identify 176 languages, to evaluate the alignment between model responses and input languages. 

The results in Table 6 show that language confusion phenomenon frequently occured in baseline SDRRL. Since SDRRL is designed to facilitate knowledge distillation from resource-rich to lowresource languages, the training data under this setup often contains inconsistencies between input and output languages. While this issue is partially mitigated through code-switching and the integration of external parallel corpora, we observed that it still frequently causes deviations from the prompt language in model responses, making SDRRL less suitable for generation tasks. In contrast, CC-TUNING, along with other baselines, do not exhibit significant language confusions. 

## 6 Conclusion

In this paper, we propose CC-TUNING, a novel multilingual fine-tuning paradigm that establishes a cross-lingual connection mechanism at latent level to address the imbalanced multilingual capabilities of current LLMs. During training, CC-TUNING fuses the feed forward activations from both English and non-English inputs, enabling the model to benefit from both languages. During inference, we simulate the cross-lingual connection using only monolingual input through representation transformation techniques. Extensive experiments across six benchmarks covering 22 languages demonstrate that CC-TUNING outperforms vanilla supervised fine-tuning and serves as a strong latent-level alternative to data-level augmentation approaches. Our results also highlight the importance of rethinking multilingual training paradigms beyond superficial data manipulation, suggesting that deeper architectural interventions may unlock greater potential in LLMs’ multilingual capabilities. 

## Limitations

This work exhibits several limitations worth noting. Firstly, though several ablation experiments are conducted to validate the benefits of our training paradigm, we believe there is much more to explore and investigate in latent cross-lingual interactions. Such interactions should not only be limited to the form discussed in our work. Secondly, our experiments were conducted on LLaMA-3.1- 8B and Qwen2.5-7B. While these models represent important milestones in open-source LLM development, the evaluation across more LLMs would improve the generalizability of our findings across the broader LLM ecosystem. Thirdly, due to the computational constraints, we did not conduct comparisons between LLMs of different model sizes (particularly larger models), resulting in a lack of insights into the impact of model capacity on performance. 

## Acknowledgements

Xiaocheng Feng is the corresponding author of this work. We thank the anonymous reviewers for their insightful comments. This work was supported by the National Natural Science Foundation of China (NSFC) (grant 62276078, U22B2059), the Key R&D Program of Heilongjiang via grant 2022ZX01A32, and the Fundamental Research Funds for the Central Universities (Grant No.HIT.OCEF.2023018). We also thank Huawei Technologies Co., Ltd for supporting part of the computing resources and funding. 

## References



Roee Aharoni, Melvin Johnson, and Orhan Firat. 2019. Massively multilingual neural machine translation. 





In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 3874–3884, Minneapolis, Minnesota. Association for Computational Linguistics. 





Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. 2023. Palm 2 technical report. arXiv preprint arXiv:2305.10403. 





Mikel Artetxe, Sebastian Ruder, and Dani Yogatama. 2020. On the cross-lingual transferability of monolingual representations. In Proceedings of the 58th Annual Meeting ofthe Associationfor Computational Linguistics, pages 4623–4637, Online. Association for Computational Linguistics. 





Pinzhen Chen, Shaoxiong Ji, Nikolay Bogoychev, Andrey Kutuzov, Barry Haddow, and Kenneth Heafield. 2024. Monolingual or multilingual instruction tuning: Which makes a better alpaca. In Findings ofthe Association for Computational Linguistics: EACL 2024, pages 1347–1356, St. Julian’s, Malta. Association for Computational Linguistics. 





Zhihong Chen, Feng Jiang, Junying Chen, Tiannan Wang, Fei Yu, Guiming Chen, Hongbo Zhang, Juhao Liang, Chen Zhang, Zhiyi Zhang, et al. 2023. Phoenix: Democratizing chatgpt across languages. arXiv preprint arXiv:2304.10453. 





Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 2020. Unsupervised cross-lingual representation learning at scale. In Proceedings of the 58th Annual Meeting of the Associationfor Computational Linguistics, pages 8440– 8451, Online. Association for Computational Linguistics. 





Alexis Conneau, Ruty Rinott, Guillaume Lample, Adina Williams, Samuel R. Bowman, Holger Schwenk, and Veselin Stoyanov. 2018. Xnli: Evaluating crosslingual sentence representations. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics. 





Yiming Cui, Ziqing Yang, and Xin Yao. 2023. Efficient and effective text encoding for chinese llama and alpaca. arXiv preprint arXiv:2304.08177. 





Damai Dai, Li Dong, Yaru Hao, Zhifang Sui, Baobao Chang, and Furu Wei. 2022. Knowledge neurons in pretrained transformers. In Proceedings ofthe 60th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 8493– 8502, Dublin, Ireland. Association for Computational Linguistics. 





Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Zhiyong Wu, Baobao Chang, Xu Sun, Jingjing Xu, and 





Zhifang Sui. 2023. A survey for in-context learning. ArXiv preprint, abs/2301.00234. 





Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783. 





Gabriel Lino Garcia, Pedro Henrique Paiola, Luis Henrique Morelli, Giovani Candido, Arnaldo Cândido Júnior, Danilo Samuel Jodas, Luis Afonso, Ivan Rizzo Guilherme, Bruno Elias Penteado, and João Paulo Papa. 2024. Introducing bode: A finetuned large language model for portuguese promptbased task. arXiv preprint arXiv:2401.02909. 





Tahmid Hasan, Abhik Bhattacharjee, Md. Saiful Islam, Kazi Mubasshir, Yuan-Fang Li, Yong-Bin Kang, M. Sohel Rahman, and Rifat Shahriyar. 2021. XLsum: Large-scale multilingual abstractive summarization for 44 languages. In Findings ofthe Association for Computational Linguistics: ACL-IJCNLP 2021, pages 4693–4703, Online. Association for Computational Linguistics. 





Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2020. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300. 





HIT-SCIR. 2024. Chinese-mixtral-8x7b: An opensource mixture-of-experts llm. https://github. com/HIT-SCIR/Chinese-Mixtral-8x7B. 





Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, et al. 2025. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. ACM Transactions on Information Systems, 43(2):1– 55. 





Wenshuai Huo, Xiaocheng Feng, Yichong Huang, Chengpeng Fu, Baohang Li, Yangfan Ye, Zhirui Zhang, Dandan Tu, Duyu Tang, Yunfei Lu, et al. 2025. Enhancing non-english capabilities of englishcentric large language models through deep supervision fine-tuning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 24185–24193. 





Eric Jang, Shixiang Gu, and Ben Poole. 2016. Categorical reparameterization with gumbel-softmax. arXiv preprint arXiv:1611.01144. 





Melvin Johnson, Mike Schuster, Quoc V. Le, Maxim Krikun, Yonghui Wu, Zhifeng Chen, Nikhil Thorat, Fernanda Viégas, Martin Wattenberg, Greg Corrado, Macduff Hughes, and Jeffrey Dean. 2017. Google‘s multilingual neural machine translation system: Enabling zero-shot translation. Transactions ofthe Associationfor Computational Linguistics, 5:339–351. 





Haonan Li, Fajri Koto, Minghao Wu, Alham Fikri Aji, and Timothy Baldwin. 2023. Bactrian-x: 





Multilingual replicable instruction-following models with low-rank adaptation. arXiv preprint arXiv:2305.15011. 





Peiqin Lin, Shaoxiong Ji, Jörg Tiedemann, André FT Martins, and Hinrich Schütze. 2024. Mala-500: Massive language adaptation of large language models. arXiv preprint arXiv:2401.13303. 





Xi Victoria Lin, Todor Mihaylov, Mikel Artetxe, Tianlu Wang, Shuohui Chen, Daniel Simig, Myle Ott, Naman Goyal, Shruti Bhosale, Jingfei Du, Ramakanth Pasunuru, Sam Shleifer, Punit Singh Koura, Vishrav Chaudhary, Brian O’Horo, Jeff Wang, Luke Zettlemoyer, Zornitsa Kozareva, Mona Diab, Veselin Stoyanov, and Xian Li. 2022. Few-shot learning with multilingual generative language models. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 9019–9052, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics. 





Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig. 2023. Pretrain, prompt, and predict: A systematic survey of prompting methods in natural language processing. ACM Computing Surveys, 55(9):1–35. 





Shayne Longpre, Yi Lu, and Joachim Daiber. 2021. MKQA: A linguistically diverse benchmark for multilingual open domain question answering. Transactions ofthe Associationfor Computational Linguistics, 9:1389–1406. 





Niklas Muennighoff, Thomas Wang, Lintang Sutawika, Adam Roberts, Stella Biderman, Teven Le Scao, M Saiful Bari, Sheng Shen, Zheng-Xin Yong, Hailey Schoelkopf, et al. 2022. Crosslingual generalization through multitask finetuning. arXiv preprint arXiv:2211.01786. 





Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744. 





Xingyuan Pan, Luyang Huang, Liyan Kang, Zhicheng Liu, Yu Lu, and Shanbo Cheng. 2024. G-DIG: Towards gradient-based DIverse and hiGh-quality instruction data selection for machine translation. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 15395–15406, Bangkok, Thailand. Association for Computational Linguistics. 





Ramon Pires, Hugo Abonizio, Thales Sales Almeida, and Rodrigo Nogueira. 2023. Sabiá: Portuguese large language models. In Brazilian Conference on Intelligent Systems, pages 226–240. Springer. 





Libo Qin, Qiguang Chen, Yuhang Zhou, Zhi Chen, Yinghui Li, Lizi Liao, Min Li, Wanxiang Che, and Philip S Yu. 2024. Multilingual large language model: A survey of resources, taxonomy and frontiers. arXiv preprint arXiv:2404.04925. 





Leonardo Ranaldi, Giulia Pucci, and Andre Freitas. 2024. Empowering cross-lingual abilities of instruction-tuned large language models by translation-following demonstrations. In Findings of the Associationfor Computational Linguistics: ACL 2024, pages 7961–7973, Bangkok, Thailand. Association for Computational Linguistics. 





Andrea Santilli and Emanuele Rodolà. 2023. Camoscio: an italian instruction-tuned llama. Preprint, arXiv:2307.16456. 





Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilic, Daniel Hesslow, Roman´ Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, et al. 2022. Bloom: A 176bparameter open-access multilingual language model. arXiv preprint arXiv:2211.05100. 





Uri Shaham, Jonathan Herzig, Roee Aharoni, Idan Szpektor, Reut Tsarfaty, and Matan Eyal. 2024. Multilingual instruction tuning with just a pinch of multilinguality. In Findings ofthe Associationfor Computational Linguistics: ACL 2024, pages 2304–2317, Bangkok, Thailand. Association for Computational Linguistics. 





Murray Shanahan. 2022. Talking about large language models. ArXiv preprint, abs/2212.03551. 





Shivalika Singh, Freddie Vargus, Daniel Dsouza, Börje F. Karlsson, Abinaya Mahendiran, Wei-Yin Ko, Herumb Shandilya, Jay Patel, Deividas Mataciunas, Laura OMahony, Mike Zhang, Ramith Hettiarachchi, Joseph Wilson, Marina Machado, Luisa Souza Moura, Dominik Krzeminski, Hakimeh´ Fadaei, Irem Ergün, Ifeoma Okoh, Aisha Alaagib, Oshan Mudannayake, Zaid Alyafeai, Vu Minh Chien, Sebastian Ruder, Surya Guthikonda, Emad A. Alghamdi, Sebastian Gehrmann, Niklas Muennighoff, Max Bartolo, Julia Kreutzer, Ahmet Üstün, Marzieh Fadaee, and Sara Hooker. 2024. Aya dataset: An open-access collection for multilingual instruction tuning. Preprint, arXiv:2402.06619. 





Yuqing Tang, Chau Tran, Xian Li, Peng-Jen Chen, Naman Goyal, Vishrav Chaudhary, Jiatao Gu, and Angela Fan. 2020. Multilingual translation with extensible multilingual pretraining and finetuning. arXiv preprint arXiv:2008.00401. 





Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971. 





Ahmet Üstün, Viraat Aryabumi, Zheng-Xin Yong, Wei-Yin Ko, Daniel D’souza, Gbemileke Onilude, Neel Bhandari, Shivalika Singh, Hui-Lee Ooi, Amr Kayid, et al. 2024. Aya model: An instruction finetuned open-access multilingual language model. arXiv preprint arXiv:2402.07827. 





Laurens Van der Maaten and Geoffrey Hinton. 2008. Visualizing data using t-sne. Journal of machine learning research, 9(11). 





Zirui Wang, Zachary C. Lipton, and Yulia Tsvetkov. 2020. On negative interference in multilingual models: Findings and a meta-learning treatment. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 4438–4450, Online. Association for Computational Linguistics. 





Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, et al. 2022a. Emergent abilities of large language models. ArXiv preprint, abs/2206.07682. 





Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. 2022b. Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems, 35:24824–24837. 





Xiangpeng Wei, Haoran Wei, Huan Lin, Tianhao Li, Pei Zhang, Xingzhang Ren, Mei Li, Yu Wan, Zhiwei Cao, Binbin Xie, et al. 2023. Polylm: An open source polyglot large language model. arXiv preprint arXiv:2307.06018. 





An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tian hao Li, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. 2024. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115. 





Jiacheng Ye, Xijia Tao, and Lingpeng Kong. 2023. Language versatilists vs. specialists: An empirical revisiting on multilingual transfer ability. arXiv preprint arXiv:2306.06688. 





Yangfan Ye, Xiachong Feng, Xiaocheng Feng, Weitao Ma, Libo Qin, Dongliang Xu, Qing Yang, Hongtao Liu, and Bing Qin. 2024a. GlobeSumm: A challenging benchmark towards unifying multi-lingual, cross-lingual and multi-document news summarization. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, pages 10803–10821, Miami, Florida, USA. Association for Computational Linguistics. 





Yangfan Ye, Xiaocheng Feng, Xiachong Feng, Libo Qin, Yichong Huang, Lei Huang, Weitao Ma, Zhirui Zhang, Yunfei Lu, Xiaohui Yan, et al. 2024b. Xtransplant: A probe into the upper bound performance of multilingual capability and culture adaptability in llms via mutual cross-lingual feed-forward transplantation. arXiv preprint arXiv:2412.12686. 





Xiang Zhang, Senyu Li, Bradley Hauer, Ning Shi, and Grzegorz Kondrak. 2023. Don’t trust ChatGPT when your question is not in English: A study of multilingual abilities and types of LLMs. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 7915–7927, Singapore. Association for Computational Linguistics. 





Yuanchi Zhang, Yile Wang, Zijun Liu, Shuo Wang, Xiaolong Wang, Peng Li, Maosong Sun, and Yang Liu. 2024. Enhancing multilingual capabilities of large language models through self-distillation from resource-rich languages. arXiv preprint arXiv:2402.12204. 





Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, et al. 2023. A survey of large language models. ArXiv preprint, abs/2303.18223. 





Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, Zhangchi Feng, and Yongqiang Ma. 2024. Llamafactory: Unified efficient fine-tuning of 100+ language models. In Proceedings of the 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 3: System Demonstrations), Bangkok, Thailand. Association for Computational Linguistics. 





Wenhao Zhu, Yunzhe Lv, Qingxiu Dong, Fei Yuan, Jingjing Xu, Shujian Huang, Lingpeng Kong, Jiajun Chen, and Lei Li. 2023. Extrapolating large language models to non-english by aligning languages. arXiv preprint arXiv:2308.04948. 



## A Experiment Details

## A.1 Baselines Settings

This section introduces the details of different training data settings. 

• +EN combines the original multilingual dataset D with its translated parallel English dataset $D ^ { e n }$ resulting in a total training dataset size of N + N = 2N. 

• +MT constructs additional translation task form data by pairing the original multilingual dataset D with its translated parallel English dataset D<sup>en</sup> as follows: 

```json
{
    "instruction": "Translate the following sentence from English to Spanish.\n The category corresponds to politics.",
    "output": "La categoría corresponde a política. "
} 
```

N pairs of parallel data from D and D<sup>en</sup> can be constructed into N additional samples of translation task form data, resulting in a total training dataset size of N + N = 2N. 

• +SDRRL (Zhang et al., 2024) is a selfdistillation-based method that integrates English instruction tuning data and its multilingual codeswitched extensions. Additionally, it incorporates partially translated data and completion data for fine-tuning (LLaMA-3.1-8B: data size 1.2N, Qwen2.5-7B: data size 1.6N). 

## A.2 Datasets and Evaluations

## A.2.1 Datasets

The language subsets used in the 6 evaluation datasets involved in our experiments and the data size used for each language subset are as follows: 

```txt
Involved Languages (10 languages each dataset)
XNLI: en, ar, el, hi, ru, sw, th, tr, ur, zh
XStoryCloze: en, ar, es, eu, hi, id, ru, sw, te, zh
MMMLU: en, ar, bn, es, hi, id, ko, pt, sw, yo
XQuAD: en, ar, de, el, hi, ru, th, tr, vi, zh
MKQA: en, ar, de, ja, ko, pt, ru, tr, vi, zh
XLSum: en, ar, fr, hi, id, ru, sw, tr, ur, vi
A total of 22 unique languages are involved 
```

```txt
Questions (ask in Chinese)

1. "...\n问题：黑豹队的防守丢了多少分？\n您的答案：
2. "...\n问题：贾里德在职业生涯中有多少次擒杀？\n您的答案："
3. "...\n问题：卢克·坎克利贡献了多少次擒抱？\n您的答案："
4. "...\n问题：约什·诺曼拦截了多少球？\n您的答案："
5. "...\n问题：本赛季谁为球队贡献的擒杀最多？\n您的答案："
6. "...\n问题：2015年黑豹队的防守有多少次拦截记录？\n您的答案："
7. "...\n问题：谁带领黑豹队擒杀？\n您的答案："
8. "...\n问题：有多少名黑豹队防守球员入选了职业碗？\n您的答案："
9. "...\n问题：托马斯·戴维斯有多少次迫使掉球？\n您的答案："
10. "...\n问题：本赛季哪个球员拦截次数最多？\n您的答案："
11. ...
Answers

1. "The answer is 308 points."
2. "Jared Allen has 136 career sacks."
3. "在他们身后，黑豹队的三名首发线卫中有两人入选了职业碗：托马斯·戴维斯和卢克·坎克"
4. "四次"
5. "Jared Allen"
6. "The answer is 24."
7. "Jared Allen"
8. "11人"
9. "Thomas Davis forced four fumbles."
10. "The answer is: Josh Norman"
11. ...
```


Figure 9: Examples of the deviations from the prompt language in model responses when applying +SDRRL.


## Sample Size

XNLI: $1000 \times 10 = 10000$ (parallel)  
XStoryCloze: $1511 \times 10 = 15110$ (parallel)  
MMMLU: $1000 \times 10 = 10000$ (parallel)  
MKQA: $1000 \times 10 = 10000$ (parallel)  
XQuAD: $1190 \times 10 = 11900$ (parallel)  
XLSum: $100 \times 10 = 1000$ (non-parallel) 

## A.2.2 Evaluations

XNLI, XStoryCloze, and MMMLU all belong to the multiple-choice category. For these datasets, a model’s response is considered correct only if it contains the correct option and excludes all other options. For the short QA generative dataset MKQA and XQuAD, a model’s answer is deemed correct if the gold answer appears in the model’s response. 

## A.3 Model Responses with +SDRRL

The results of applying +SDRRL to NLG tasks are not reported in the main body, as it may lead to deviations from the prompt language in model responses. Since +SDRRL aims to achieve distillation from resource-rich languages to low-resource languages, many of the training data’s input and output languages under this setup are inconsistent. Although this issue is partially mitigated through code-switching and the incorporation of external parallel corpora, we still observed that it easily leads to deviations from the prompt language in model responses, making it unsuitable for NLG tasks. As in the examples shown in Figure 9, only 3 of the 10 questions given are correctly answered in Chinese, while the rest are all answered in English. 

<table><tr><td>Training Time Cost (h:m:s)</td><td>LLaMA-3.1-8B</td><td>Qwen2.5-7B</td></tr><tr><td>ML-SFT</td><td>01:36:43</td><td>01:33:10</td></tr><tr><td>CC-TUNING</td><td>01:51:58</td><td>01:45:15</td></tr><tr><td>Time Cost Ratio</td><td>1.16</td><td>1.13</td></tr><tr><td>ML-SFT+EN</td><td>03:08:25</td><td>03:03:02</td></tr><tr><td>CC-TUNING+EN</td><td>03:34:28</td><td>03:25:30</td></tr><tr><td>Time Cost Ratio</td><td>1.14</td><td>1.12</td></tr><tr><td>ML-SFT+MT</td><td>03:08:24</td><td>03:04:13</td></tr><tr><td>CC-TUNING+MT</td><td>03:34:19</td><td>03:25:59</td></tr><tr><td>Time Cost Ratio</td><td>1.14</td><td>1.12</td></tr><tr><td>ML-SFT+SDRRL</td><td>01:52:17</td><td>02:23:00</td></tr><tr><td>CC-TUNING+SDRRL</td><td>02:08:52</td><td>02:41:20</td></tr><tr><td>Time Cost Ratio</td><td>1.15</td><td>1.13</td></tr></table>


Table 4: Comparisons of training time cost.


<table><tr><td>Inference Time Cost (s)</td><td>LLaMA-3.1-8B</td><td>Qwen2.5-7B</td></tr><tr><td>vanilla inference</td><td>2012.26</td><td>1898.89</td></tr><tr><td>inference w/ Transform Matrix</td><td>2209.90</td><td>2064.50</td></tr><tr><td>Time Cost Ratio</td><td>1.10</td><td>1.09</td></tr></table>


Table 5: Comparisons of inference time cost on the Arabic subset of XNLI dataset.


<table><tr><td>Consistency</td><td>MKQA</td><td>XQuAD</td><td>XLSum</td></tr><tr><td colspan="4">LLaMA-3.1-8B</td></tr><tr><td>Base Model</td><td>0.880</td><td>0.988</td><td>0.879</td></tr><tr><td>ML-SFT</td><td>0.879</td><td>0.901</td><td>0.983</td></tr><tr><td>ML-SFT+EN</td><td>0.911</td><td>0.888</td><td>0.988</td></tr><tr><td>ML-SFT+MT</td><td>0.827</td><td>0.888</td><td>0.986</td></tr><tr><td>ML-SFT+SDRRL</td><td>0.360</td><td>0.447</td><td>0.352</td></tr><tr><td>CC-TUNING</td><td>0.972</td><td>0.967</td><td>0.993</td></tr><tr><td colspan="4">Qwen2.5-7B</td></tr><tr><td>Base Model</td><td>0.963</td><td>0.915</td><td>0.955</td></tr><tr><td>ML-SFT</td><td>0.996</td><td>0.999</td><td>0.999</td></tr><tr><td>ML-SFT+EN</td><td>0.994</td><td>0.998</td><td>0.998</td></tr><tr><td>ML-SFT+MT</td><td>0.995</td><td>0.998</td><td>0.981</td></tr><tr><td>ML-SFT+SDRRL</td><td>0.541</td><td>0.642</td><td>0.582</td></tr><tr><td>CC-TUNING</td><td>0.996</td><td>0.997</td><td>0.997</td></tr></table>


Table 6: Input and output language consistency results.


<table><tr><td colspan="5">Models</td><td colspan="7">Dataset: XNLI</td></tr><tr><td></td><td>en</td><td>ar</td><td>el</td><td>hi</td><td>ru</td><td>sw</td><td>th</td><td>tr</td><td>ur</td><td>zh</td><td>Avg</td></tr><tr><td>Vanilla Model (LLaMA-3.1-8B)</td><td>36.20</td><td>15.20</td><td>23.90</td><td>31.80</td><td>29.70</td><td>28.20</td><td>29.70</td><td>28.10</td><td>24.50</td><td>8.90</td><td>25.62</td></tr><tr><td>ML-SFT (LLaMA-3.1-8B)</td><td>12.90</td><td>35.50</td><td>35.80</td><td>31.10</td><td>34.50</td><td>31.20</td><td>31.60</td><td>37.70</td><td>33.10</td><td>35.40</td><td>31.88</td></tr><tr><td>+EN</td><td>46.60</td><td>38.70</td><td>24.40</td><td>32.20</td><td>32.70</td><td>31.90</td><td>32.00</td><td>37.60</td><td>38.90</td><td>35.20</td><td>35.02</td></tr><tr><td>+MT</td><td>47.20</td><td>35.10</td><td>22.10</td><td>35.80</td><td>40.90</td><td>31.00</td><td>32.90</td><td>39.20</td><td>36.90</td><td>37.90</td><td>35.90</td></tr><tr><td>+SDRRL</td><td>29.80</td><td>29.70</td><td>29.70</td><td>29.80</td><td>29.70</td><td>29.80</td><td>29.70</td><td>29.70</td><td>29.80</td><td>29.70</td><td>29.74</td></tr><tr><td>CC-TUNING (LLaMA-3.1-8B)</td><td>51.10</td><td>40.50</td><td>38.90</td><td>33.20</td><td>42.50</td><td>30.70</td><td>37.10</td><td>39.00</td><td>35.10</td><td>36.10</td><td>38.42</td></tr><tr><td>+EN</td><td>39.50</td><td>32.00</td><td>33.40</td><td>29.20</td><td>34.50</td><td>30.00</td><td>31.20</td><td>31.70</td><td>34.50</td><td>31.20</td><td>32.72</td></tr><tr><td>+MT</td><td>48.70</td><td>36.20</td><td>37.30</td><td>30.40</td><td>39.70</td><td>31.10</td><td>32.50</td><td>38.00</td><td>33.00</td><td>37.50</td><td>36.44</td></tr><tr><td>+SDRRL</td><td>29.70</td><td>29.60</td><td>29.70</td><td>31.00</td><td>29.50</td><td>29.00</td><td>29.70</td><td>29.70</td><td>30.70</td><td>29.80</td><td>29.84</td></tr><tr><td>Vanilla Model (Qwen2.5-7B)</td><td>90.20</td><td>43.60</td><td>40.40</td><td>41.80</td><td>58.70</td><td>11.60</td><td>46.90</td><td>42.50</td><td>33.60</td><td>49.00</td><td>45.83</td></tr><tr><td>ML-SFT (Qwen2.5-7B)</td><td>60.60</td><td>52.60</td><td>44.80</td><td>45.90</td><td>56.60</td><td>29.70</td><td>51.20</td><td>48.10</td><td>36.30</td><td>56.50</td><td>48.23</td></tr><tr><td>+EN</td><td>81.70</td><td>54.10</td><td>39.70</td><td>43.30</td><td>59.60</td><td>30.50</td><td>50.70</td><td>49.30</td><td>39.70</td><td>59.00</td><td>50.76</td></tr><tr><td>+MT</td><td>61.60</td><td>49.60</td><td>36.70</td><td>45.80</td><td>56.80</td><td>29.70</td><td>51.60</td><td>48.10</td><td>42.00</td><td>48.60</td><td>47.05</td></tr><tr><td>+SDRRL</td><td>81.60</td><td>56.60</td><td>34.00</td><td>51.00</td><td>60.20</td><td>33.10</td><td>55.20</td><td>54.10</td><td>38.90</td><td>58.90</td><td>52.36</td></tr><tr><td>CC-TUNING (Qwen2.5-7B)</td><td>78.90</td><td>51.80</td><td>41.70</td><td>44.00</td><td>60.10</td><td>30.70</td><td>52.30</td><td>50.60</td><td>40.50</td><td>59.40</td><td>51.00</td></tr><tr><td>+EN</td><td>72.00</td><td>54.30</td><td>43.10</td><td>47.00</td><td>56.50</td><td>28.60</td><td>51.70</td><td>48.40</td><td>35.90</td><td>57.30</td><td>49.48</td></tr><tr><td>+MT</td><td>64.40</td><td>51.40</td><td>36.70</td><td>43.70</td><td>57.70</td><td>29.70</td><td>52.50</td><td>49.40</td><td>37.90</td><td>57.90</td><td>48.13</td></tr><tr><td>+SDRRL</td><td>83.80</td><td>56.30</td><td>33.40</td><td>46.00</td><td>60.00</td><td>32.20</td><td>58.00</td><td>58.30</td><td>42.30</td><td>60.30</td><td>53.06</td></tr><tr><td colspan="5">Models</td><td colspan="7">Dataset: XStoryCloze</td></tr><tr><td></td><td>en</td><td>ar</td><td>es</td><td>eu</td><td>hi</td><td>id</td><td>ru</td><td>sw</td><td>te</td><td>zh</td><td>Avg</td></tr><tr><td>Vanilla Model (LLaMA-3.1-8B)</td><td>49.70</td><td>41.69</td><td>14.89</td><td>28.72</td><td>48.44</td><td>60.03</td><td>36.47</td><td>49.70</td><td>6.02</td><td>19.72</td><td>35.54</td></tr><tr><td>ML-SFT (LLaMA-3.1-8B)</td><td>88.62</td><td>65.32</td><td>21.91</td><td>64.86</td><td>70.81</td><td>83.39</td><td>43.61</td><td>62.54</td><td>63.40</td><td>87.82</td><td>65.23</td></tr><tr><td>+EN</td><td>77.04</td><td>40.44</td><td>65.45</td><td>59.36</td><td>77.10</td><td>79.62</td><td>49.44</td><td>60.49</td><td>55.46</td><td>86.90</td><td>65.13</td></tr><tr><td>+MT</td><td>91.73</td><td>77.70</td><td>85.04</td><td>60.82</td><td>75.78</td><td>80.48</td><td>62.14</td><td>57.84</td><td>20.91</td><td>86.57</td><td>69.90</td></tr><tr><td>+SDRRL</td><td>72.93</td><td>65.32</td><td>45.80</td><td>20.32</td><td>68.63</td><td>64.13</td><td>66.05</td><td>45.14</td><td>49.64</td><td>60.29</td><td>55.82</td></tr><tr><td>CC-TUNING (LLaMA-3.1-8B)</td><td>89.34</td><td>73.73</td><td>64.26</td><td>51.09</td><td>79.48</td><td>79.81</td><td>70.22</td><td>58.24</td><td>55.33</td><td>84.45</td><td>70.60</td></tr><tr><td>+EN</td><td>69.36</td><td>66.71</td><td>75.38</td><td>25.74</td><td>62.48</td><td>75.78</td><td>49.77</td><td>50.36</td><td>49.24</td><td>84.58</td><td>60.94</td></tr><tr><td>+MT</td><td>87.43</td><td>73.99</td><td>87.62</td><td>57.91</td><td>82.06</td><td>82.86</td><td>76.44</td><td>59.43</td><td>38.65</td><td>89.01</td><td>73.54</td></tr><tr><td>+SDRRL</td><td>86.96</td><td>71.67</td><td>70.42</td><td>34.61</td><td>80.61</td><td>77.17</td><td>77.63</td><td>50.56</td><td>66.18</td><td>76.04</td><td>69.19</td></tr><tr><td>Vanilla Model (Qwen2.5-7B)</td><td>85.97</td><td>85.84</td><td>91.40</td><td>18.07</td><td>78.76</td><td>69.89</td><td>91.33</td><td>17.94</td><td>55.92</td><td>75.84</td><td>67.09</td></tr><tr><td>ML-SFT (Qwen2.5-7B)</td><td>92.12</td><td>78.89</td><td>93.51</td><td>52.95</td><td>79.48</td><td>79.48</td><td>71.21</td><td>37.06</td><td>28.92</td><td>86.96</td><td>70.06</td></tr><tr><td>+EN</td><td>78.23</td><td>54.27</td><td>91.00</td><td>56.25</td><td>81.80</td><td>87.36</td><td>71.34</td><td>44.74</td><td>61.02</td><td>90.27</td><td>71.63</td></tr><tr><td>+MT</td><td>82.06</td><td>56.12</td><td>92.19</td><td>57.64</td><td>82.20</td><td>88.15</td><td>73.92</td><td>29.52</td><td>57.78</td><td>85.44</td><td>70.50</td></tr><tr><td>+SDRRL</td><td>93.85</td><td>88.42</td><td>94.51</td><td>62.61</td><td>82.00</td><td>89.15</td><td>93.05</td><td>52.88</td><td>62.01</td><td>88.22</td><td>80.67</td></tr><tr><td>CC-TUNING (Qwen2.5-7B)</td><td>91.59</td><td>81.60</td><td>91.00</td><td>54.86</td><td>77.96</td><td>80.68</td><td>78.82</td><td>35.94</td><td>37.59</td><td>84.25</td><td>71.43</td></tr><tr><td>+EN</td><td>39.38</td><td>37.46</td><td>90.40</td><td>55.26</td><td>79.55</td><td>86.70</td><td>85.24</td><td>45.00</td><td>42.55</td><td>85.31</td><td>64.69</td></tr><tr><td>+MT</td><td>65.32</td><td>66.64</td><td>91.66</td><td>55.06</td><td>81.67</td><td>86.43</td><td>75.12</td><td>52.75</td><td>53.47</td><td>85.77</td><td>71.39</td></tr><tr><td>+SDRRL</td><td>93.45</td><td>90.87</td><td>92.26</td><td>57.58</td><td>82.26</td><td>88.68</td><td>94.51</td><td>57.25</td><td>59.03</td><td>93.45</td><td>80.93</td></tr><tr><td colspan="5">Models</td><td colspan="7">Dataset: MMMLU</td></tr><tr><td></td><td>en</td><td>ar</td><td>bn</td><td>es</td><td>hi</td><td>id</td><td>ko</td><td>pt</td><td>sw</td><td>yo</td><td>Avg</td></tr><tr><td>Vanilla Model (LLaMA-3.1-8B)</td><td>45.40</td><td>28.20</td><td>17.70</td><td>11.30</td><td>25.10</td><td>26.40</td><td>25.00</td><td>11.70</td><td>16.20</td><td>0.60</td><td>20.76</td></tr><tr><td>ML-SFT (LLaMA-3.1-8B)</td><td>57.40</td><td>41.60</td><td>31.70</td><td>51.20</td><td>37.60</td><td>44.70</td><td>39.40</td><td>51.70</td><td>20.70</td><td>26.00</td><td>40.20</td></tr><tr><td>+EN</td><td>56.80</td><td>35.90</td><td>32.00</td><td>50.90</td><td>36.00</td><td>40.80</td><td>40.50</td><td>48.10</td><td>29.80</td><td>25.40</td><td>39.62</td></tr><tr><td>+MT</td><td>59.20</td><td>37.80</td><td>33.10</td><td>51.50</td><td>37.60</td><td>42.80</td><td>42.50</td><td>48.10</td><td>28.30</td><td>25.90</td><td>40.68</td></tr><tr><td>+SDRRL</td><td>53.70</td><td>32.20</td><td>23.00</td><td>27.80</td><td>35.30</td><td>30.50</td><td>27.20</td><td>36.70</td><td>12.80</td><td>1.40</td><td>28.06</td></tr><tr><td>CC-TUNING (LLaMA-3.1-8B)</td><td>57.50</td><td>41.30</td><td>33.40</td><td>51.70</td><td>37.60</td><td>43.30</td><td>41.70</td><td>46.80</td><td>27.00</td><td>27.10</td><td>40.74</td></tr><tr><td>+EN</td><td>56.50</td><td>38.10</td><td>30.80</td><td>49.90</td><td>37.00</td><td>40.70</td><td>39.00</td><td>47.30</td><td>26.80</td><td>21.20</td><td>38.73</td></tr><tr><td>+MT</td><td>55.70</td><td>36.80</td><td>30.70</td><td>49.60</td><td>36.10</td><td>39.90</td><td>38.70</td><td>48.40</td><td>26.70</td><td>26.10</td><td>38.87</td></tr><tr><td>+SDRRL</td><td>53.10</td><td>38.60</td><td>33.40</td><td>46.40</td><td>36.50</td><td>41.10</td><td>35.70</td><td>47.20</td><td>31.70</td><td>14.00</td><td>37.77</td></tr><tr><td>Vanilla Model (Qwen2.5-7B)</td><td>68.20</td><td>53.50</td><td>43.70</td><td>64.00</td><td>47.40</td><td>60.90</td><td>46.00</td><td>62.40</td><td>17.90</td><td>1.30</td><td>46.53</td></tr><tr><td>ML-SFT (Qwen2.5-7B)</td><td>69.80</td><td>53.30</td><td>42.00</td><td>65.60</td><td>41.10</td><td>59.50</td><td>55.70</td><td>62.60</td><td>28.60</td><td>22.30</td><td>50.05</td></tr><tr><td>+EN</td><td>65.90</td><td>53.20</td><td>37.90</td><td>64.60</td><td>40.30</td><td>56.80</td><td>55.70</td><td>62.10</td><td>28.20</td><td>23.30</td><td>48.80</td></tr><tr><td>+MT</td><td>60.50</td><td>51.40</td><td>40.00</td><td>63.90</td><td>39.50</td><td>58.30</td><td>49.00</td><td>63.00</td><td>28.80</td><td>20.50</td><td>47.49</td></tr><tr><td>+SDRRL</td><td>66.00</td><td>46.00</td><td>38.30</td><td>60.70</td><td>41.50</td><td>56.90</td><td>52.10</td><td>58.60</td><td>29.90</td><td>22.80</td><td>47.28</td></tr><tr><td>CC-TUNING (Qwen2.5-7B)</td><td>69.10</td><td>52.80</td><td>40.50</td><td>65.10</td><td>41.20</td><td>59.10</td><td>54.90</td><td>62.30</td><td>30.90</td><td>20.60</td><td>49.65</td></tr><tr><td>+EN</td><td>67.10</td><td>54.50</td><td>38.10</td><td>64.20</td><td>41.90</td><td>55.50</td><td>53.90</td><td>61.30</td><td>24.10</td><td>12.90</td><td>47.35</td></tr><tr><td>+MT</td><td>66.60</td><td>53.10</td><td>40.30</td><td>64.70</td><td>41.70</td><td>60.50</td><td>53.60</td><td>63.10</td><td>29.90</td><td>20.40</td><td>49.39</td></tr><tr><td>+SDRRL</td><td>66.30</td><td>50.60</td><td>39.30</td><td>60.30</td><td>41.60</td><td>56.30</td><td>52.30</td><td>57.50</td><td>28.10</td><td>26.40</td><td>47.87</td></tr><tr><td colspan="5">Models</td><td colspan="7">Dataset: MKQA</td></tr><tr><td></td><td>en</td><td>ar</td><td>de</td><td>ja</td><td>ko</td><td>pt</td><td>ru</td><td>tr</td><td>vi</td><td>zh</td><td>Avg</td></tr><tr><td>Vanilla Model (LLaMA-3.1-8B)</td><td>22.50</td><td>5.20</td><td>3.50</td><td>3.50</td><td>3.00</td><td>4.80</td><td>4.90</td><td>15.70</td><td>6.80</td><td>5.70</td><td>7.56</td></tr><tr><td>ML-SFT (LLaMA-3.1-8B)</td><td>33.60</td><td>4.90</td><td>23.30</td><td>9.10</td><td>6.00</td><td>20.70</td><td>10.00</td><td>15.60</td><td>14.30</td><td>8.90</td><td>14.64</td></tr><tr><td>+EN</td><td>26.00</td><td>6.30</td><td>18.80</td><td>9.70</td><td>5.70</td><td>19.40</td><td>10.00</td><td>15.60</td><td>13.00</td><td>8.30</td><td>13.28</td></tr><tr><td>+MT</td><td>29.20</td><td>5.80</td><td>19.50</td><td>9.10</td><td>5.70</td><td>17.10</td><td>10.70</td><td>15.50</td><td>14.40</td><td>8.60</td><td>13.56</td></tr><tr><td>+SDRRL</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>CC-TUNING (LLaMA-3.1-8B)</td><td>32.00</td><td>6.00</td><td>24.10</td><td>10.90</td><td>6.30</td><td>22.40</td><td>10.50</td><td>17.80</td><td>18.20</td><td>11.20</td><td>15.94</td></tr><tr><td>+EN</td><td>27.70</td><td>6.40</td><td>20.20</td><td>11.10</td><td>7.30</td><td>20.50</td><td>9.50</td><td>17.20</td><td>15.50</td><td>10.70</td><td>14.61</td></tr><tr><td>+MT</td><td>32.10</td><td>6.90</td><td>21.90</td><td>10.70</td><td>7.30</td><td>21.10</td><td>10.10</td><td>17.70</td><td>17.30</td><td>10.80</td><td>15.59</td></tr><tr><td>+SDRRL</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Vanilla Model (Qwen2.5-7B)</td><td>1.00</td><td>6.60</td><td>8.50</td><td>10.40</td><td>8.30</td><td>7.20</td><td>7.50</td><td>10.10</td><td>15.70</td><td>15.20</td><td>9.05</td></tr><tr><td>ML-SFT (Qwen2.5-7B)</td><td>30.30</td><td>6.60</td><td>19.10</td><td>11.20</td><td>8.90</td><td>19.70</td><td>9.40</td><td>12.10</td><td>15.80</td><td>14.20</td><td>14.73</td></tr><tr><td>+EN</td><td>27.80</td><td>6.90</td><td>14.80</td><td>10.80</td><td>7.40</td><td>17.50</td><td>8.10</td><td>10.10</td><td>14.70</td><td>12.40</td><td>13.05</td></tr><tr><td>+MT</td><td>27.10</td><td>6.90</td><td>16.10</td><td>9.60</td><td>7.90</td><td>19.40</td><td>8.60</td><td>11.00</td><td>14.70</td><td>14.10</td><td>13.54</td></tr><tr><td>+SDRRL</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>CC-TUNING (Qwen2.5-7B)</td><td>30.3</td><td>7.2</td><td>18.60</td><td>11.6</td><td>8.5</td><td>20.5</td><td>8.3</td><td>12.9</td><td>15.80</td><td>14.7</td><td>14.84</td></tr><tr><td>+EN</td><td>27.60</td><td>5.90</td><td>15.10</td><td>10.80</td><td>7.60</td><td>19.20</td><td>9.60</td><td>12.90</td><td>13.60</td><td>13.30</td><td>13.56</td></tr><tr><td>+MT</td><td>29.30</td><td>6.50</td><td>16.10</td><td>11.20</td><td>7.90</td><td>18.30</td><td>8.50</td><td>12.50</td><td>13.30</td><td>14.10</td><td>13.77</td></tr><tr><td>+SDRRL</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td colspan="5">Models</td><td colspan="7">Dataset: XQuAD</td></tr><tr><td></td><td>en</td><td>ar</td><td>bn</td><td>es</td><td>hi</td><td>id</td><td>ko</td><td>pt</td><td>sw</td><td>yo</td><td>Avg</td></tr><tr><td>Vanilla Model (LLaMA-3.1-8B)</td><td>72.18</td><td>52.86</td><td>58.07</td><td>47.73</td><td>61.26</td><td>43.87</td><td>46.97</td><td>51.93</td><td>53.53</td><td>68.32</td><td>55.67</td></tr><tr><td>ML-SFT (LLaMA-3.1-8B)</td><td>72.61</td><td>56.13</td><td>64.62</td><td>52.18</td><td>60.00</td><td>47.73</td><td>58.49</td><td>53.70</td><td>64.87</td><td>73.87</td><td>60.42</td></tr><tr><td>+EN</td><td>63.28</td><td>53.45</td><td>62.02</td><td>51.09</td><td>57.73</td><td>46.39</td><td>58.40</td><td>50.84</td><td>61.18</td><td>69.66</td><td>57.40</td></tr><tr><td>+MT</td><td>71.76</td><td>53.78</td><td>60.84</td><td>50.84</td><td>58.99</td><td>49.33</td><td>54.03</td><td>47.73</td><td>65.21</td><td>71.51</td><td>58.40</td></tr><tr><td>+SDRRL</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>CC-TUNING (LLaMA-3.1-8B)</td><td>75.29</td><td>55.29</td><td>64.96</td><td>51.34</td><td>62.27</td><td>52.10</td><td>60.42</td><td>54.20</td><td>67.82</td><td>74.79</td><td>61.85</td></tr><tr><td>+EN</td><td>69.08</td><td>58.32</td><td>64.03</td><td>52.77</td><td>60.59</td><td>51.51</td><td>60.25</td><td>52.69</td><td>66.64</td><td>73.03</td><td>60.89</td></tr><tr><td>+MT</td><td>77.73</td><td>55.29</td><td>63.45</td><td>53.61</td><td>61.34</td><td>52.18</td><td>56.72</td><td>53.11</td><td>68.24</td><td>73.78</td><td>61.55</td></tr><tr><td>+SDRRL</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Vanilla Model (Qwen2.5-7B)</td><td>53.19</td><td>71.26</td><td>71.01</td><td>50.17</td><td>49.92</td><td>56.39</td><td>64.62</td><td>57.98</td><td>77.31</td><td>89.24</td><td>64.11</td></tr><tr><td>ML-SFT (Qwen2.5-7B)</td><td>79.92</td><td>66.97</td><td>70.08</td><td>40.00</td><td>46.39</td><td>53.95</td><td>64.96</td><td>56.05</td><td>72.77</td><td>85.04</td><td>63.61</td></tr><tr><td>+EN</td><td>74.29</td><td>64.54</td><td>69.41</td><td>36.13</td><td>47.39</td><td>54.37</td><td>64.03</td><td>59.41</td><td>73.19</td><td>80.59</td><td>62.34</td></tr><tr><td>+MT</td><td>79.33</td><td>65.13</td><td>69.33</td><td>41.01</td><td>50.67</td><td>52.61</td><td>67.56</td><td>58.24</td><td>72.69</td><td>83.70</td><td>64.03</td></tr><tr><td>+SDRRL</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>CC-TUNING (Qwen2.5-7B)</td><td>79.24</td><td>64.12</td><td>71.34</td><td>39.75</td><td>47.06</td><td>53.61</td><td>65.71</td><td>57.73</td><td>74.03</td><td>84.62</td><td>63.72</td></tr><tr><td>+EN</td><td>72.18</td><td>64.45</td><td>68.40</td><td>41.01</td><td>47.31</td><td>54.37</td><td>66.30</td><td>58.99</td><td>72.94</td><td>80.92</td><td>62.69</td></tr><tr><td>+MT</td><td>77.98</td><td>67.31</td><td>71.26</td><td>39.75</td><td>49.41</td><td>52.86</td><td>69.33</td><td>57.82</td><td>72.27</td><td>84.62</td><td>64.26</td></tr><tr><td>+SDRRL</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td colspan="5">Models</td><td colspan="7">Dataset: XLSum</td></tr><tr><td></td><td>en</td><td>ar</td><td>fr</td><td>hi</td><td>id</td><td>ru</td><td>sw</td><td>tr</td><td>ur</td><td>vi</td><td>Avg</td></tr><tr><td>Vanilla Model (LLaMA-3.1-8B)</td><td>6.60</td><td>3.88</td><td>11.91</td><td>1.02</td><td>5.63</td><td>7.62</td><td>3.53</td><td>5.74</td><td>1.54</td><td>9.59</td><td>5.71</td></tr><tr><td>ML-SFT (LLaMA-3.1-8B)</td><td>24.36</td><td>9.67</td><td>18.66</td><td>1.94</td><td>13.72</td><td>14.47</td><td>8.05</td><td>11.07</td><td>6.64</td><td>14.14</td><td>12.27</td></tr><tr><td>+EN</td><td>22.46</td><td>10.62</td><td>19.66</td><td>2.97</td><td>13.72</td><td>14.02</td><td>6.76</td><td>7.14</td><td>5.77</td><td>17.27</td><td>12.04</td></tr><tr><td>+MT</td><td>25.74</td><td>11.06</td><td>19.50</td><td>3.97</td><td>14.78</td><td>14.74</td><td>7.56</td><td>9.58</td><td>7.16</td><td>14.78</td><td>12.89</td></tr><tr><td>+SDRRL</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>CC-TUNING (LLaMA-3.1-8B)</td><td>25.00</td><td>10.87</td><td>19.46</td><td>3.02</td><td>13.46</td><td>15.55</td><td>8.63</td><td>10.01</td><td>7.20</td><td>15.63</td><td>12.88</td></tr><tr><td>+EN</td><td>23.76</td><td>10.26</td><td>21.45</td><td>3.67</td><td>14.30</td><td>14.45</td><td>8.94</td><td>9.94</td><td>6.15</td><td>14.92</td><td>12.78</td></tr><tr><td>+MT</td><td>27.57</td><td>11.38</td><td>21.08</td><td>3.23</td><td>13.34</td><td>15.71</td><td>9.14</td><td>10.88</td><td>4.41</td><td>13.73</td><td>13.05</td></tr><tr><td>+SDRRL</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Vanilla Model (Qwen2.5-7B)</td><td>10.45</td><td>3.59</td><td>10.86</td><td>0.00</td><td>5.43</td><td>6.89</td><td>2.73</td><td>3.54</td><td>3.09</td><td>4.21</td><td>5.08</td></tr><tr><td>ML-SFT (Qwen2.5-7B)</td><td>24.13</td><td>12.20</td><td>22.10</td><td>0.33</td><td>14.89</td><td>16.10</td><td>5.95</td><td>8.04</td><td>5.47</td><td>14.74</td><td>12.40</td></tr><tr><td>+EN</td><td>23.75</td><td>11.70</td><td>20.14</td><td>0.33</td><td>14.97</td><td>15.61</td><td>6.90</td><td>8.51</td><td>5.78</td><td>14.36</td><td>12.20</td></tr><tr><td>+MT</td><td>26.72</td><td>12.32</td><td>21.47</td><td>0.67</td><td>14.00</td><td>15.29</td><td>5.66</td><td>8.73</td><td>5.12</td><td>14.78</td><td>12.48</td></tr><tr><td>+SDRRL</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>CC-TUNING (Qwen2.5-7B)</td><td>23.22</td><td>10.75</td><td>22.21</td><td>0.62</td><td>14.47</td><td>17.61</td><td>6.47</td><td>8.37</td><td>5.43</td><td>15.84</td><td>12.50</td></tr><tr><td>+EN</td><td>25.06</td><td>12.79</td><td>19.58</td><td>0.33</td><td>14.71</td><td>15.63</td><td>7.12</td><td>10.62</td><td>5.39</td><td>15.01</td><td>12.63</td></tr><tr><td>+MT</td><td>25.84</td><td>11.46</td><td>22.62</td><td>1.00</td><td>15.77</td><td>16.43</td><td>5.69</td><td>9.37</td><td>5.06</td><td>15.46</td><td>12.87</td></tr><tr><td>+SDRRL</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr></table>


Table 7: The detailed performance results of different language subsets on NLU tasks (XNLI, XStoryCloze, MMMLU) across all involved models and baselines.


Table 8: The detailed performance results of different language subsets on NLG tasks (MKQA, XQuAD, XLSum) across all involved models and baselines. 