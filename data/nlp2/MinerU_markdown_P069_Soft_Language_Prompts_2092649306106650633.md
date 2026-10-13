# Soft Language Prompts for Language Transfer

Ivan Vykopal<sup>1,2</sup>, Simon Ostermann<sup>3,4</sup> and Marián Šimko<sup>2</sup> 

<sup>1</sup> Faculty of Information Technology, Brno University of Technology, Brno, Czech Republic <sup>2</sup> Kempelen Institute of Intelligent Technologies, Bratislava, Slovakia {name.surname}@kinit.sk 

<sup>3</sup> German Research Center for Artificial Intelligence (DFKI), Saarbrücken, Germany <sup>4</sup> Centre for European Research in Trusted AI (CERTAIN), Saarbrücken, Germany simon.ostermann@dfki.de 

## Abstract

Cross-lingual knowledge transfer, especially between high- and low-resource languages, remains challenging in natural language processing (NLP). This study offers insights for improving cross-lingual NLP applications through the combination of parameter-efficient finetuning methods. We systematically explore strategies for enhancing cross-lingual transfer through the incorporation of language-specific and task-specific adapters and soft prompts. We present a detailed investigation of various combinations of these methods, exploring their efficiency across 16 languages, focusing on 10 mid- and low-resource languages. We further present to our knowledge the first use of soft prompts for language transfer, a technique we call soft language prompts. Our findings demonstrate that in contrast to claims of previous work, a combination of language and task adapters does not always work best; instead, combining a soft language prompt with a task adapter outperforms most configurations in many cases. 

## 1 Introduction

Many multilingual large language models (LLMs) have been developed in recent years, demonstrating promising performance on various NLP tasks across multiple languages (Xue et al., 2021; Workshop et al., 2023). These models are pre-trained on extensive corpora of unlabelled data in numerous languages, allowing an adaptation to linguistic characteristics and nuances. In addition, LLMs are often further trained on downstream tasks in a selected subset of languages (Muennighoff et al., 2023). However, only few LLMs focus on lowresource languages (Tang et al., 2020; Xue et al., 2021; Üstün et al., 2024; Gurgurov et al., 2024). 

As the number of covered languages in the model increases, the issue of the curse of multilinguality arises. This problem occurs when the LLM’s capacity is limited, causing languages with less training data to perform poorly (Conneau et al., 2020). Various approaches have been employed to address this limitation, primarily involving additional trainable parameters specific to individual languages (Pfeiffer et al., 2020, 2023). 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/6cec3d6f-b612-41bf-b581-12cb1790085d/d9ba93898985df01e51fd3e752ec24dbfa8a1d6172a934cc6d5b88a6b385539f.jpg)



Figure 1: The full pipeline consists of (1) training lan guage representation on 16 languages on the unlabelled data from Wikipedia; (2) training task representations on six high-resource languages on four NLP tasks; and (3) evaluating of cross-lingual transfer capabilities on 10 mid- and low-resource languages.


An alternative to language-specific tuning is cross-lingual transfer, where researchers investigate the knowledge transfer between high and lowresource languages. In cross-lingual transfer methods, an LLM is trained on a downstream task in one language, most often high-resource, and evaluated in other languages (Pikuliak et al., 2021). However, training only task-specific representations does not always capture the nuances of languages on which the LLM has not been trained or has been trained only to a small extent. Therefore, incorporating language-specific features can enhance knowledge transfer across languages. 

Previous work has primarily investigated language and task representations by training language and task-specific adapters (Pfeiffer et al., 2020; Parovic et al.´ , 2022) or by employing language arithmetics (Klimaszewski et al., 2024). Nonetheless, other approaches that involve incorporating additional parameters to the model for language representation have not been thoroughly explored. This opens the opportunity to explore a combination of language and task representations using other methods and their impact on cross-lingual knowledge transfer. 

To explore the utilization of language and task representations, we evaluate various configurations by combining two parameter-efficient fine-tuning (PEFT) methods that incorporate additional parameters into the LLM, namely adapters and prompttuning. Leveraging these additional language- and task-specific parameters increases the capacity of an mT0-BASE model and improves performance for cross-lingual transfer. We evaluate the performance of each configuration by training on six highresource languages and evaluating its effectiveness on 10 mid- and low-resource languages on four selected tasks<sup>1</sup>. 

Our main contributions are: 

• We propose soft language prompts as an alternative method for cross-lingual transfer. 

• We comprehensively evaluate combinations of adapters and soft prompts for cross-lingual transfer and find that language prompts provide a viable alternative to language adapters, especially for low-resource languages. 

• In addition, we provide an exhaustive evaluation of both prompts and adapters for task transfer. We find that the best combination of adapters and prompts for task and language transfer depends highly on task and language, resp., and that no solution clearly outperforms the others. 

## 2 Related Work

Adapters and Soft Prompts. PEFT methods are designed to address the problem of the increasing number of trainable parameters in LLMs (He et al., 2022; Dettmers et al., 2023; Zhang et al., 2023; Xu et al., 2023; Xie et al., 2024). These methods reduce the number of trained parameters and incorporate new parameters commonly used to train LLMs on other tasks. Adapters (Houlsby et al., 2019a) and Prompt-Tuning (Lester et al., 2021) represent two PEFT methods for adapting LLMs to different NLP domains. Adapters incorporate new parameters into the transformer architecture by including down- and up-projection layers along with residual connection, while prompt-tuning introduced trainable soft-prompts prepended to input embeddings to condition the LLM’s generation. 

Limitations of Multilingual LLMs. One major limitation of LLMs is catastrophic forgetting, which occurs when training the LLM on a new task, causing it to partially or entirely forget previously learned knowledge for other tasks (McCloskey and Cohen, 1989; Luo et al., 2024; Ren et al., 2024). This forgetting extends beyond task-specific knowledge to language-specific knowledge if the model is fine-tuned on a subset of the original languages (Vu et al., 2022a; Liu and Huang, 2023). 

Another challenge with multilingual LLMs is associated with the number of languages on which these LLMs have been pre-trained (Conneau et al., 2020; Pfeiffer et al., 2022). Previous research has shown that as the number of languages covered by LLMs increases, their performance on various NLP tasks degrades (Hu et al., 2020; Ponti et al., 2020). Additionally, low-resource languages are often underrepresented during pre-training, resulting in poor performance in these languages (Wu and Dredze, 2020). 

Cross-Lingual Transfer. Given the many lowresource and underrepresented languages, crosslingual transfer is crucial for training LLMs to address NLP tasks in various languages (Pikuliak et al., 2021). A common approach involves training LLMs in one language and evaluating them in another. Recent methods use additional parameters to create language-specific representations, assisting LLMs in solving NLP tasks in low-resource languages (Üstün et al., 2020; Ansell et al., 2022). These include training task adapters on top of language adapters (Pfeiffer et al., 2020; Ansell et al., 2021; Pfeiffer et al., 2023; Kunz and Holmström, 2024), training language adapters on source and target languages (Parovic et al.´ , 2022), and fusing multiple task (Lee et al., 2022) or language adapters (Rathore et al., 2023). Other approaches leverage soft prompts (Huang et al., 2022; Philippy et al., 2024) or grammar prompting (Wang et al., 2024). While many works focus on specific tasks, our study explored different combinations of adapters and soft prompts for cross-lingual transfer on four tasks, minimizing the reliance on machine translation, which is often unreliable for low-resource languages. 

## 3 Methodology

We propose a comprehensive study on combinations of language and task representations using adapters and soft prompts. We evaluate for the first time the capabilities of soft language prompts in a systematic manner and evaluate the performance of diverse combinations of soft prompts and adapters for cross-lingual transfer. Our pipeline, illustrated in Figure 1, consists of three steps. Firstly, we train language adapters and soft prompts in 16 languages on unlabelled data. Secondly, along with trained language representation, we incorporate task representation into the LLM’s architecture and train the task representation on four tasks in six high-resource languages. Finally, we evaluate the cross-lingual transfer on 10 mid- and low-resource languages and four selected tasks. 

In the following sections, we first give details on methods that we investigate for representing language (Section 3.1) and task information (Section 3.2). We then explain the combinations of soft prompts and adapters we evaluate (Section 3.3). 

## 3.1 Language Representation

Language Adapters. Previous work has investigated the efficiency of training languagespecific transformation using the adapter architecture (Houlsby et al., 2019b). Pfeiffer et al. (2020) proposed a MAD-X framework, which includes training language adapters using the masked language modeling objective on unlabelled data. Inspired by language adapters proposed by the authors, we build upon their architecture and the approach used to train language adapters. In our settings, language adapters are incorporated into each transformer layer of the LLM and trained using unlabelled data from Wikipedia. 

Soft Language Prompts. Soft Prompt Tuning offers a promising, parameter-efficient method for adapting LLMs. Previous work has predominantly explored task-specific soft prompts to enhance task transferability, typically focusing on a single language (Vu et al., 2022b; Asai et al., 2022). In contrast, we extend this approach by training languagespecific soft prompts to guide multilingual LLMs toward a target language. Given that multilingual LLMs can generate responses in various languages, we defined a soft language prompt as a set of token embeddings prepended to the original input embedding. These embeddings are then fed into the LLM to condition its output in the desired language. 

Existing studies have highlighted the importance of soft prompt initialization in optimizing the performance of LLMs. Lester et al. (2021) outlined three possible strategies: (1) random initialization using a Gaussian distribution; (2) initialization from the model’s vocabulary; and (3) initialization with the embeddings ofoutput classesfor classification tasks. While each method has its strengths and limitations, none directly apply to our experiments, which focus on multilingual LLMs. To address this, we introduce a language-specific text instruction for soft prompt initialization (see Appendix C). In this approach, the text instruction is first embedded, and if its length is shorter than the required soft prompt size, the embedding is repeated until the desired length is achieved. This approach already steers the model to the desired language and can fasten the convergence during the training. 

Language Modeling Objective. Training language-specific representations requires unlabelled data from the selected languages and careful selection of an appropriate training objective. Given our use of an encoder-decoder architecture, we adopt span corruption as the training objective, which has been shown to be efficient in prior work (Raffel et al., 2020; Xue et al., 2021). Unlike the casual language modeling objective, where the LLM predicts the next token in a sequence, span corruption randomly masks 15% of the tokens in the input text using sentinel tokens. These tokens serve solely to mark the masked parts, which the LLM is tasked to reconstruct (Raffel et al., 2020). Finally, the LLM is trained to predict the original tokens for the masked portions, enabling it to learn linguistic nuances and patterns that are crucial for training task-specific adapters and soft prompts. 

## 3.2 Task Representation

Task Adapters. Similarly to language adapters, we use task-specific adapters, represented by the same adapter architecture, which are incorporated into each transformer layer of the LLM. However, when combining task representations with language representations, the final architecture differs across configurations and depends on the type of language representation used during the training and inference. Detailed information on the architecture for all combinations is in Section 3.3. 

Task adapters are updated only during training on the desired downstream task, while the rest of the model, along with the language representation, is kept frozen. In the case of task-specific representations, LLMs learn knowledge that is characteristic of the specified tasks, and that should be language-independent. 

Soft Task Prompts. In addition to task adapters, we use soft task prompts with the same architecture and parameters as soft language prompts. The difference when using a soft task prompt occurs in the configuration consisting of a soft language prompt and a soft task prompt. With this configuration, both soft prompts are combined using a concatenation operation and further fed into the model to condition the final generation. This concatenation represents the stacking of two representations (language and task), similar to stacking two adapters on top of each other used in the MAD-X approach (Pfeiffer et al., 2020). 

## 3.3 Evaluated Combinations of Adapters and Soft Prompts

Since our experiments are focused on evaluating language and task representations and their combinations, we define six possible configurations: (1) only task adapter; (2) only soft task prompt; (3) MAD-X (Pfeiffer et al., 2020), i.e. the combination of language and task adapter; (4) the combination of language adapter and soft task prompt; (5) the combination of soft language prompt and task adapter; and (6) the combination of soft language prompt and soft task prompt. The position of task representations within the LLM highly depends on the type of language representation used in experiments. The architecture, along with the form of the input for all configurations, are illustrated in Figure 2. 

Single Task Adapters & Soft Task Prompts. The configurations that employ only task adapters or task soft prompts aim at training task representations without incorporating language-specific representation. Adapters and soft prompts were trained independently on each selected dataset from four tasks, and the resulting task representations were evaluated across all 10 mid- and low-resource languages. Only the adapters and soft prompts are trained during this process, while the rest of the LLM remains frozen. 

Language Adapters & Task Adapters. Beyond training task representations alone, we also trained a task adapter on top of a pre-trained language adapter, reproducing the approach outlined in 

MAD-X (Pfeiffer et al., 2020). Our method leverages the same architecture with distinct training hyperparameters fitted to the tasks at hand. In this setup, the task adapter takes the output of the language adapter as input and further processes it. Only the task adapter is trained during training, while the language adapter and LLM remain frozen. 

Adapters and Soft Prompts Combinations. Our study introduces two combinations of language and task representations using adapters and soft prompts. The first configuration involves soft task prompts along with a language adapter. This combination incorporates trained language-specific knowledge using a language adapter and a soft task prompt trained on the desired downstream task. 

The second combination includes training a task adapter with the trained soft language prompt. Soft language prompts condition LLMs to activate knowledge specific to the desired language, while task adapters learn task-specific knowledge. In both settings, language representations are kept frozen and only task representations are trained on the desired downstream tasks. 

Soft Language Prompts & Soft Task Prompts. The last configuration includes soft language and soft task prompts. Inspired by stacking language and task adapters on top of each other, we concatenated embeddings of language and task prompts to a final soft prompt, with the LLM and soft language prompt being frozen during training. 

## 4 Experiments

In the following section, we describe the process of selection of models (Section 4.1), languages (Section 4.2) and tasks (Section 4.3). 

## 4.1 Model Selection

We selected an encoder-decoder architecture, the mT0-BASE model, to conduct a cross-lingual evaluation. The mT0 model is based on the pre-trained multilingual mT5 model, which has been further fine-tuned on a collection of 46 languages across 16 NLP tasks (Muennighoff et al., 2023). The model selection played a crucial role in further experiments, and we conducted several preliminary experiments with the original mT5-BASE model. However, we observed that in the case of using pre-trained models, which have not been further fine-tuned on downstream tasks, prompt-tuning is not sufficient to guide the LLM to produce meaningful outputs. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/6cec3d6f-b612-41bf-b581-12cb1790085d/9b8f74c0baec0fa0dd0f690a00ea0fcda6d42b17b80bbc58f3cff4286742cc3b.jpg)



Figure 2: The architecture for all combinations of language and task representations in our experiments. These configurations include: (1) Language and Task Adapters; (2) Language Adapter and Soft Task Prompt; (3) Soft Language Prompt and Task Adapter; and (4) Soft Language and Soft Task Prompts. Language representations are in red, while task representations are in green color.


## 4.2 Languages

The original mT5 model was pre-trained on over 100 languages, while mT0 employed only 46 for further fine-tuning. From the list of languages supported by mT5, we selected 16 languages and categorized them into two groups: high- and mid- along with low-resource languages. On the one hand, we consider Arabic, German, English, Spanish, Russian and Chinese to be high-resource languages. On the other hand, we consider Czech, Greek, Romanian and Slovenian as mid-resource and Bulgarian, Malayalam, Slovak, Swahili, Telugu and Urdu as low-resource languages. Our distinction between these two groups is based on the number of resources available for each language (in terms of unlabelled and labelled data). 

We included languages from various families (e.g., Indo-European, Dravidian, Sino-Tibetan) and script types in the low-resource category, such as Latin, Arabic, Cyrillic and other non-Latin. The purpose of including multiple scripts and language families in our cross-lingual evaluation is to investigate the ability of the mT0 model to transfer knowledge between more similar and more distant languages with respect to both script and language features. 

To train language representations on unlabelled data, we selected Wikipedia as a source that contains many articles in various languages, including low-resource ones. All Wikipedia data were taken from the latest preprocessed dump from Hugging-Face<sup>2</sup>, from November 2023. 

## 4.3 Tasks

To evaluate the capabilities of mT0-BASE and our proposed method for cross-lingual transfer, we choose four distinct tasks involving various NLP areas to explore the model performance. These tasks differ in the type of the provided output and include question answering (QA), named-entity recognition (NER), natural language inference (NLI), and check-worthy claim detection (CWCD). They were selected based on the availability of datasets for selected languages and to include various NLP tasks related to reading comprehension, recognizing textual entailment, or fact-checking. Table 1 lists the datasets used in our experiments. For Bulgarian, no question answering dataset is available. 

<table><tr><td>Dataset</td><td>Task</td><td>Languages</td><td>Citation</td></tr><tr><td>SQUAD</td><td>QA</td><td>en</td><td>Rajpurkar et al. (2016)</td></tr><tr><td>MLQA</td><td>QA</td><td>ar, de, hi, zh, es, vi</td><td>Lewis et al. (2019)</td></tr><tr><td>XQuAD</td><td>QA</td><td>el, ro</td><td>Artetxe et al. (2020)</td></tr><tr><td>SK-QUAD</td><td>QA</td><td>sk</td><td>Hládek et al. (2023)</td></tr><tr><td>CZECH SQUAD</td><td>QA</td><td>cs</td><td>Macková and Straka (2020)</td></tr><tr><td>TeQuAD</td><td>QA</td><td>te</td><td>Vemula et al. (2022)</td></tr><tr><td>KenSWQuAD</td><td>QA</td><td>sw</td><td>Wanjawa et al. (2023)</td></tr><tr><td>UQA</td><td>QA</td><td>ur</td><td>Arif et al. (2024)</td></tr><tr><td>Slovene SQuAD</td><td>QA</td><td>sl</td><td>Borovič et al. (2022)</td></tr><tr><td>IndicQA</td><td>QA</td><td>ml</td><td>Doddapaneni et al. (2023)</td></tr><tr><td>WikiANN</td><td>NER</td><td>ar, bg, cs, de, el,en, es, ml, ro, ru,sl, sk, sw, te, ur, zh</td><td>Rahimi et al. (2019)</td></tr><tr><td>XNLI</td><td>NLI</td><td>ar, bg, de, el, en,es, ru, sw, ur, zh</td><td>Conneau et al. (2018)</td></tr><tr><td>IndicXNLI</td><td>NLI</td><td>ml, te</td><td>Aggarwal et al. (2022)</td></tr><tr><td>CS ANLI</td><td>NLI</td><td>cs, sk*</td><td>CS-ANLI</td></tr><tr><td>RoNLI</td><td>NLI</td><td>ro</td><td>Poesina et al. (2024)</td></tr><tr><td>Sl-NLI</td><td>NLI</td><td>sl</td><td>Klemen et al. (2024)</td></tr><tr><td>MultiClaim</td><td>CWCD</td><td>ar, bg, cs, de*, el*, en,es, ml*, ro*, ru*, sl*,sk, sw*, te*, ur*, zh*</td><td>Pikuliak et al. (2023)Hyben et al. (2023)</td></tr></table>


Table 1: The list of datasets used in our experiments. Languages marked with * represent language versions of datasets that are not original but were obtained by translating texts from Czech (CS ANLI) or English (MultiClaim).


Due to the absence of datasets for some languages, we employed Google Translate API to translate data for several languages. This concerns, in particular, the dataset for the Slovak NLI task and the dataset for check-worthy claim detection. In the case of the missing Slovak NLI dataset, we utilized the CS ANLI dataset and translated it from Czech to Slovak. For check-worthy claim detection, we translated the English dataset into multiple languages to obtain results for comparison.<sup>3</sup> 

## 4.4 Experimental Setup

Language Representations. Language adapters and soft prompts were trained using a span corruption objective with different learning rates for training language adapters and soft language prompts, which were identified based on experiments on English data. Detailed parameters are listed in Table 2 in Appendix D. 

Task Representations. In training task representations, we divided the training set into training and validation splits using 15% of the records for validation, which was done only for datasets that do not include a test set and the original validation split was considered a test set. This is especially the case for the question answering and check-worthy claim detection tasks. Secondly, we preprocessed each dataset by transforming each record from the particular dataset into the text-to-text format employing prompt templates listed in Appendix B. 

Task representations in all configurations were trained using the same training parameters across all tasks, with differences only between learning rates and weight decay.<sup>4</sup> In addition, the instruction used for training soft prompts differs across languages and tasks. These variations are based on the language in which the answer is to be generated and the task that the LLM is solving. 

The best model was chosen based on the performance on the validation split with respect to the loss. For classification tasks, we set the maximum number of tokens to generate based on the predicted classes. This minimizes the problem of the LLM continuing to generate an answer and enables us to evaluate the LLM’s performance correctly. Table 2 in Appendix D shows the exact parameters for training language and task representations. 

Evaluation. For evaluation, we selected several standard metrics employed for particular tasks. Specifically, we use the F1-Score or Accuracy for classification tasks and QA in the SQUAD format. Besides the F1-Score for QA, we also calculated Exact Match, assessing how many of the answers exactly match the ground truth.<sup>5</sup> For the evaluation, we employed metrics implemented in the Hugging Face evaluate library<sup>6</sup>. 

We evaluated the results on cross-lingual transfer from high-resource languages to mid- and lowresource ones, where task representations were trained on datasets in high-resource languages. We aim to assess the combination of language representations of low-resource languages with task representations trained on datasets from high-resource languages, i.e., high-resource language as source language and low-resource as target ones. Extended results are shown in Appendix E. 

Baselines. To evaluate the proposed methods, we employed several baseline approaches and configurations. Baselines include task adapters, soft task prompts (prompt-tuning approach), and MAD-X – the combination of a language and task adapter (Pfeiffer et al., 2020). These baselines provided a foundation for assessing the effectiveness of cross-lingual transfer in our experiments. 

## 5 Results and Analyis

This section summarizes our observations based on the experiments that were conducted. We provide the overall findings in Section 5.1 and then the results for particular tasks, especially question answering (Section 5.2), named entity recognition (Section 5.3), natural language inference (Section 5.4) and check-worthy claim detection tasks (Section 5.5). 

## 5.1 Overall Results

Our study on cross-lingual transfer performance between high-resource and mid- and low-resource languages is summarized in Table 3 in Appendix E, which reports averaged metrics across four tasks for mid- and low-resource languages. Additionally, Figure 3 demonstrates the comparison of all combinations across high-resource languages, where the presented scores represent the average calculated across all tasks and all low-resource languages. 

The results demonstrate that the selection of source languages plays an important role in the overall results, with distinct languages demonstrating different performance gains. Using English as a source language yielded the highest performance for most mid- and low-resource languages when employing task representations alone. A possible explanation might be that multilingual models often remain biased toward the source language, even after adaptation, as demonstrated in Alabi et al. (2024). They show that language adaptation in models primarily occurs in the final layers, while the source language still influences earlier predictions. However, for Bulgarian and Slovak, the combination of soft language prompts with task adapters proved to be more effective. 

In contrast, when Arabic, German, Spanish, and Russian were used as source languages, configurations that combined language and task representations yielded superior scores. Specifically, transferring knowledge from Spanish using a combination of soft language prompts and task adapters resulted in the highest overall performance. This configuration proved to be the most effective, making Spanish the most effective highresource language for cross-lingual transfer between languages across various scripts. The superior performance of Spanish can be attributed to its performance gain in both NER and CWCD tasks compared to other high-resource languages. For instance, in NER, the gap between the best and second-best averaged performance was approximately 4%. The most significant improvement was observed when transferring knowledge to Roma nian, likely due to the linguistic similarity between Spanish and Romanian. Additionally, Spanish also shares similarities with some Slavic languages, further contributing to its effectiveness in cross-lingual transfer. 

## 5.2 Question Answering

Our experiments (see Table 4) revealed that the configuration of a soft language prompt and task adapter achieved the highest performance in many cases in the QA task when transferring to mid- and low-resource languages, with only small differences across languages. This configuration was particularly effective for Greek, Romanian and Slovak, while for Telugu and Urdu, the task adapter without language representation outperformed other configurations. This suggests that the complexity of the target language cannot be sufficiently modelled based on the small number of Wikipedia articles in those languages. Furthermore, English excelled across the board, particularly with Latin and Greek scripts, showcasing its adaptability in cross-lingual transfer. 

In addition to investigating the effects of individual configurations, we also evaluated the improvement of a soft language prompt combined with a task adapter over the original mT0-BASE model without any language or task representations (see Figure 4). Figure 4 contains relative F1- Score improvements and demonstrates that training task representations in English and evaluating in other languages provide the most evident improvement. We also observed that German, English, and Spanish improved performance for most low-resource languages except Telugu and Malayalam. In contrast, Arabic, Russian and Chinese, which have different scripts, exhibited negative transfer across all cases, with Arabic and Chinese offering no improvement for any languages. We conjecture that the cross-lingual transfer depends on the script used for the language, where we achieved the highest performance for languages in the Latin script. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/6cec3d6f-b612-41bf-b581-12cb1790085d/6fd203c33b7710ed17d61e255dc23b5cf49b90bb5ec11d0724d7fae6426dd14a.jpg)



Figure 3: Average performance for the cross-lingual transfer from six high-resource to all low-resource languages, averaged over all low-resource languages. The graph compares different configurations with varying performance for cross-lingual transfer from high-resource to low-resource languages. In most cases, combining soft language prompts with task adapters (purple) proved best.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/6cec3d6f-b612-41bf-b581-12cb1790085d/4ecd3fbaab65495e4235465033e88675f7d08b082e436bcaf9844ad8fdc04ba2.jpg)



Figure 4: Relative F1 improvement for the QA task in transferring knowledge between languages using soft language prompts and task adapters. The effectiveness of the selected configuration is compared with the results obtained without using any language and task representations (i.e., mT0-BASE inference).


## 5.3 Named Entity Recognition

In the case of the NER task, Arabic, German, Spanish and Russian, among high-resource languages, performed best in cross-lingual transfer to mid- and low-resource languages, while English and Chinese performed poorly. However, based on Table 5, the best improvements were observed using a soft language prompt with a task adapter, outperforming the other combinations for languages such as Arabic, Spanish and German. This is especially the case for Telugu, where the difference between these configurations is more than 37% in favor of the combination of soft language prompt and task adapter using Russian data. 

## 5.4 Natural Language Inference

The cross-lingual evaluation of the NLI task from Table 6 demonstrated the effectiveness of almost all proposed configurations for knowledge transfer. In particular, we mostly achieve superior results using the combination of language adapters with soft task prompts in Czech, Slovenian and Slovak as target languages. Meanwhile, the best performance was achieved for Swahili, Telugu, and Urdu without employing language representations. Furthermore, the high effect on the Romanian language observed in the cross-lingual evaluation is probably because Romanian has been involved during the training of mT5, but not as part of finetuning the mT0 model. 

We observed improvement for most languages across the six proposed configurations for transferring knowledge. However, for Czech, Slovenian and Slovak, several configurations resulted in lower performance compared to inference-only baselines. Notably, for Slovenian, using Russian for the soft task prompt was the only configuration that outperformed the inference-only approach. Furthermore, the combination of language and task adapters for Slovenian resulted in the poorest performance, with an average deterioration of 63%. 

## 5.5 Check-Worthy Claim Detection

For check-worthy claim detection, the configuration of soft language prompts and task adapters performs comparably to methods without language representations (see Table 7). When considering both the best and second-best results, this combination proves effective across most language pairs, demonstrating the model’s enhanced capabilities for check-worthy claim detection. Notably, using 

Spanish for knowledge transfer within this setup resulted in the highest performance gains. 

## 6 Discussion

Based on our experiments, we summarize our observations below. 

Prompt Tuning Performs Better with Fine-Tuned Models. In our preliminary model selection experiments, we found that prompt tuning does not improve the performance of pre-trained LLMs (e.g., mT5) trained only on unlabelled data for downstream tasks. However, prompt-tuning can enhance the performance of already fine-tuned LLMs on any labelled data, even if the specific tasks were not part of the prior fine-tuning. This was confirmed in our experiments with NER and check-worthy claim detection, where fine-tuned LLMs delivered superior results despite no previous task-specific training on these tasks. 

Soft Language Prompts with Task Adapters Perform Best in Many Cases. Our approach of combining soft language prompts with task adapters demonstrated better performance in many cases, compared to the approach of combined language and task adapters, which has been shown to be very effective in previous work. Specifically, the combination of soft language prompts and task adapters is most effective on the classification tasks, achieving superior results most often. For languages with a different script (e.g., Spanish and Telugu), these differences were over 20%. 

Language Representations are Unable to Capture Linguistic Characteristics Using Small Number of Unlabelled Data. Language representations have several limitations that led to configurations without language representations performing consistently better on cross-lingual transfer to highly low-resource languages, such as Telugu, Urdu, and Malayalam. We postulate that the reason is the small number of Wikipedia articles on which the language representations were trained, rendering them unable to adequately capture sufficient linguistic characteristics. 

## 7 Conclusion

Our study provides a comprehensive evaluation of various configurations of adapters and soft prompts for cross-lingual transfer in mid- and low-resource languages. With the systematic evaluation of task adapters, soft task prompts, and combinations of language and task representations, we identified configurations that positively affect LLM’s performance across different tasks and languages. Our findings demonstrated that the combination of soft language prompts and task adapters emerged as an effective alternative for transferring knowledge between languages. Furthermore, our findings provide valuable insights for the utilization of a combination of PEFT methods for cross-lingual transfer, while highlighting the need to incorporate language-specific knowledge. 

## Limitations

Model Selection. Our analysis of the effectiveness of the language and task representations focused on highly multilingual LLMs that include a wider variety of low-resource languages. From this perspective, there is not a vast number of opensource multilingual LLMs with such extensive language coverage as the mT5 or BLOOM model, while having fewer than 1B parameters. We also considered the AYA model (Üstün et al., 2024), but due to limited computational resources, it was not feasible to conduct our experiments. Another aspect of the selection was the involvement of only generative models consisting of encoder-decoder or decoder-only architecture. 

Other Languages. In selecting appropriate languages, we were limited by the languages covered by the mT5 model. To select high-resource languages, we considered languages that are the most extensive in terms of available resources and are in different scripts, e.g., not only Latin script. On the other hand, when selecting mid- and lowresource languages, we also considered the availability of datasets in multiple languages from different language families as well as the availability of datasets in those languages (both human-annotated and machine-translated). 

Other Tasks. The tasks in our experiments were selected based on the availability of datasets for each selected language and covered multiple areas of the NLP domain, i.e., reading comprehension, fact-checking, and recognizing textual entailment. We mostly considered tasks involved in the instruction fine-tuning of the mT0-model, but we also included tasks that were not originally used to train mT0, e.g., NER and check-worthy claim detection. 

## Acknowledgements

We thank the anonymous reviewers for their feedback; and especially for spending their time through several iterations of the ARR cycles. This research was partially supported by DisAI - Improving scientific excellence and creativity in combating disinformation with artificial intelligence and language technologies, a project funded by Horizon Europe under GA No.101079164, by MIMEDIS, a project funded by the Slovak Research and Development Agency under GA No. APVV-21-0114, by the Ministry of Education, Youth and Sports of the Czech Republic through the e-INFRA CZ (ID:90254), and by the German Ministry of Education and Research (BMBF) as part of the project TRAILS (01IW24005). 

## References



CS ANLI. https://huggingface.co/datasets/ ctu-aic/anli_cs. Accessed: 2024-05-30. 





Divyanshu Aggarwal, Vivek Gupta, and Anoop Kunchukuttan. 2022. IndicXNLI: Evaluating multilingual inference for Indian languages. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 10994–11006, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics. 





Jesujoba Alabi, Marius Mosbach, Matan Eyal, Dietrich Klakow, and Mor Geva. 2024. The hidden space of transformer language adapters. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 6588–6607, Bangkok, Thailand. Association for Computational Linguistics. 





Alan Ansell, Edoardo Ponti, Anna Korhonen, and Ivan Vulic. 2022.´ Composable sparse fine-tuning for crosslingual transfer. In Proceedings ofthe 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1778–1796, Dublin, Ireland. Association for Computational Linguistics. 





Alan Ansell, Edoardo Maria Ponti, Jonas Pfeiffer, Sebastian Ruder, Goran Glavaš, Ivan Vulic, and Anna´ Korhonen. 2021. MAD-G: Multilingual adapter generation for efficient cross-lingual transfer. In Find ings of the Association for Computational Linguistics: EMNLP 2021, pages 4762–4781, Punta Cana, Dominican Republic. Association for Computational Linguistics. 





Samee Arif, Sualeha Farid, Awais Athar, and Agha Ali Raza. 2024. UQA: Corpus for Urdu question answering. In Proceedings ofthe 2024 Joint International Conference on Computational Linguistics, Language 





Resources and Evaluation (LREC-COLING 2024), pages 17237–17244, Torino, Italia. ELRA and ICCL. 





Mikel Artetxe, Sebastian Ruder, and Dani Yogatama. 2020. On the cross-lingual transferability of monolingual representations. In Proceedings of the 58th Annual Meeting ofthe Associationfor Computational Linguistics, pages 4623–4637, Online. Association for Computational Linguistics. 





Akari Asai, Mohammadreza Salehi, Matthew Peters, and Hannaneh Hajishirzi. 2022. ATTEMPT: Parameter-efficient multi-task tuning via attentional mixtures of soft prompts. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 6655–6672, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics. 





Mladen Borovic, Kristjan Žagar, Marko Ferme, Sandiˇ Majninger, Milan Ojsteršek, Uroš Šmajdek, Maj Zirkelbach, Matjaž Zupanic, Meta Jazbinšek, Slavkoˇ Žitnik, et al. 2022. Slovene translation of the squad2. 0 dataset. 





Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 2020. Unsupervised cross-lingual representation learning at scale. In Proceedings of the 58th Annual Meeting of the Associationfor Computational Linguistics, pages 8440– 8451, Online. Association for Computational Linguistics. 





Alexis Conneau, Ruty Rinott, Guillaume Lample, Adina Williams, Samuel R. Bowman, Holger Schwenk, and Veselin Stoyanov. 2018. Xnli: Evaluating crosslingual sentence representations. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics. 





Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. 2023. Qlora: Efficient finetuning of quantized llms. In Advances in Neural Information Processing Systems, volume 36, pages 10088–10115. Curran Associates, Inc. 





Sumanth Doddapaneni, Rahul Aralikatte, Gowtham Ramesh, Shreya Goyal, Mitesh M. Khapra, Anoop Kunchukuttan, and Pratyush Kumar. 2023. Towards leaving no Indic language behind: Building monolingual corpora, benchmark and models for Indic languages. In Proceedings ofthe 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 12402–12426, Toronto, Canada. Association for Computational Linguistics. 





Daniil Gurgurov, Mareike Hartmann, and Simon Ostermann. 2024. Adapting multilingual LLMs to low-resource languages with knowledge graphs via adapters. In Proceedings of the 1st Workshop on Knowledge Graphs and Large Language Models (KaLLM 2024), pages 63–74, Bangkok, Thailand. Association for Computational Linguistics. 





Junxian He, Chunting Zhou, Xuezhe Ma, Taylor Berg-Kirkpatrick, and Graham Neubig. 2022. Towards a unified view of parameter-efficient transfer learning. Preprint, arXiv:2110.04366. 





Daniel Hládek, Ján Staš, Jozef Juhár, and Tomáš Koc túr. 2023. Slovak dataset for multilingual question answering. IEEE Access, 11:32869–32881. 





Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. 2019a. Parameter-efficient transfer learning for nlp. In International conference on machine learning, pages 2790–2799. PMLR. 





Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. 2019b. Parameter-efficient transfer learning for NLP. In Proceedings ofthe 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pages 2790–2799. PMLR. 





Junjie Hu, Sebastian Ruder, Aditya Siddhant, Graham Neubig, Orhan Firat, and Melvin Johnson. 2020. XTREME: A massively multilingual multitask benchmark for evaluating cross-lingual generalisation. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings ofMachine Learning Research, pages 4411–4421. PMLR. 





Lianzhe Huang, Shuming Ma, Dongdong Zhang, Furu Wei, and Houfeng Wang. 2022. Zero-shot crosslingual transfer of prompt-based tuning with a unified multilingual prompt. In Proceedings ofthe 2022 Conference on Empirical Methods in Natural Language Processing, pages 11488–11497, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics. 





Martin Hyben, Sebastian Kula, Ivan Srba, Robert Moro, and Jakub Simko. 2023. Is it indeed bigger better? the comprehensive study of claim detection lms applied for disinformation tackling. Preprint, arXiv:2311.06121. 





Matej Klemen, Aleš Žagar, Jaka Cibej, and Marko<sup>ˇ</sup> Robnik-Šikonja. 2024. SI-NLI: A Slovene natural language inference dataset and its evaluation. In Proceedings ofthe 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 14859– 14870, Torino, Italia. ELRA and ICCL. 





Mateusz Klimaszewski, Piotr Andruszkiewicz, and Alexandra Birch. 2024. No train but gain: Lan guage arithmetic for training-free language adapters enhancement. Preprint, arXiv:2404.15737. 





Jenny Kunz and Oskar Holmström. 2024. The impact of language adapters in cross-lingual transfer for nlu. Preprint, arXiv:2402.00149. 





Jaeseong Lee, Seung-won Hwang, and Taesup Kim. 2022. FAD-X: Fusing adapters for cross-lingual transfer to low-resource languages. In Proceedings of the 2nd Conference ofthe Asia-Pacific Chapter ofthe Association for Computational Linguistics and the 12th International Joint Conference on Natural Language Processing (Volume 2: Short Papers), pages 57–64, Online only. Association for Computational Linguistics. 





Brian Lester, Rami Al-Rfou, and Noah Constant. 2021. The power of scale for parameter-efficient prompt tuning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 3045–3059, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics. 





Patrick Lewis, Barlas Oguz, Ruty Rinott, Sebastian Riedel, and Holger Schwenk. 2019. Mlqa: Evaluating cross-lingual extractive question answering. arXiv preprint arXiv:1910.07475, arXiv: 1910.07475. 





Lei Liu and Jimmy Xiangji Huang. 2023. Prompt learning to mitigate catastrophic forgetting in cross-lingual transfer for open-domain dialogue generation. In Proceedings ofthe 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’23, page 2287–2292, New York, NY, USA. Association for Computing Machinery. 





Yun Luo, Zhen Yang, Fandong Meng, Yafu Li, Jie Zhou, and Yue Zhang. 2024. An empirical study of catastrophic forgetting in large language models during continual fine-tuning. Preprint, arXiv:2308.08747. 





Kateˇrina Macková and Milan Straka. 2020. Reading comprehension in czech via machine translation and cross-lingual transfer. In Text, Speech, and Dialogue, pages 171–179, Cham. Springer International Publishing. 





Michael McCloskey and Neal J. Cohen. 1989. Catastrophic interference in connectionist networks: The sequential learning problem. volume 24 of Psychology ofLearning and Motivation, pages 109–165. Academic Press. 





Niklas Muennighoff, Thomas Wang, Lintang Sutawika, Adam Roberts, Stella Biderman, Teven Le Scao, M Saiful Bari, Sheng Shen, Zheng-Xin Yong, Hailey Schoelkopf, Xiangru Tang, Dragomir Radev, Alham Fikri Aji, Khalid Almubarak, Samuel Albanie, Zaid Alyafeai, Albert Webson, Edward Raff, and Colin Raffel. 2023. Crosslingual generalization through multitask finetuning. Preprint, arXiv:2211.01786. 





Marinela Parovic, Goran Glavaš, Ivan Vuli´ c, and Anna´ Korhonen. 2022. BAD-X: Bilingual adapters improve zero-shot cross-lingual transfer. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 





1791–1799, Seattle, United States. Association for Computational Linguistics. 





Jonas Pfeiffer, Naman Goyal, Xi Lin, Xian Li, James Cross, Sebastian Riedel, and Mikel Artetxe. 2022. Lifting the curse of multilinguality by pre-training modular transformers. In Proceedings of the 2022 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies, pages 3479–3495, Seattle, United States. Association for Computational Linguistics. 





Jonas Pfeiffer, Francesco Piccinno, Massimo Nicosia, Xinyi Wang, Machel Reid, and Sebastian Ruder. 2023. mmT5: Modular multilingual pre-training solves source language hallucinations. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 1978–2008, Singapore. Association for Computational Linguistics. 





Jonas Pfeiffer, Ivan Vulic, Iryna Gurevych, and Se-´ bastian Ruder. 2020. MAD-X: An Adapter-Based Framework for Multi-Task Cross-Lingual Transfer. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 7654–7673, Online. Association for Computational Linguistics. 





Fred Philippy, Siwen Guo, Shohreh Haddadan, Cedric Lothritz, Jacques Klein, and Tegawendé F. Bissyandé. 2024. Soft prompt tuning for cross-lingual transfer: When less is more. In Proceedings of the 1st Workshop on Modular and Open Multilingual NLP (MOOMIN 2024), pages 7–15, St Julians, Malta. Association for Computational Linguistics. 





Matúš Pikuliak, Ivan Srba, Robert Moro, Timo Hromadka, Timotej Smolen, Martin Melisek, Ivan Vykopal, Jakub Simko, Juraj Podrouzek, and Maria Bielikova. 2023. Multilingual previously factchecked claim retrieval. Preprint, arXiv:2305.07991. 





Matúš Pikuliak, Marián Šimko, and Mária Bieliková. 2021. Cross-lingual learning for text processing: A survey. Expert Systems with Applications, 165:113765. 





Eduard Poesina, Cornelia Caragea, and Radu Ionescu. 2024. A novel cartography-based curriculum learning method applied on RoNLI: The first Romanian natural language inference corpus. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 236–253, Bangkok, Thailand. Association for Computational Linguistics. 





Edoardo Maria Ponti, Goran Glavaš, Olga Majewska, Qianchu Liu, Ivan Vulic, and Anna Korhonen. 2020.´ Xcopa: A multilingual dataset for causal commonsense reasoning. Preprint, arXiv:2005.00333. 





Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2020. Exploring the limits 





of transfer learning with a unified text-to-text transformer. J. Mach. Learn. Res., 21(1). 





Afshin Rahimi, Yuan Li, and Trevor Cohn. 2019. Massively multilingual transfer for NER. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 151–164, Florence, Italy. Association for Computational Linguistics. 





Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. SQuAD: 100,000+ questions for machine comprehension of text. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 2383–2392, Austin, Texas. Association for Computational Linguistics. 





Vipul Rathore, Rajdeep Dhingra, Parag Singla, and Mausam. 2023. ZGUL: Zero-shot generalization to unseen languages using multi-source ensembling of language adapters. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 6969–6987, Singapore. Association for Computational Linguistics. 





Weijieying Ren, Xinlong Li, Lei Wang, Tianxiang Zhao, and Wei Qin. 2024. Analyzing and reducing catastrophic forgetting in parameter efficient tuning. Preprint, arXiv:2402.18865. 





Yuqing Tang, Chau Tran, Xian Li, Peng-Jen Chen, Naman Goyal, Vishrav Chaudhary, Jiatao Gu, and Angela Fan. 2020. Multilingual translation with extensible multilingual pretraining and finetuning. Preprint, arXiv:2008.00401. 





Ahmet Üstün, Arianna Bisazza, Gosse Bouma, and Gertjan van Noord. 2020. UDapter: Language adaptation for truly Universal Dependency parsing. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 2302–2315, Online. Association for Computational Linguistics. 





Rakesh Vemula, Mani Nuthi, and Manish Srivastava. 2022. TeQuAD:Telugu question answering dataset. In Proceedings ofthe 19th International Conference on Natural Language Processing (ICON), pages 300– 307, New Delhi, India. Association for Computational Linguistics. 





Tu Vu, Aditya Barua, Brian Lester, Daniel Cer, Mohit Iyyer, and Noah Constant. 2022a. Overcoming catastrophic forgetting in zero-shot cross-lingual generation. In Proceedings ofthe 2022 Conference on Empirical Methods in Natural Language Processing, pages 9279–9300, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics. 





Tu Vu, Brian Lester, Noah Constant, Rami Al-Rfou’, and Daniel Cer. 2022b. SPoT: Better frozen model adaptation through soft prompt transfer. In Proceedings ofthe 60th Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), pages 5039–5059, Dublin, Ireland. Association for Computational Linguistics. 





Bailin Wang, Zi Wang, Xuezhi Wang, Yuan Cao, Rif A. Saurous, and Yoon Kim. 2024. Grammar prompting for domain-specific language generation with large language models. In Proceedings ofthe 37th Interna tional Conference on Neural Information Processing Systems, NIPS ’23, Red Hook, NY, USA. Curran Associates Inc. 





Barack W. Wanjawa, Lilian D. A. Wanzare, Florence Indede, Owen Mconyango, Lawrence Muchemi, and Edward Ombui. 2023. Kenswquad—a question answering dataset for swahili low-resource language. ACM Trans. Asian Low-Resour. Lang. Inf. Process., 22(4). 





BigScience Workshop, :, Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilic, Daniel´ Hesslow, Roman Castagné, Alexandra Sasha Luc cioni, François Yvon, Matthias Gallé, Jonathan Tow, Alexander M. Rush, Stella Biderman, Albert Webson, Pawan Sasanka Ammanamanchi, Thomas Wang, Benoît Sagot, Niklas Muennighoff, Albert Vil lanova del Moral, Olatunji Ruwase, Rachel Bawden, Stas Bekman, Angelina McMillan-Major, Iz Belt agy, Huu Nguyen, Lucile Saulnier, Samson Tan, Pe dro Ortiz Suarez, Victor Sanh, Hugo Laurençon, Yacine Jernite, Julien Launay, Margaret Mitchell, Colin Raffel, Aaron Gokaslan, Adi Simhi, Aitor Soroa, Alham Fikri Aji, Amit Alfassy, Anna Rogers, Ariel Kreisberg Nitzav, Canwen Xu, Chenghao Mou, Chris Emezue, Christopher Klamm, Colin Leong, Daniel van Strien, David Ifeoluwa Adelani, Dragomir Radev, Eduardo González Ponferrada, Efrat Lev kovizh, Ethan Kim, Eyal Bar Natan, Francesco De Toni, Gérard Dupont, Germán Kruszewski, Giada Pistilli, Hady Elsahar, Hamza Benyamina, Hieu Tran, Ian Yu, Idris Abdulmumin, Isaac Johnson, Itziar Gonzalez-Dios, Javier de la Rosa, Jenny Chim, Jesse Dodge, Jian Zhu, Jonathan Chang, Jörg Frohberg, Joseph Tobing, Joydeep Bhattacharjee, Khalid Al mubarak, Kimbo Chen, Kyle Lo, Leandro Von Werra, Leon Weber, Long Phan, Loubna Ben allal, Lu dovic Tanguy, Manan Dey, Manuel Romero Muñoz, Maraim Masoud, María Grandury, Mario Šaško, Max Huang, Maximin Coavoux, Mayank Singh, Mike Tian-Jian Jiang, Minh Chien Vu, Moham mad A. Jauhar, Mustafa Ghaleb, Nishant Subramani, Nora Kassner, Nurulaqilla Khamis, Olivier Nguyen, Omar Espejel, Ona de Gibert, Paulo Villegas, Pe ter Henderson, Pierre Colombo, Priscilla Amuok, Quentin Lhoest, Rheza Harliman, Rishi Bommasani, Roberto Luis López, Rui Ribeiro, Salomey Osei, Sampo Pyysalo, Sebastian Nagel, Shamik Bose, Shamsuddeen Hassan Muhammad, Shanya Sharma, Shayne Longpre, Somaieh Nikpoor, Stanislav Silber berg, Suhas Pai, Sydney Zink, Tiago Timponi Tor rent, Timo Schick, Tristan Thrush, Valentin Danchev, Vassilina Nikoulina, Veronika Laippala, Violette Lepercq, Vrinda Prabhu, Zaid Alyafeai, Zeerak Ta lat, Arun Raja, Benjamin Heinzerling, Chenglei Si, Davut Emre Ta¸sar, Elizabeth Salesky, Sabrina J. Mielke, Wilson Y. Lee, Abheesht Sharma, Andrea Santilli, Antoine Chaffin, Arnaud Stiegler, Debajy oti Datta, Eliza Szczechla, Gunjan Chhablani, Han 



Wang, Harshit Pandey, Hendrik Strobelt, Jason Alan Fries, Jos Rozen, Leo Gao, Lintang Sutawika, M Sai ful Bari, Maged S. Al-shaibani, Matteo Manica, Ni hal Nayak, Ryan Teehan, Samuel Albanie, Sheng Shen, Srulik Ben-David, Stephen H. Bach, Taewoon Kim, Tali Bers, Thibault Fevry, Trishala Neeraj, Ur mish Thakker, Vikas Raunak, Xiangru Tang, Zheng Xin Yong, Zhiqing Sun, Shaked Brody, Yallow Uri Hadar Tojarieh, Adam Roberts, Hyung Won Chung Jaesung Tae, Jason Phang, Ofir Press, Conglong Li, Deepak Narayanan, Hatim Bourfoune, Jared Casper Jeff Rasley, Max Ryabinin, Mayank Mishra, Minjia Zhang, Mohammad Shoeybi, Myriam Peyrounette Nicolas Patry, Nouamane Tazi, Omar Sanseviero Patrick von Platen, Pierre Cornette, Pierre François Lavallée, Rémi Lacroix, Samyam Rajbhandari, San chit Gandhi, Shaden Smith, Stéphane Requena, Suraj Patil, Tim Dettmers, Ahmed Baruwa, Amanpree Singh, Anastasia Cheveleva, Anne-Laure Ligozat, Arjun Subramonian, Aurélie Névéol, Charles Lover ing, Dan Garrette, Deepak Tunuguntla, Ehud Reiter, Ekaterina Taktasheva, Ekaterina Voloshina, Eli Bog danov, Genta Indra Winata, Hailey Schoelkopf, Jan Christoph Kalo, Jekaterina Novikova, Jessica Zosa Forde, Jordan Clive, Jungo Kasai, Ken Kawamura Liam Hazan, Marine Carpuat, Miruna Clinciu, Na joung Kim, Newton Cheng, Oleg Serikov, Omer Antverg, Oskar van der Wal, Rui Zhang, Ruochen Zhang, Sebastian Gehrmann, Shachar Mirkin, Shan Pais, Tatiana Shavrina, Thomas Scialom, Tian Yun Tomasz Limisiewicz, Verena Rieser, Vitaly Protasov Vladislav Mikhailov, Yada Pruksachatkun, Yonatan Belinkov, Zachary Bamberger, Zdenek Kasner, Al-ˇ ice Rueda, Amanda Pestana, Amir Feizpour, Ammar Khan, Amy Faranak, Ana Santos, Anthony Hevia Antigona Unldreaj, Arash Aghagol, Arezoo Abdol lahi, Aycha Tammour, Azadeh HajiHosseini, Bahareh Behroozi, Benjamin Ajibade, Bharat Saxena, Car los Muñoz Ferrandis, Daniel McDuff, Danish Con tractor, David Lansky, Davis David, Douwe Kiela Duong A. Nguyen, Edward Tan, Emi Baylor, Ez inwanne Ozoani, Fatima Mirza, Frankline Onon iwu, Habib Rezanejad, Hessie Jones, Indrani Bhat tacharya, Irene Solaiman, Irina Sedenko, Isar Ne jadgholi, Jesse Passmore, Josh Seltzer, Julio Bonis Sanz, Livia Dutra, Mairon Samagaio, Maraim El badri, Margot Mieskes, Marissa Gerchick, Martha Akinlolu, Michael McKenna, Mike Qiu, Muhammed Ghauri, Mykola Burynok, Nafis Abrar, Nazneen Ra jani, Nour Elkott, Nour Fahmy, Olanrewaju Samuel Ran An, Rasmus Kromann, Ryan Hao, Samira Al izadeh, Sarmad Shubber, Silas Wang, Sourav Roy Sylvain Viguier, Thanh Le, Tobi Oyebade, Trieu Le Yoyo Yang, Zach Nguyen, Abhinav Ramesh Kashyap, Alfredo Palasciano, Alison Callahan, Anima Shukla Antonio Miranda-Escalada, Ayush Singh, Benjamin Beilharz, Bo Wang, Caio Brito, Chenxi Zhou, Chirag Jain, Chuxin Xu, Clémentine Fourrier, Daniel León Periñán, Daniel Molano, Dian Yu, Enrique Manjava cas, Fabio Barth, Florian Fuhrimann, Gabriel Altay Giyaseddin Bayrak, Gully Burns, Helena U. Vrabec Imane Bello, Ishani Dash, Jihyun Kang, John Giorgi Jonas Golde, Jose David Posada, Karthik Ranga sai Sivaraman, Lokesh Bulchandani, Lu Liu, Luisa 

Shinzato, Madeleine Hahn de Bykhovetz, Maiko Takeuchi, Marc Pàmies, Maria A Castillo, Marianna Nezhurina, Mario Sänger, Matthias Samwald, Michael Cullan, Michael Weinberg, Michiel De Wolf, Mina Mihaljcic, Minna Liu, Moritz Freidank, Myungsun Kang, Natasha Seelam, Nathan Dahlberg, Nicholas Michio Broad, Nikolaus Muellner, Pascale Fung, Patrick Haller, Ramya Chandrasekhar, Renata Eisenberg, Robert Martin, Rodrigo Canalli, Rosaline Su, Ruisi Su, Samuel Cahyawijaya, Samuele Garda, Shlok S Deshmukh, Shubhanshu Mishra, Sid Kiblawi, Simon Ott, Sinee Sang-aroonsiri, Srishti Kumar, Stefan Schweter, Sushil Bharati, Tanmay Laud, Théo Gigant, Tomoya Kainuma, Wojciech Kusa, Yanis Labrak, Yash Shailesh Bajaj, Yash Venkatraman, Yifan Xu, Yingxin Xu, Yu Xu, Zhe Tan, Zhongli Xie, Zifan Ye, Mathilde Bras, Younes Belkada, and Thomas Wolf. 2023. Bloom: A 176b-parameter open-access multilingual language model. Preprint, arXiv:2211.05100. 

Shijie Wu and Mark Dredze. 2020. Are all languages created equal in multilingual bert? Preprint, arXiv:2005.09093. 



Zhihui Xie, Handong Zhao, Tong Yu, and Shuai Li. 2024. Discovering low-rank subspaces for languageagnostic multilingual representations. Preprint, arXiv:2401.05792. 





Lingling Xu, Haoran Xie, Si-Zhao Joe Qin, Xiaohui Tao, and Fu Lee Wang. 2023. Parameter-efficient fine-tuning methods for pretrained language models: A critical review and assessment. Preprint, arXiv:2312.12148. 





Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, and Colin Raffel. 2021. mt5: A massively multilin gual pre-trained text-to-text transformer. Preprint, arXiv:2010.11934. 





Qingru Zhang, Minshuo Chen, Alexander Bukharin, Nikos Karampatziakis, Pengcheng He, Yu Cheng, Weizhu Chen, and Tuo Zhao. 2023. Adalora: Adaptive budget allocation for parameter-efficient finetuning. Preprint, arXiv:2303.10512. 





Ahmet Üstün, Viraat Aryabumi, Zheng-Xin Yong, Wei-Yin Ko, Daniel D’souza, Gbemileke Onilude, Neel Bhandari, Shivalika Singh, Hui-Lee Ooi, Amr Kayid, Freddie Vargus, Phil Blunsom, Shayne Longpre, Niklas Muennighoff, Marzieh Fadaee, Julia Kreutzer, and Sara Hooker. 2024. Aya model: An instruction finetuned open-access multilingual language model. Preprint, arXiv:2402.07827. 



## A Computational Resources

For our experiments, we utilized a computational infrastructure consisting of A10 and A40 NVIDIA GPUs, while our experiments ran in parallel on multiple GPUs. In total, our experiments required around 3,200 GPU hours, ensuring model training and validation for cross-lingual transfer. 

## B Prompts Used

For the purpose of the encoder-decoder model, the record from each dataset needs to be transformed into a text-to-text format. To choose an appropriate prompt format, we experimented with all the prompts used in the mT0 paper (Muennighoff et al., 2023) and with prompts used in the T5 paper (Raffel et al., 2020). Prompts, which achieved the best performance during inference with the mT0-BASE model, were selected for transforming the records into a text-to-text format. In the following paragraphs, there are the prompts for the individual tasks that have been used to convert to text-to-text format. 

## B.1 Question Answering

Template: question: {question} context: {context} 

## B.2 Natural Language Inference

Template: {premise} \n\n Question: Does this imply that "{hypothesis}"? Yes, no, or maybe? 

## B.3 Named-Entity Recognition

Template: tag: {text} 

## B.4 Check-Worthy Claim Detection

Template: checkworthiness claim: {claim} 

## C Soft Prompt Initialization

This section includes templates for soft prompts used for the initialization for each language and each task. Templates are divided into language and task templates. 

## C.1 Language Templates

To train language representation using a language modeling objective, we employed a specific prompt that varied only based on the language present in the instruction, leaving the rest of the instruction the same. 

The template we used for initialization is as follows: "Generate the output in {Language}:", where the Language is replaced by the desired language. 

## C.2 Task Templates

The following are initialization prompt templates for each task, where the instruction depends not only on the task but also on the language. 

Question Answering. For the question answering task, we utilized "Answer the question in {Language} language:", while replacing Language with the desired language. 

Natural Language Inference. Natural language inference is the task of assessing whether a hypothesis logically follows from the premise. It is defined as a classification with three possible classes: entailment, contradiction or neutral. However, based on the previous work and instruction tuning of the mT0 model, we replaced above mentioned classes with Yes, No and Maybe, based on the used prompt template. 

According to the employed classes, we defined an initialization prompt as follows: "Select Yes, No or Maybe based on the implication of the premise on the hypothesis in {Language}:", while Language is replaced by the desired language. 

Named-Entity Recognition. The named-entity recognition task aims to identify named entities within the input text. While there are many possible categories, the WikiANN dataset focuses only on detecting three categories: location (LOC), person (PER) and organization (ORG). Based on the defined classes, we created the initialization prompt as follows: "Identify NER tags (ORG, PER, LOC) in the text in {Language}:", where Language is substituted with the specific language. 

Check-Worthy Claim Detection. The latter task includes check-worthy claim detection, which is a binary classification of assessing whether the given claim is worthy of fact-checking or not. As text labels, we used Not checkworthy and Checkworthy. This is the initialization prompt for the check-worthy claim detection task: Determine whether a given claim in {Language} is checkworthy:", where Language is replaced by the desired language. 

## D Hyperparameters

Table 2 shows hyperparameters used for training language and task representations using adapters and soft prompts. 

## E Cross-Lingual Evaluation

Tables 4 to 7 present the results for transferring knowledge from all high-resource languages to all mid- and low-resource languages. The first row in each table represents the scores obtained by inference of the original mT0-BASE model without additional training of language or task representations. 

## F Evaluation with Multiple Training Seeds

In Table 8, we report the evaluation results of all configurations that were trained on the German version of the WikiANN dataset using three different seeds. Along with the mean values, we also report the standard deviation 

The obtained results demonstrate that the best results for knowledge transfer from German to other languages are obtained by using task adapters for Bulgarian, Greek, Malayalam, Romanian and Swahili. In contrast, the best combination for Czech, Slovenian, Slovak, Telugu and Urdu was a soft language prompt with a task adapter. This observation supports our previous findings that both configurations achieved superior results on the NER task when transferring knowledge from German. 

<table><tr><td rowspan="2">Hyperparameters</td><td colspan="2">Language Modeling</td><td colspan="2">Task Modeling</td></tr><tr><td>Language Adapter</td><td>Soft Language Prompt</td><td>Task Adapter</td><td>Soft Task Prompt</td></tr><tr><td>Learning rate</td><td>5e-5</td><td>5e-1</td><td>5e-5</td><td>5e-1</td></tr><tr><td>Weight decay</td><td>0</td><td>1e-5</td><td>0</td><td>1e-5</td></tr><tr><td>Batch size</td><td>32</td><td>32</td><td>32</td><td>32</td></tr><tr><td>No. Training steps</td><td>100,000</td><td>100,000</td><td>50,000</td><td>50,000</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>Adafactor</td><td>AdamW</td><td>Adafactor</td></tr><tr><td>Evaluation steps</td><td>500</td><td>500</td><td>1000</td><td>1000</td></tr><tr><td>Max input length</td><td>256</td><td>256</td><td>256</td><td>256</td></tr><tr><td>Token size of soft prompt</td><td>NaN</td><td>50</td><td>NaN</td><td>50</td></tr></table>


Table 2: Final parameters employed to train language and task representation using adapters and soft prompts.


<table><tr><td>Task Language</td><td>Language Representation</td><td>Task Representation</td><td>bg</td><td>cs</td><td>el</td><td>ml</td><td>ro</td><td>sl</td><td>sk</td><td>sw</td><td>te</td><td>ur</td><td>Average</td></tr><tr><td rowspan="6">ar</td><td rowspan="2">None</td><td>Adapter</td><td>68.03</td><td>48.98</td><td>65.25</td><td>46.74</td><td>63.44</td><td>48.74</td><td>48.26</td><td>50.86</td><td>53.80</td><td>48.15</td><td>54.23</td></tr><tr><td>Soft Prompt</td><td>64.67</td><td>43.16</td><td>62.66</td><td>47.55</td><td>64.43</td><td>44.60</td><td>40.33</td><td>47.22</td><td>51.44</td><td>43.81</td><td>50.99</td></tr><tr><td rowspan="2">Adapter</td><td>Adapter</td><td>64.82</td><td>42.56</td><td>64.59</td><td>49.82</td><td>61.73</td><td>30.44</td><td>39.92</td><td>48.99</td><td>40.36</td><td>41.03</td><td>48.43</td></tr><tr><td>Soft Prompt</td><td>69.90</td><td>39.84</td><td>63.36</td><td>48.49</td><td>55.78</td><td>55.53</td><td>41.28</td><td>49.52</td><td>41.45</td><td>43.75</td><td>50.89</td></tr><tr><td rowspan="2">Soft Prompt</td><td>Adapter</td><td>71.23</td><td>48.06</td><td>67.22</td><td>48.22</td><td>65.66</td><td>51.68</td><td>48.26</td><td>51.21</td><td>53.54</td><td>49.63</td><td>55.47</td></tr><tr><td>Soft Prompt</td><td>66.73</td><td>41.59</td><td>62.82</td><td>47.66</td><td>58.81</td><td>44.69</td><td>40.02</td><td>44.10</td><td>50.11</td><td>44.72</td><td>50.13</td></tr><tr><td rowspan="6">de</td><td rowspan="2">None</td><td>Adapter</td><td>67.76</td><td>51.72</td><td>71.12</td><td>51.52</td><td>71.55</td><td>54.88</td><td>51.27</td><td>56.78</td><td>57.50</td><td>51.82</td><td>58.59</td></tr><tr><td>Soft Prompt</td><td>65.31</td><td>47.61</td><td>69.55</td><td>51.77</td><td>70.02</td><td>50.97</td><td>46.53</td><td>53.16</td><td>52.91</td><td>46.66</td><td>55.45</td></tr><tr><td rowspan="2">Adapter</td><td>Adapter</td><td>79.98</td><td>53.81</td><td>73.48</td><td>50.39</td><td>66.34</td><td>49.69</td><td>54.57</td><td>54.29</td><td>45.65</td><td>45.88</td><td>57.41</td></tr><tr><td>Soft Prompt</td><td>72.61</td><td>47.19</td><td>68.12</td><td>48.81</td><td>60.14</td><td>56.41</td><td>46.63</td><td>51.20</td><td>44.41</td><td>44.76</td><td>54.03</td></tr><tr><td rowspan="2">Soft Prompt</td><td>Adapter</td><td>70.71</td><td>51.91</td><td>70.83</td><td>50.86</td><td>69.86</td><td>57.76</td><td>53.98</td><td>55.90</td><td>55.90</td><td>52.34</td><td>59.01</td></tr><tr><td>Soft Prompt</td><td>68.98</td><td>49.02</td><td>69.83</td><td>50.98</td><td>69.18</td><td>52.18</td><td>48.36</td><td>54.50</td><td>54.50</td><td>45.04</td><td>56.26</td></tr><tr><td rowspan="6">en</td><td rowspan="2">None</td><td>Adapter</td><td>68.77</td><td>49.64</td><td>70.37</td><td>46.34</td><td>67.29</td><td>52.54</td><td>47.98</td><td>49.70</td><td>52.13</td><td>48.29</td><td>55.31</td></tr><tr><td>Soft Prompt</td><td>64.81</td><td>41.26</td><td>68.02</td><td>46.60</td><td>70.58</td><td>52.32</td><td>40.19</td><td>51.43</td><td>52.43</td><td>50.48</td><td>53.81</td></tr><tr><td rowspan="2">Adapter</td><td>Adapter</td><td>60.90</td><td>46.51</td><td>65.03</td><td>41.80</td><td>72.02</td><td>37.96</td><td>43.30</td><td>46.98</td><td>38.47</td><td>40.09</td><td>49.31</td></tr><tr><td>Soft Prompt</td><td>64.68</td><td>38.29</td><td>65.04</td><td>42.93</td><td>74.50</td><td>54.80</td><td>37.41</td><td>48.36</td><td>37.48</td><td>52.54</td><td>51.60</td></tr><tr><td rowspan="2">Soft Prompt</td><td>Adapter</td><td>68.95</td><td>48.37</td><td>69.39</td><td>43.44</td><td>64.20</td><td>51.34</td><td>50.88</td><td>51.05</td><td>49.16</td><td>45.78</td><td>54.26</td></tr><tr><td>Soft Prompt</td><td>56.33</td><td>47.75</td><td>65.98</td><td>43.22</td><td>58.94</td><td>52.64</td><td>46.48</td><td>48.96</td><td>47.12</td><td>38.62</td><td>50.60</td></tr><tr><td rowspan="6">es</td><td rowspan="2">None</td><td>Adapter</td><td>68.65</td><td>53.61</td><td>70.06</td><td>49.75</td><td>73.83</td><td>56.23</td><td>51.82</td><td>57.47</td><td>57.12</td><td>54.01</td><td>59.26</td></tr><tr><td>Soft Prompt</td><td>62.16</td><td>50.28</td><td>66.95</td><td>47.26</td><td>71.84</td><td>50.34</td><td>49.05</td><td>54.09</td><td>53.94</td><td>49.71</td><td>55.56</td></tr><tr><td rowspan="2">Adapter</td><td>Adapter</td><td>73.89</td><td>50.03</td><td>73.33</td><td>51.15</td><td>70.92</td><td>45.79</td><td>53.04</td><td>54.24</td><td>45.45</td><td>44.63</td><td>56.25</td></tr><tr><td>Soft Prompt</td><td>72.63</td><td>50.19</td><td>64.62</td><td>44.62</td><td>72.70</td><td>47.07</td><td>51.71</td><td>51.95</td><td>39.95</td><td>43.94</td><td>53.94</td></tr><tr><td rowspan="2">Soft Prompt</td><td>Adapter</td><td>75.37</td><td>55.23</td><td>73.50</td><td>51.39</td><td>72.03</td><td>58.88</td><td>54.84</td><td>57.41</td><td>56.90</td><td>54.00</td><td>60.96</td></tr><tr><td>Soft Prompt</td><td>69.07</td><td>52.16</td><td>64.43</td><td>48.19</td><td>72.98</td><td>53.21</td><td>50.93</td><td>53.87</td><td>50.89</td><td>50.26</td><td>56.60</td></tr><tr><td rowspan="6">ru</td><td rowspan="2">None</td><td>Adapter</td><td>82.44</td><td>45.29</td><td>65.74</td><td>46.79</td><td>69.68</td><td>49.24</td><td>45.24</td><td>54.40</td><td>56.48</td><td>52.27</td><td>56.76</td></tr><tr><td>Soft Prompt</td><td>79.26</td><td>41.51</td><td>60.83</td><td>48.64</td><td>68.09</td><td>45.29</td><td>40.70</td><td>50.73</td><td>55.13</td><td>50.94</td><td>54.11</td></tr><tr><td rowspan="2">Adapter</td><td>Adapter</td><td>80.74</td><td>52.52</td><td>73.86</td><td>45.76</td><td>73.75</td><td>40.19</td><td>51.99</td><td>52.08</td><td>39.55</td><td>42.11</td><td>55.26</td></tr><tr><td>Soft Prompt</td><td>77.96</td><td>34.33</td><td>66.75</td><td>44.48</td><td>71.76</td><td>49.84</td><td>36.41</td><td>50.61</td><td>40.14</td><td>45.31</td><td>51.76</td></tr><tr><td rowspan="2">Soft Prompt</td><td>Adapter</td><td>83.96</td><td>44.69</td><td>70.25</td><td>49.87</td><td>72.15</td><td>51.06</td><td>47.34</td><td>55.21</td><td>56.89</td><td>53.62</td><td>58.50</td></tr><tr><td>Soft Prompt</td><td>79.68</td><td>44.50</td><td>71.02</td><td>50.63</td><td>70.94</td><td>52.60</td><td>40.99</td><td>52.37</td><td>54.91</td><td>52.04</td><td>56.97</td></tr><tr><td rowspan="6">zh</td><td rowspan="2">None</td><td>Adapter</td><td>61.19</td><td>43.74</td><td>57.03</td><td>40.50</td><td>64.60</td><td>44.42</td><td>42.70</td><td>50.23</td><td>45.65</td><td>44.33</td><td>49.44</td></tr><tr><td>Soft Prompt</td><td>58.20</td><td>45.17</td><td>59.24</td><td>42.23</td><td>66.36</td><td>44.72</td><td>43.38</td><td>46.27</td><td>50.10</td><td>43.72</td><td>49.94</td></tr><tr><td rowspan="2">Adapter</td><td>Adapter</td><td>62.73</td><td>46.45</td><td>61.14</td><td>37.45</td><td>66.19</td><td>35.30</td><td>42.70</td><td>43.73</td><td>35.98</td><td>36.28</td><td>46.80</td></tr><tr><td>Soft Prompt</td><td>49.40</td><td>43.50</td><td>47.04</td><td>42.18</td><td>45.70</td><td>56.09</td><td>36.34</td><td>42.75</td><td>37.22</td><td>39.00</td><td>43.92</td></tr><tr><td rowspan="2">Soft Prompt</td><td>Adapter</td><td>65.97</td><td>45.20</td><td>60.49</td><td>41.62</td><td>65.64</td><td>48.49</td><td>45.52</td><td>52.19</td><td>50.20</td><td>42.72</td><td>51.80</td></tr><tr><td>Soft Prompt</td><td>62.44</td><td>48.74</td><td>61.62</td><td>43.09</td><td>67.48</td><td>49.20</td><td>44.85</td><td>48.53</td><td>51.09</td><td>44.69</td><td>52.17</td></tr></table>


Table 3: Average scores for each configuration across all tasks for low-resource languages. The languages in rows represent the language in which the task representation was trained, and the languages in columns represent the language representation that was used, if any (except for configurations with None in the language representation). For each language pair, the best results are boldfaced and the second best are underlined.


<table><tr><td>Task Language</td><td>Language Representation</td><td>Task Representation</td><td>cs</td><td>el</td><td>ml</td><td>ro</td><td>sl</td><td>sk</td><td>sw</td><td>te</td><td>ur</td><td>Average</td></tr><tr><td></td><td>None</td><td>None</td><td>31.34 (24.78)</td><td>57.00 (47.56)</td><td>1.37 (1.07)</td><td>57.00 (47.56)</td><td>31.50 (22.58)</td><td>26.39 (9.78)</td><td>3.24 (0.36)</td><td>18.64 (12.10)</td><td>13.37 (7.02)</td><td></td></tr><tr><td rowspan="6">ar</td><td rowspan="2">None</td><td>Adapter</td><td>29.14 (21.80)</td><td>55.90 (46.30)</td><td>0.22 (18.94)</td><td>55.90 (46.30)</td><td>27.83 (19.74)</td><td>23.15 (8.44)</td><td>2.39 (0.36)</td><td>15.70 (11.00)</td><td>13.00 (10.38)</td><td>24.80 (20.36)</td></tr><tr><td>Soft Prompt</td><td>22.93 (18.02)</td><td>56.08 (46.89)</td><td>0.10 (0.82)</td><td>56.08 (46.89)</td><td>23.91 (17.33)</td><td>19.89 (7.85)</td><td>0.56 (0.18)</td><td>11.53 (8.60)</td><td>12.05 (8.29)</td><td>22.57 (17.21)</td></tr><tr><td rowspan="2">Adapter</td><td>Adapter</td><td>25.60 (14.45)</td><td>45.96 (35.71)</td><td>1.12 (4.15)</td><td>43.17 (31.85)</td><td>23.55 (11.82)</td><td>22.84 (6.03)</td><td>1.77 (0.18)</td><td>9.54 (4.90)</td><td>9.04 (4.08)</td><td>20.29 (12.57)</td></tr><tr><td>Soft Prompt</td><td>24.70 (1.67)</td><td>52.76 (42.94)</td><td>0.95 (0.63)</td><td>46.35 (35.38)</td><td>27.65 (16.04)</td><td>25.10 (1.28)</td><td>2.57 (1.09)</td><td>10.50 (5.10)</td><td>10.43 (5.84)</td><td>22.33 (12.22)</td></tr><tr><td rowspan="2">Soft Prompt</td><td>Adapter</td><td>27.95 (20.78)</td><td>56.45 (46.97)</td><td>0.10 (21.27)</td><td>55.81 (46.47)</td><td>28.17 (19.88)</td><td>23.60 (8.72)</td><td>2.43 (0.45)</td><td>14.21 (10.10)</td><td>12.74 (10.07)</td><td>24.61 (20.52)</td></tr><tr><td>Soft Prompt</td><td>27.14 (20.55)</td><td>55.51 (46.72)</td><td>0.03 (0.31)</td><td>56.11 (47.23)</td><td>26.00 (18.62)</td><td>20.96 (8.04)</td><td>0.77 (0.27)</td><td>11.04 (8.30)</td><td>12.37 (8.63)</td><td>23.33 (17.63)</td></tr><tr><td rowspan="6">de</td><td rowspan="2">None</td><td>Adapter</td><td>35.37 (27.29)</td><td>58.88 (49.08)</td><td>0.99 (2.27)</td><td>58.88 (49.08)</td><td>36.15 (24.66)</td><td>27.51 (10.16)</td><td>3.84 (0.82)</td><td>18.81 (12.80)</td><td>14.02 (10.27)</td><td>28.27 (20.71)</td></tr><tr><td>Soft Prompt</td><td>28.56 (21.31)</td><td>57.12 (47.65)</td><td>0.57 (2.20)</td><td>57.12 (47.65)</td><td>30.34 (20.52)</td><td>24.54 (8.92)</td><td>1.93 (0.73)</td><td>12.46 (9.70)</td><td>10.54 (8.21)</td><td>24.80 (18.54)</td></tr><tr><td rowspan="2">Adapter</td><td>Adapter</td><td>36.78 (27.82)</td><td>57.46 (47.31)</td><td>1.05 (1.38)</td><td>57.69 (47.90)</td><td>38.03 (25.07)</td><td>31.43 (11.35)</td><td>3.01 (0.54)</td><td>11.32 (6.00)</td><td>9.62 (4.72)</td><td>27.37 (19.12)</td></tr><tr><td>Soft Prompt</td><td>38.13 (31.20)</td><td>54.70 (45.29)</td><td>1.88 (0.94)</td><td>55.11 (46.55)</td><td>38.88 (27.69)</td><td>30.70 (11.98)</td><td>3.13 (1.00)</td><td>16.05 (8.70)</td><td>12.63 (8.01)</td><td>27.91 (20.15)</td></tr><tr><td rowspan="2">Soft Prompt</td><td>Adapter</td><td>31.81 (24.29)</td><td>59.67 (48.91)</td><td>0.80 (2.08)</td><td>59.85 (49.08)</td><td>35.74 (23.84)</td><td>27.42 (10.16)</td><td>3.45 (0.36)</td><td>17.13 (11.20)</td><td>13.41 (10.25)</td><td>27.70 (20.02)</td></tr><tr><td>Soft Prompt</td><td>32.68 (24.65)</td><td>57.86 (48.15)</td><td>0.58 (8.68)</td><td>58.38 (49.08)</td><td>30.67 (20.78)</td><td>27.95 (10.30)</td><td>2.25 (1.09)</td><td>12.34 (9.50)</td><td>9.35 (9.58)</td><td>25.78 (20.20)</td></tr><tr><td rowspan="6">en</td><td rowspan="2">None</td><td>Adapter</td><td>36.95 (28.57)</td><td>60.24 (50.34)</td><td>1.18 (0.94)</td><td>60.24 (50.34)</td><td>37.04 (26.07)</td><td>30.11 (11.46)</td><td>3.21 (0.36)</td><td>19.65 (12.70)</td><td>13.93 (9.37)</td><td>29.17 (21.13)</td></tr><tr><td>Soft Prompt</td><td>33.59 (25.55)</td><td>60.35 (50.76)</td><td>0.82 (0.94)</td><td>60.35 (50.76)</td><td>34.77 (24.36)</td><td>27.76 (10.07)</td><td>2.81 (0.45)</td><td>19.38 (12.80)</td><td>13.81 (9.25)</td><td>28.18 (20.55)</td></tr><tr><td rowspan="2">Adapter</td><td>Adapter</td><td>33.75 (24.33)</td><td>57.44 (48.24)</td><td>1.25 (1.70)</td><td>58.11 (49.33)</td><td>31.65 (19.76)</td><td>28.98 (10.58)</td><td>3.25 (0.63)</td><td>9.85 (5.00)</td><td>9.06 (4.20)</td><td>25.93 (18.20)</td></tr><tr><td>Soft Prompt</td><td>33.94 (26.22)</td><td>58.49 (49.58)</td><td>0.97 (0.25)</td><td>57.72 (48.74)</td><td>35.89 (24.36)</td><td>29.82 (11.54)</td><td>3.06 (0.45)</td><td>12.67 (6.60)</td><td>9.67 (4.94)</td><td>26.91 (19.19)</td></tr><tr><td rowspan="2">Soft Prompt</td><td>Adapter</td><td>35.52 (27.27)</td><td>61.17 (51.68)</td><td>1.39 (2.71)</td><td>61.99 (52.10)</td><td>37.28 (25.96)</td><td>30.84 (11.73)</td><td>3.57 (0.27)</td><td>20.19 (13.50)</td><td>14.06 (10.54)</td><td>29.56 (21.75)</td></tr><tr><td>Soft Prompt</td><td>35.35 (27.02)</td><td>59.80 (50.34)</td><td>0.68 (2.20)</td><td>60.02 (50.34)</td><td>36.49 (26.02)</td><td>29.89 (11.39)</td><td>3.46 (0.54)</td><td>11.71 (7.30)</td><td>9.83 (5.68)</td><td>27.47 (20.09)</td></tr><tr><td rowspan="6">es</td><td rowspan="2">None</td><td>Adapter</td><td>33.72 (25.43)</td><td>59.61 (50.34)</td><td>1.29 (1.20)</td><td>59.61 (50.34)</td><td>34.97 (32.80)</td><td>27.24 (9.71)</td><td>3.42 (0.82)</td><td>19.10 (12.50)</td><td>13.03 (9.48)</td><td>28.00 (20.40)</td></tr><tr><td>Soft Prompt</td><td>25.98 (19.06)</td><td>54.72 (45.29)</td><td>0.12 (0.31)</td><td>54.72 (45.29)</td><td>27.38 (18.50)</td><td>22.43 (7.62)</td><td>1.10 (0.36)</td><td>11.94 (9.00)</td><td>9.13 (6.84)</td><td>23.06 (16.92)</td></tr><tr><td rowspan="2">Adapter</td><td>Adapter</td><td>33.98 (23.88)</td><td>55.44 (45.13)</td><td>1.06 (1.26)</td><td>56.58 (46.22)</td><td>34.52 (21.06)</td><td>28.66 (10.01)</td><td>2.34 (0.45)</td><td>10.76 (5.00)</td><td>9.12 (4.81)</td><td>25.83 (17.54)</td></tr><tr><td>Soft Prompt</td><td>32.86 (24.45)</td><td>53.57 (43.78)</td><td>0.98 (0.57)</td><td>52.82 (43.45)</td><td>34.73 (23.35)</td><td>27.82 (10.35)</td><td>2.49 (0.54)</td><td>10.89 (5.40)</td><td>9.27 (4.17)</td><td>25.05 (17.34)</td></tr><tr><td rowspan="2">Soft Prompt</td><td>Adapter</td><td>32.43 (24.92)</td><td>59.41 (49.75)</td><td>0.74 (1.07)</td><td>59.05 (49.24)</td><td>36.02 (24.92)</td><td>27.95 (10.27)</td><td>3.22 (0.45)</td><td>17.64 (11.40)</td><td>12.41 (8.42)</td><td>27.65 (20.05)</td></tr><tr><td>Soft Prompt</td><td>27.21 (19.82)</td><td>54.39 (44.45)</td><td>0.27 (0.82)</td><td>54.93 (45.21)</td><td>28.25 (19.39)</td><td>21.49 (7.49)</td><td>1.11 (0.18)</td><td>9.28 (7.20)</td><td>8.36 (6.11)</td><td>22.81 (16.74)</td></tr><tr><td rowspan="6">ru</td><td rowspan="2">None</td><td>Adapter</td><td>27.45 (15.86)</td><td>55.56 (42.52)</td><td>0.73 (5.73)</td><td>55.56 (42.52)</td><td>24.81 (14.05)</td><td>23.69 (7.96)</td><td>2.93 (0.45)</td><td>16.80 (10.50)</td><td>12.45 (10.12)</td><td>24.44 (16.63)</td></tr><tr><td>Soft Prompt</td><td>17.94 (8.27)</td><td>51.61 (37.73)</td><td>0.25 (0.13)</td><td>51.61 (37.73)</td><td>17.58 (8.13)</td><td>16.98 (4.10)</td><td>0.78 (0.27)</td><td>12.40 (8.00)</td><td>8.71 (5.49)</td><td>19.76 (12.21)</td></tr><tr><td rowspan="2">Adapter</td><td>Adapter</td><td>31.24 (14.98)</td><td>54.17 (40.25)</td><td>1.27 (0.76)</td><td>54.72 (40.50)</td><td>32.90 (17.04)</td><td>32.13 (9.71)</td><td>2.72 (0.18)</td><td>10.66 (5.30)</td><td>8.98 (3.85)</td><td>25.42 (14.73)</td></tr><tr><td>Soft Prompt</td><td>32.53 (19.94)</td><td>51.63 (38.91)</td><td>0.89 (0.25)</td><td>51.78 (37.31)</td><td>33.57 (18.08)</td><td>34.00 (13.34)</td><td>1.96 (0.09)</td><td>10.37 (4.80)</td><td>8.78 (3.91)</td><td>25.06 (15.18)</td></tr><tr><td rowspan="2">Soft Prompt</td><td>Adapter</td><td>22.06 (12.33)</td><td>55.24 (42.44)</td><td>0.91 (1.32)</td><td>54.38 (41.60)</td><td>24.61 (14.26)</td><td>22.75 (7.53)</td><td>2.63 (0.27)</td><td>15.54 (9.50)</td><td>12.22 (7.80)</td><td>23.37 (15.23)</td></tr><tr><td>Soft Prompt</td><td>32.13 (18.00)</td><td>53.90 (40.00)</td><td>0.99 (0.25)</td><td>53.37 (39.83)</td><td>29.68 (14.91)</td><td>31.42 (10.26)</td><td>2.06 (0.18)</td><td>16.86 (9.20)</td><td>11.63 (6.04)</td><td>25.78 (15.41)</td></tr><tr><td rowspan="6">zh</td><td rowspan="2">None</td><td>Adapter</td><td>22.06 (16.33)</td><td>50.83 (40.08)</td><td>0.65 (0.50)</td><td>50.83 (40.08)</td><td>21.96 (14.96)</td><td>18.38 (6.74)</td><td>1.42 (0.27)</td><td>13.29 (9.20)</td><td>9.66 (6.31)</td><td>21.01 (14.94)</td></tr><tr><td>Soft Prompt</td><td>26.25 (20.33)</td><td>56.57 (47.39)</td><td>0.65 (0.44)</td><td>56.57 (47.39)</td><td>25.64 (18.29)</td><td>22.32 (8.53)</td><td>0.91 (0.45)</td><td>16.00 (11.80)</td><td>12.49 (9.17)</td><td>24.16 (18.20)</td></tr><tr><td rowspan="2">Adapter</td><td>Adapter</td><td>26.10 (16.29)</td><td>43.57 (30.34)</td><td>1.30 (0.57)</td><td>41.48 (28.57)</td><td>26.16 (14.21)</td><td>22.50 (6.17)</td><td>1.94 (0.36)</td><td>11.77 (6.50)</td><td>9.04 (5.01)</td><td>20.43 (12.00)</td></tr><tr><td>Soft Prompt</td><td>24.83 (14.06)</td><td>47.06 (35.80)</td><td>1.08 (0.06)</td><td>39.35 (25.55)</td><td>25.58 (12.45)</td><td>24.19 (5.44)</td><td>2.64 (0.82)</td><td>10.34 (4.90)</td><td>8.90 (4.24)</td><td>20.44 (11.48)</td></tr><tr><td rowspan="2">Soft Prompt</td><td>Adapter</td><td>22.21 (16.90)</td><td>52.01 (41.01)</td><td>0.53 (0.44)</td><td>51.45 (40.25)</td><td>23.85 (16.20)</td><td>18.82 (6.95)</td><td>2.11 (0.45)</td><td>13.91 (9.10)</td><td>9.59 (6.26)</td><td>21.61 (15.28)</td></tr><tr><td>Soft Prompt</td><td>31.34 (24.20)</td><td>56.51 (47.90)</td><td>1.23 (0.82)</td><td>56.39 (47.98)</td><td>29.25 (21.17)</td><td>25.06 (9.42)</td><td>1.85 (0.45)</td><td>17.52 (12.20)</td><td>12.80 (9.48)</td><td>25.77 (19.29)</td></tr></table>


Table 4: Results for the question answering task for cross-lingual transfer from high-resource to mid- and lowresource languages. The results are reported as F1-Score (Exact Match). For each source-target language pair, the best-performing result is highlighted in bold, while the second-best scores are underlined. Additionally, language pairs with improved performance compared to inference-only (without incorporating any language or task representation) are marked in green, and those with decreased performance are marked in red.


<table><tr><td>Task Language</td><td>Language Representation</td><td>Task Representation</td><td>bg</td><td>cs</td><td>el</td><td>ml</td><td>ro</td><td>sl</td><td>sk</td><td>sw</td><td>te</td><td>ur</td><td>Average</td></tr><tr><td></td><td>None</td><td>None</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td rowspan="6">ar</td><td rowspan="2">None</td><td>Adapter</td><td>44.54</td><td>60.54</td><td>44.29</td><td>31.32</td><td>45.73</td><td>55.63</td><td>60.30</td><td>49.92</td><td>48.53</td><td>32.10</td><td>47.29</td></tr><tr><td>Soft Prompt</td><td>38.11</td><td>50.72</td><td>39.73</td><td>38.32</td><td>36.44</td><td>48.28</td><td>48.69</td><td>46.90</td><td>45.53</td><td>20.61</td><td>41.33</td></tr><tr><td rowspan="2">Adapter</td><td>Adapter</td><td>63.91</td><td>63.98</td><td>63.89</td><td>49.19</td><td>53.44</td><td>24.02</td><td>62.98</td><td>49.38</td><td>18.05</td><td>27.71</td><td>47.66</td></tr><tr><td>Soft Prompt</td><td>64.09</td><td>40.09</td><td>52.23</td><td>48.21</td><td>55.95</td><td>44.67</td><td>54.13</td><td>51.37</td><td>26.65</td><td>25.77</td><td>46.32</td></tr><tr><td rowspan="2">Soft Prompt</td><td>Adapter</td><td>53.66</td><td>62.72</td><td>52.68</td><td>41.95</td><td>54.82</td><td>60.70</td><td>64.62</td><td>55.43</td><td>51.43</td><td>43.36</td><td>54.14</td></tr><tr><td>Soft Prompt</td><td>48.84</td><td>40.30</td><td>46.02</td><td>42.93</td><td>33.25</td><td>37.20</td><td>44.48</td><td>30.41</td><td>46.01</td><td>25.18</td><td>39.46</td></tr><tr><td rowspan="6">de</td><td rowspan="2">None</td><td>Adapter</td><td>30.63</td><td>66.61</td><td>53.35</td><td>40.49</td><td>62.39</td><td>56.11</td><td>67.96</td><td>60.19</td><td>48.56</td><td>34.37</td><td>52.07</td></tr><tr><td>Soft Prompt</td><td>31.61</td><td>61.04</td><td>54.21</td><td>44.55</td><td>51.52</td><td>54.04</td><td>62.28</td><td>57.26</td><td>43.20</td><td>24.21</td><td>48.39</td></tr><tr><td rowspan="2">Adapter</td><td>Adapter</td><td>68.49</td><td>64.70</td><td>68.83</td><td>46.45</td><td>60.64</td><td>49.32</td><td>68.42</td><td>60.92</td><td>24.52</td><td>28.41</td><td>54.07</td></tr><tr><td>Soft Prompt</td><td>62.16</td><td>54.79</td><td>60.52</td><td>42.90</td><td>54.74</td><td>44.46</td><td>51.17</td><td>57.28</td><td>27.96</td><td>47.28</td><td>50.33</td></tr><tr><td rowspan="2">Soft Prompt</td><td>Adapter</td><td>39.84</td><td>67.12</td><td>55.29</td><td>40.43</td><td>61.59</td><td>63.28</td><td>71.04</td><td>56.53</td><td>49.70</td><td>45.00</td><td>54.98</td></tr><tr><td>Soft Prompt</td><td>43.30</td><td>62.84</td><td>61.20</td><td>44.44</td><td>52.13</td><td>57.21</td><td>70.80</td><td>54.24</td><td>48.67</td><td>34.37</td><td>52.92</td></tr><tr><td rowspan="6">en</td><td rowspan="2">None</td><td>Adapter</td><td>35.49</td><td>41.53</td><td>49.44</td><td>24.71</td><td>44.91</td><td>55.13</td><td>44.15</td><td>36.52</td><td>32.02</td><td>25.96</td><td>38.99</td></tr><tr><td>Soft Prompt</td><td>29.04</td><td>42.31</td><td>46.31</td><td>27.92</td><td>53.48</td><td>53.06</td><td>45.40</td><td>48.41</td><td>33.41</td><td>39.54</td><td>41.89</td></tr><tr><td rowspan="2">Adapter</td><td>Adapter</td><td>21.42</td><td>33.99</td><td>39.86</td><td>17.16</td><td>42.90</td><td>23.39</td><td>23.18</td><td>32.18</td><td>13.91</td><td>21.95</td><td>26.99</td></tr><tr><td>Soft Prompt</td><td>50.16</td><td>40.62</td><td>56.07</td><td>37.99</td><td>63.94</td><td>56.46</td><td>35.97</td><td>45.38</td><td>19.98</td><td>61.68</td><td>46.83</td></tr><tr><td rowspan="2">Soft Prompt</td><td>Adapter</td><td>44.12</td><td>40.65</td><td>47.99</td><td>24.97</td><td>52.36</td><td>51.04</td><td>51.62</td><td>48.55</td><td>31.23</td><td>23.32</td><td>41.59</td></tr><tr><td>Soft Prompt</td><td>22.90</td><td>44.33</td><td>41.92</td><td>25.22</td><td>49.67</td><td>55.77</td><td>48.64</td><td>48.77</td><td>33.85</td><td>9.87</td><td>38.09</td></tr><tr><td rowspan="6">es</td><td rowspan="2">None</td><td>Adapter</td><td>35.41</td><td>68.66</td><td>49.50</td><td>35.38</td><td>66.54</td><td>66.07</td><td>70.46</td><td>64.65</td><td>46.24</td><td>46.35</td><td>54.93</td></tr><tr><td>Soft Prompt</td><td>24.70</td><td>63.95</td><td>48.38</td><td>31.45</td><td>62.20</td><td>60.00</td><td>63.25</td><td>62.21</td><td>47.81</td><td>45.11</td><td>50.91</td></tr><tr><td rowspan="2">Adapter</td><td>Adapter</td><td>63.38</td><td>58.05</td><td>67.33</td><td>42.61</td><td>59.17</td><td>44.54</td><td>66.00</td><td>56.36</td><td>17.84</td><td>30.86</td><td>50.61</td></tr><tr><td>Soft Prompt</td><td>69.95</td><td>52.71</td><td>64.27</td><td>42.75</td><td>59.96</td><td>52.05</td><td>60.82</td><td>58.11</td><td>31.27</td><td>48.71</td><td>54.06</td></tr><tr><td rowspan="2">Soft Prompt</td><td>Adapter</td><td>54.37</td><td>67.45</td><td>64.81</td><td>40.37</td><td>64.62</td><td>70.83</td><td>68.78</td><td>66.53</td><td>50.91</td><td>53.78</td><td>60.25</td></tr><tr><td>Soft Prompt</td><td>44.72</td><td>65.55</td><td>39.99</td><td>39.17</td><td>63.38</td><td>63.52</td><td>66.04</td><td>61.86</td><td>40.59</td><td>47.45</td><td>53.23</td></tr><tr><td rowspan="6">ru</td><td rowspan="2">None</td><td>Adapter</td><td>72.90</td><td>46.64</td><td>34.14</td><td>25.54</td><td>49.66</td><td>45.84</td><td>48.16</td><td>50.02</td><td>49.73</td><td>37.42</td><td>46.01</td></tr><tr><td>Soft Prompt</td><td>69.54</td><td>53.60</td><td>25.83</td><td>34.23</td><td>51.11</td><td>51.04</td><td>51.10</td><td>48.37</td><td>49.36</td><td>40.56</td><td>47.47</td></tr><tr><td rowspan="2">Adapter</td><td>Adapter</td><td>71.70</td><td>54.14</td><td>72.40</td><td>38.30</td><td>59.91</td><td>30.15</td><td>50.78</td><td>48.08</td><td>18.41</td><td>25.92</td><td>46.98</td></tr><tr><td>Soft Prompt</td><td>73.07</td><td>53.33</td><td>65.15</td><td>37.04</td><td>53.23</td><td>58.78</td><td>52.96</td><td>50.20</td><td>21.37</td><td>27.01</td><td>49.21</td></tr><tr><td rowspan="2">Soft Prompt</td><td>Adapter</td><td>77.27</td><td>54.48</td><td>54.56</td><td>42.01</td><td>60.01</td><td>53.82</td><td>56.33</td><td>56.61</td><td>55.41</td><td>46.92</td><td>55.74</td></tr><tr><td>Soft Prompt</td><td>71.73</td><td>59.97</td><td>65.74</td><td>44.44</td><td>58.04</td><td>60.23</td><td>56.40</td><td>54.53</td><td>47.95</td><td>46.62</td><td>56.57</td></tr><tr><td rowspan="6">zh</td><td rowspan="2">None</td><td>Adapter</td><td>14.91</td><td>43.07</td><td>8.42</td><td>1.21</td><td>39.80</td><td>41.54</td><td>42.05</td><td>40.50</td><td>11.46</td><td>11.33</td><td>25.43</td></tr><tr><td>Soft Prompt</td><td>12.35</td><td>51.86</td><td>17.37</td><td>7.50</td><td>45.41</td><td>53.70</td><td>49.48</td><td>46.97</td><td>28.33</td><td>17.65</td><td>33.06</td></tr><tr><td rowspan="2">Adapter</td><td>Adapter</td><td>29.40</td><td>41.60</td><td>39.63</td><td>7.39</td><td>37.02</td><td>19.26</td><td>28.99</td><td>21.72</td><td>5.56</td><td>6.54</td><td>23.71</td></tr><tr><td>Soft Prompt</td><td>37.86</td><td>46.02</td><td>19.56</td><td>43.69</td><td>30.45</td><td>34.32</td><td>18.60</td><td>55.69</td><td>38.07</td><td>33.44</td><td>35.77</td></tr><tr><td rowspan="2">Soft Prompt</td><td>Adapter</td><td>28.41</td><td>46.21</td><td>24.62</td><td>8.49</td><td>50.18</td><td>44.69</td><td>47.55</td><td>51.13</td><td>28.76</td><td>15.06</td><td>34.51</td></tr><tr><td>Soft Prompt</td><td>23.34</td><td>55.43</td><td>32.02</td><td>12.41</td><td>43.96</td><td>53.55</td><td>55.26</td><td>48.51</td><td>32.11</td><td>18.78</td><td>37.54</td></tr><tr><td></td><td>None</td><td>None</td><td>43.35</td><td>35.50</td><td>40.88</td><td>40.62</td><td>4.98</td><td>68.74</td><td>36.42</td><td>38.90</td><td>39.58</td><td>37.64</td><td>38.66</td></tr><tr><td rowspan="6">ar</td><td rowspan="2">None</td><td>Adapter</td><td>74.77</td><td>35.50</td><td>74.05</td><td>68.98</td><td>66.06</td><td>28.76</td><td>37.08</td><td>64.55</td><td>66.67</td><td>66.01</td><td>58.24</td></tr><tr><td>Soft Prompt</td><td>69.42</td><td>35.92</td><td>69.94</td><td>65.47</td><td>77.59</td><td>25.45</td><td>35.83</td><td>59.94</td><td>64.23</td><td>60.68</td><td>56.45</td></tr><tr><td rowspan="2">Adapter</td><td>Adapter</td><td>72.16</td><td>35.67</td><td>74.17</td><td>65.59</td><td>77.15</td><td>2.91</td><td>37.17</td><td>62.18</td><td>55.93</td><td>47.17</td><td>53.01</td></tr><tr><td>Soft Prompt</td><td>56.39</td><td>38.08</td><td>62.69</td><td>57.98</td><td>42.24</td><td>67.54</td><td>37.17</td><td>58.16</td><td>46.19</td><td>57.19</td><td>52.36</td></tr><tr><td rowspan="2">Soft Prompt</td><td>Adapter</td><td>74.21</td><td>34.42</td><td>72.46</td><td>68.70</td><td>63.62</td><td>32.77</td><td>36.17</td><td>62.61</td><td>66.05</td><td>64.05</td><td>57.51</td></tr><tr><td>Soft Prompt</td><td>65.45</td><td>37.50</td><td>67.84</td><td>62.53</td><td>59.03</td><td>30.96</td><td>37.25</td><td>61.02</td><td>61.44</td><td>59.60</td><td>54.26</td></tr><tr><td rowspan="6">de</td><td rowspan="2">None</td><td>Adapter</td><td>75.03</td><td>35.17</td><td>74.51</td><td>69.92</td><td>64.61</td><td>30.16</td><td>35.92</td><td>65.83</td><td>68.90</td><td>65.73</td><td>58.58</td></tr><tr><td>Soft Prompt</td><td>69.84</td><td>35.50</td><td>70.44</td><td>65.67</td><td>79.10</td><td>31.36</td><td>37.00</td><td>62.75</td><td>64.45</td><td>62.79</td><td>57.89</td></tr><tr><td rowspan="2">Adapter</td><td>Adapter</td><td>74.41</td><td>34.92</td><td>72.34</td><td>68.74</td><td>49.85</td><td>16.13</td><td>35.50</td><td>62.85</td><td>66.81</td><td>60.18</td><td>54.17</td></tr><tr><td>Soft Prompt</td><td>62.51</td><td>37.17</td><td>62.81</td><td>55.29</td><td>35.16</td><td>46.19</td><td>36.67</td><td>56.67</td><td>46.45</td><td>53.13</td><td>49.21</td></tr><tr><td rowspan="2">Soft Prompt</td><td>Adapter</td><td>75.23</td><td>33.75</td><td>71.50</td><td>70.06</td><td>60.25</td><td>35.17</td><td>34.50</td><td>65.81</td><td>67.13</td><td>64.25</td><td>57.77</td></tr><tr><td>Soft Prompt</td><td>69.12</td><td>36.67</td><td>69.06</td><td>64.61</td><td>70.75</td><td>26.35</td><td>37.67</td><td>61.18</td><td>64.37</td><td>60.00</td><td>55.98</td></tr><tr><td rowspan="6">en</td><td rowspan="2">None</td><td>Adapter</td><td>75.03</td><td>35.00</td><td>74.33</td><td>70.30</td><td>69.48</td><td>28.66</td><td>34.58</td><td>67.07</td><td>69.38</td><td>64.63</td><td>58.85</td></tr><tr><td>Soft Prompt</td><td>70.42</td><td>33.83</td><td>70.32</td><td>64.53</td><td>77.00</td><td>29.96</td><td>34.58</td><td>63.31</td><td>63.43</td><td>60.22</td><td>56.76</td></tr><tr><td rowspan="2">Adapter</td><td>Adapter</td><td>64.07</td><td>35.33</td><td>64.79</td><td>56.51</td><td>90.09</td><td>1.00</td><td>35.08</td><td>61.30</td><td>44.11</td><td>39.78</td><td>49.21</td></tr><tr><td>Soft Prompt</td><td>55.19</td><td>38.00</td><td>51.30</td><td>46.53</td><td>81.35</td><td>33.77</td><td>38.75</td><td>56.97</td><td>33.33</td><td>52.81</td><td>48.80</td></tr><tr><td rowspan="2">Soft Prompt</td><td>Adapter</td><td>75.11</td><td>33.83</td><td>72.50</td><td>69.50</td><td>54.05</td><td>35.07</td><td>35.58</td><td>64.67</td><td>67.88</td><td>64.07</td><td>57.23</td></tr><tr><td>Soft Prompt</td><td>65.37</td><td>35.58</td><td>69.50</td><td>64.51</td><td>35.64</td><td>34.07</td><td>37.08</td><td>60.94</td><td>63.35</td><td>59.62</td><td>52.57</td></tr><tr><td rowspan="6">es</td><td rowspan="2">None</td><td>Adapter</td><td>74.87</td><td>35.08</td><td>75.29</td><td>70.40</td><td>72.12</td><td>28.06</td><td>35.17</td><td>66.17</td><td>70.00</td><td>66.09</td><td>59.33</td></tr><tr><td>Soft Prompt</td><td>68.98</td><td>34.83</td><td>70.12</td><td>65.55</td><td>79.83</td><td>31.76</td><td>35.75</td><td>60.66</td><td>64.11</td><td>61.36</td><td>57.30</td></tr><tr><td rowspan="2">Adapter</td><td>Adapter</td><td>73.95</td><td>34.92</td><td>74.61</td><td>68.26</td><td>71.78</td><td>10.02</td><td>35.83</td><td>64.77</td><td>59.40</td><td>50.42</td><td>54.40</td></tr><tr><td>Soft Prompt</td><td>54.97</td><td>38.75</td><td>44.47</td><td>41.48</td><td>81.98</td><td>8.42</td><td>36.42</td><td>55.11</td><td>33.49</td><td>46.13</td><td>44.12</td></tr><tr><td rowspan="2">Soft Prompt</td><td>Adapter</td><td>74.27</td><td>33.92</td><td>72.38</td><td>69.68</td><td>67.58</td><td>32.67</td><td>36.25</td><td>64.35</td><td>67.33</td><td>62.95</td><td>58.14</td></tr><tr><td>Soft Prompt</td><td>66.15</td><td>35.92</td><td>67.54</td><td>62.77</td><td>78.52</td><td>28.76</td><td>36.33</td><td>59.66</td><td>63.19</td><td>57.66</td><td>55.65</td></tr><tr><td rowspan="6">ru</td><td rowspan="2">None</td><td>Adapter</td><td>75.55</td><td>34.92</td><td>74.81</td><td>70.56</td><td>75.24</td><td>29.96</td><td>34.67</td><td>67.54</td><td>69.54</td><td>66.29</td><td>59.91</td></tr><tr><td>Soft Prompt</td><td>69.90</td><td>36.25</td><td>69.78</td><td>64.85</td><td>82.62</td><td>28.16</td><td>35.67</td><td>61.60</td><td>63.73</td><td>62.36</td><td>57.49</td></tr><tr><td rowspan="2">Adapter</td><td>Adapter</td><td>72.63</td><td>35.75</td><td>74.19</td><td>61.14</td><td>83.50</td><td>4.91</td><td>36.25</td><td>65.77</td><td>51.16</td><td>46.19</td><td>53.15</td></tr><tr><td>Soft Prompt</td><td>63.45</td><td>37.58</td><td>53.43</td><td>44.51</td><td>87.01</td><td>11.32</td><td>38.08</td><td>54.67</td><td>37.60</td><td>53.13</td><td>48.08</td></tr><tr><td rowspan="2">Soft Prompt</td><td>Adapter</td><td>76.03</td><td>34.92</td><td>73.75</td><td>68.18</td><td>75.68</td><td>28.06</td><td>35.83</td><td>65.77</td><td>68.24</td><td>63.99</td><td>59.05</td></tr><tr><td>Soft Prompt</td><td>68.42</td><td>37.25</td><td>67.72</td><td>63.53</td><td>75.05</td><td>23.65</td><td>37.25</td><td>57.72</td><td>62.55</td><td>60.08</td><td>55.32</td></tr><tr><td rowspan="6">zh</td><td rowspan="2">None</td><td>Adapter</td><td>74.95</td><td>35.75</td><td>73.79</td><td>69.16</td><td>74.32</td><td>28.66</td><td>36.33</td><td>68.00</td><td>69.38</td><td>66.11</td><td>59.65</td></tr><tr><td>Soft Prompt</td><td>70.80</td><td>35.75</td><td>70.44</td><td>66.59</td><td>78.52</td><td>21.84</td><td>35.17</td><td>62.36</td><td>65.75</td><td>62.85</td><td>57.01</td></tr><tr><td rowspan="2">Adapter</td><td>Adapter</td><td>61.72</td><td>34.08</td><td>64.27</td><td>50.90</td><td>89.01</td><td>0.20</td><td>35.17</td><td>60.66</td><td>35.71</td><td>38.98</td><td>47.07</td></tr><tr><td>Soft Prompt</td><td>43.47</td><td>33.33</td><td>56.09</td><td>56.29</td><td>45.90</td><td>98.20</td><td>35.00</td><td>47.07</td><td>34.57</td><td>47.76</td><td>49.77</td></tr><tr><td rowspan="2">Soft Prompt</td><td>Adapter</td><td>73.43</td><td>35.42</td><td>72.16</td><td>68.06</td><td>65.77</td><td>31.06</td><td>35.50</td><td>64.37</td><td>66.91</td><td>63.67</td><td>57.64</td></tr><tr><td>Soft PromptNone</td><td>68.680</td><td>34.670</td><td>68.860</td><td>64.690</td><td>76.950</td><td>23.550</td><td>36.500</td><td>60.420</td><td>63.390</td><td>61.780</td><td>55.950</td></tr><tr><td></td><td>None</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="6">ar</td><td rowspan="2">None</td><td>Adapter</td><td>84.78</td><td>70.74</td><td>86.77</td><td>86.45</td><td>86.06</td><td>82.74</td><td>72.52</td><td>86.59</td><td>84.32</td><td>81.50</td><td>82.25</td></tr><tr><td>Soft Prompt</td><td>86.48</td><td>63.08</td><td>84.90</td><td>86.32</td><td>87.61</td><td>80.77</td><td>56.91</td><td>81.47</td><td>84.47</td><td>81.92</td><td>79.39</td></tr><tr><td rowspan="2">Adapter</td><td>Adapter</td><td>58.38</td><td>45.01</td><td>74.35</td><td>83.39</td><td>73.17</td><td>71.29</td><td>36.69</td><td>82.63</td><td>77.91</td><td>80.18</td><td>68.30</td></tr><tr><td>Soft Prompt</td><td>89.22</td><td>56.49</td><td>85.76</td><td>86.82</td><td>78.56</td><td>82.24</td><td>48.71</td><td>85.96</td><td>82.44</td><td>81.62</td><td>77.78</td></tr><tr><td rowspan="2">Soft Prompt</td><td>Adapter</td><td>85.83</td><td>67.14</td><td>87.29</td><td>82.13</td><td>88.41</td><td>85.08</td><td>68.66</td><td>84.36</td><td>82.47</td><td>78.36</td><td>80.97</td></tr><tr><td>Soft Prompt</td><td>85.89</td><td>61.41</td><td>81.91</td><td>85.13</td><td>86.84</td><td>84.60</td><td>57.40</td><td>84.20</td><td>81.95</td><td>81.71</td><td>79.10</td></tr><tr><td rowspan="6">de</td><td rowspan="2">None</td><td>Adapter</td><td>97.63</td><td>69.72</td><td>97.74</td><td>94.66</td><td>98.11</td><td>97.11</td><td>73.70</td><td>97.25</td><td>93.73</td><td>93.17</td><td>91.28</td></tr><tr><td>Soft Prompt</td><td>94.49</td><td>65.32</td><td>96.42</td><td>96.28</td><td>92.33</td><td>88.16</td><td>62.31</td><td>90.69</td><td>91.54</td><td>89.10</td><td>86.66</td></tr><tr><td rowspan="2">Adapter</td><td>Adapter</td><td>97.03</td><td>78.84</td><td>95.30</td><td>85.30</td><td>97.19</td><td>95.28</td><td>82.91</td><td>90.39</td><td>79.96</td><td>85.29</td><td>88.75</td></tr><tr><td>Soft Prompt</td><td>93.15</td><td>58.68</td><td>94.46</td><td>95.17</td><td>95.57</td><td>96.11</td><td>67.99</td><td>87.70</td><td>93.06</td><td>65.98</td><td>84.79</td></tr><tr><td rowspan="2">Soft Prompt</td><td>Adapter</td><td>97.06</td><td>74.94</td><td>96.84</td><td>92.17</td><td>97.76</td><td>96.85</td><td>82.96</td><td>95.40</td><td>89.64</td><td>86.71</td><td>91.03</td></tr><tr><td>Soft Prompt</td><td>94.51</td><td>63.87</td><td>91.18</td><td>94.29</td><td>95.44</td><td>94.49</td><td>57.84</td><td>86.67</td><td>93.02</td><td>76.01</td><td>84.73</td></tr><tr><td rowspan="6">en</td><td rowspan="2">None</td><td>Adapter</td><td>95.79</td><td>85.10</td><td>97.46</td><td>89.15</td><td>94.52</td><td>89.31</td><td>83.09</td><td>91.98</td><td>87.46</td><td>88.65</td><td>90.25</td></tr><tr><td>Soft Prompt</td><td>94.98</td><td>55.29</td><td>95.09</td><td>93.12</td><td>91.50</td><td>91.47</td><td>53.03</td><td>91.17</td><td>93.50</td><td>88.35</td><td>84.75</td></tr><tr><td rowspan="2">Adapter</td><td>Adapter</td><td>97.22</td><td>82.95</td><td>98.04</td><td>92.28</td><td>96.98</td><td>95.80</td><td>85.98</td><td>91.20</td><td>85.99</td><td>89.57</td><td>91.60</td></tr><tr><td>Soft Prompt</td><td>88.69</td><td>40.60</td><td>94.31</td><td>86.24</td><td>94.97</td><td>93.09</td><td>46.60</td><td>88.02</td><td>81.91</td><td>86.00</td><td>80.04</td></tr><tr><td rowspan="2">Soft Prompt</td><td>Adapter</td><td>87.61</td><td>83.47</td><td>95.89</td><td>77.89</td><td>88.38</td><td>81.98</td><td>85.49</td><td>87.42</td><td>77.34</td><td>81.67</td><td>84.71</td></tr><tr><td>Soft Prompt</td><td>80.71</td><td>75.72</td><td>92.71</td><td>82.49</td><td>90.44</td><td>84.21</td><td>70.08</td><td>82.65</td><td>79.55</td><td>75.16</td><td>81.37</td></tr><tr><td rowspan="6">es</td><td rowspan="2">None</td><td>Adapter</td><td>95.66</td><td>76.98</td><td>95.85</td><td>91.93</td><td>97.06</td><td>95.81</td><td>74.40</td><td>95.62</td><td>93.15</td><td>90.57</td><td>90.70</td></tr><tr><td>Soft Prompt</td><td>92.80</td><td>76.38</td><td>94.59</td><td>91.93</td><td>90.62</td><td>82.23</td><td>74.78</td><td>92.40</td><td>91.90</td><td>83.25</td><td>87.09</td></tr><tr><td rowspan="2">Adapter</td><td>Adapter</td><td>84.35</td><td>73.15</td><td>95.94</td><td>92.67</td><td>96.16</td><td>94.07</td><td>81.65</td><td>93.48</td><td>93.80</td><td>88.13</td><td>89.34</td></tr><tr><td>Soft Prompt</td><td>92.98</td><td>77.88</td><td>96.15</td><td>93.27</td><td>96.03</td><td>93.07</td><td>81.02</td><td>92.09</td><td>83.19</td><td>71.66</td><td>87.73</td></tr><tr><td rowspan="2">Soft Prompt</td><td>Adapter</td><td>97.47</td><td>87.14</td><td>97.41</td><td>94.78</td><td>96.89</td><td>96.01</td><td>86.36</td><td>95.55</td><td>91.72</td><td>86.85</td><td>93.02</td></tr><tr><td>Soft Prompt</td><td>96.35</td><td>79.97</td><td>95.79</td><td>90.56</td><td>95.08</td><td>92.59</td><td>78.85</td><td>92.85</td><td>89.11</td><td>87.56</td><td>89.87</td></tr><tr><td rowspan="6">ru</td><td rowspan="2">None</td><td>Adapter</td><td>98.87</td><td>72.15</td><td>98.46</td><td>90.34</td><td>98.26</td><td>96.35</td><td>74.45</td><td>97.10</td><td>89.87</td><td>92.93</td><td>90.88</td></tr><tr><td>Soft Prompt</td><td>98.33</td><td>58.26</td><td>96.09</td><td>95.24</td><td>87.02</td><td>84.39</td><td>59.06</td><td>92.15</td><td>95.03</td><td>92.11</td><td>85.77</td></tr><tr><td rowspan="2">Adapter</td><td>Adapter</td><td>97.90</td><td>88.93</td><td>94.70</td><td>82.33</td><td>96.89</td><td>92.79</td><td>88.80</td><td>91.75</td><td>77.95</td><td>87.34</td><td>89.94</td></tr><tr><td>Soft Prompt</td><td>97.36</td><td>13.87</td><td>96.77</td><td>95.50</td><td>95.03</td><td>95.70</td><td>20.58</td><td>95.59</td><td>91.23</td><td>92.34</td><td>79.40</td></tr><tr><td rowspan="2">Soft Prompt</td><td>Adapter</td><td>98.59</td><td>67.29</td><td>97.46</td><td>88.39</td><td>98.53</td><td>97.73</td><td>74.44</td><td>95.83</td><td>88.39</td><td>91.36</td><td>89.80</td></tr><tr><td>Soft Prompt</td><td>98.89</td><td>48.65</td><td>96.71</td><td>93.57</td><td>97.32</td><td>96.85</td><td>38.89</td><td>95.17</td><td>92.26</td><td>89.83</td><td>84.81</td></tr><tr><td rowspan="6">zh</td><td rowspan="2">None</td><td>Adapter</td><td>93.70</td><td>74.08</td><td>95.07</td><td>90.97</td><td>93.44</td><td>85.53</td><td>74.05</td><td>90.99</td><td>88.47</td><td>90.22</td><td>87.65</td></tr><tr><td>Soft Prompt</td><td>91.44</td><td>66.83</td><td>92.58</td><td>94.18</td><td>84.92</td><td>77.69</td><td>66.55</td><td>74.85</td><td>90.33</td><td>81.90</td><td>82.13</td></tr><tr><td rowspan="2">Adapter</td><td>Adapter</td><td>97.08</td><td>84.02</td><td>97.09</td><td>90.20</td><td>97.27</td><td>95.58</td><td>84.14</td><td>90.62</td><td>90.89</td><td>90.58</td><td>91.75</td></tr><tr><td>Soft Prompt</td><td>66.88</td><td>69.81</td><td>65.43</td><td>67.65</td><td>67.11</td><td>66.28</td><td>67.57</td><td>65.58</td><td>65.92</td><td>65.92</td><td>66.82</td></tr><tr><td rowspan="2">Soft Prompt</td><td>Adapter</td><td>96.07</td><td>76.96</td><td>93.18</td><td>89.41</td><td>95.15</td><td>94.37</td><td>80.19</td><td>91.13</td><td>91.22</td><td>82.55</td><td>89.02</td></tr><tr><td>Soft Prompt</td><td>95.30</td><td>73.52</td><td>89.10</td><td>94.01</td><td>92.61</td><td>90.44</td><td>62.60</td><td>83.33</td><td>91.33</td><td>85.38</td><td>85.76</td></tr></table>


Table 5: Results for the named-entity recognition task using F1-Score. The best scores are boldfaced, and the second best are underlined.


Table 6: For NLI, we report accuracy as a metric. The best results for each language pair are highlighted in bold and the second best are underlined. Additionally, language pairs with improved performance compared to inference-only are marked in green, and those with decreased performance are marked in red 

Table 7: Results for the check-worthy claim detection task for cross-lingual transfer. Results are reported using F1-Score, with the best scores in bold and the second best underlined. 

<table><tr><td>Language Representation</td><td>Task Representation</td><td>bg</td><td>cs</td><td>el</td><td>ml</td><td>ro</td><td>sl</td><td>sk</td><td>sw</td><td>te</td><td>ur</td></tr><tr><td rowspan="2">None</td><td>Adapter</td><td><eq>38.46 \pm 9.60</eq></td><td><eq>\underline{66.00} \pm 0.77</eq></td><td><eq>55.88 \pm 5.33</eq></td><td><eq>40.99 \pm 0.67</eq></td><td><eq>60.97 \pm 1.74</eq></td><td><eq>56.29 \pm 0.62</eq></td><td><eq>67.55 \pm 1.22</eq></td><td><eq>\underline{58.60} \pm 1.97</eq></td><td><eq>47.69 \pm 2.35</eq></td><td><eq>29.15 \pm 8.78</eq></td></tr><tr><td>Soft Prompt</td><td><eq>28.98 \pm 3.24</eq></td><td><eq>61.48 \pm 0.69</eq></td><td><eq>50.64 \pm 6.63</eq></td><td><eq>43.08 \pm 1.81</eq></td><td><eq>52.65 \pm 1.58</eq></td><td><eq>54.03 \pm 0.98</eq></td><td><eq>62.42 \pm 0.75</eq></td><td><eq>56.49 \pm 2.86</eq></td><td><eq>43.94 \pm 1.64</eq></td><td><eq>26.67 \pm 3.05</eq></td></tr><tr><td rowspan="2">Adapter</td><td>Adapter</td><td><eq>\underline{66.85} \pm 2.03</eq></td><td><eq>64.78 \pm 0.15</eq></td><td><eq>\underline{69.13} \pm 1.48</eq></td><td><eq>\underline{47.34} \pm 1.86</eq></td><td><eq>\underline{64.07} \pm 4.97</eq></td><td><eq>49.29 \pm 2.42</eq></td><td><eq>68.15 \pm 1.19</eq></td><td><eq>\underline{59.45} \pm 2.28</eq></td><td><eq>25.96 \pm 2.86</eq></td><td><eq>20.20 \pm 11.17</eq></td></tr><tr><td>Soft Prompt</td><td><eq>\underline{63.93} \pm 3.01</eq></td><td><eq>50.76 \pm 14.17</eq></td><td><eq>62.05 \pm 3.25</eq></td><td><eq>43.61 \pm 2.42</eq></td><td><eq>56.11 \pm 3.25</eq></td><td><eq>50.30 \pm 7.88</eq></td><td><eq>53.63 \pm 4.44</eq></td><td><eq>55.27 \pm 3.95</eq></td><td><eq>26.83 \pm 4.04</eq></td><td><eq>\underline{46.63} \pm 1.51</eq></td></tr><tr><td rowspan="2">Soft Prompt</td><td>Adapter</td><td><eq>42.19 \pm 2.93</eq></td><td><eq>\underline{67.25} \pm 0.23</eq></td><td><eq>60.19 \pm 6.02</eq></td><td><eq>43.67 \pm 4.32</eq></td><td><eq>\underline{61.89} \pm 1.73</eq></td><td><eq>\underline{61.75} \pm 1.90</eq></td><td><eq>\underline{72.95} \pm 2.35</eq></td><td><eq>56.29 \pm 0.65</eq></td><td><eq>\underline{52.58} \pm 5.43</eq></td><td><eq>\underline{47.86} \pm 3.77</eq></td></tr><tr><td>Soft Prompt</td><td><eq>50.66 \pm 12.65</eq></td><td><eq>64.00 \pm 2.18</eq></td><td><eq>\underline{63.38} \pm 3.85</eq></td><td><eq>\underline{46.37} \pm 2.46</eq></td><td><eq>55.90 \pm 4.95</eq></td><td><eq>\underline{57.77} \pm 0.71</eq></td><td><eq>\underline{69.87} \pm 2.21</eq></td><td><eq>54.26 \pm 1.31</eq></td><td><eq>\underline{48.35} \pm 2.19</eq></td><td><eq>37.46 \pm 3.93</eq></td></tr></table>


Table 8: Results of cross-lingual transfer from German to six languages for the NER task. We report the mean of three runs along with the standard deviation. The best results are bolded and the second best results are underlined.
