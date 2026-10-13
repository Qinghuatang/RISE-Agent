# Discarding the Crutches: Adaptive Parameter-Efficient Expert Meta-Learning for Continual Semantic Parsing

Ruiheng Liu<sup>1,2</sup>, Jinyu Zhang<sup>2</sup>, Yanqi Song<sup>2</sup>, Yu Zhang<sup>2</sup>*, Bailong Yang<sup>1</sup>* <sup>1</sup>Xi’an Research Institute of High-Tech, Xi’an, China <sup>2</sup>Harbin Institute of Technology, Harbin, China {rhliu, jyzhang, yqsong, zhangyu}@ir.hit.edu.cn, xa_403@163.com 

## Abstract

Continual Semantic Parsing (CSP) enables parsers to generate SQL from natural language questions in task streams, using minimal annotated data to handle dynamically evolving databases in real-world scenarios. Previous works often rely on replaying historical data, which poses privacy concerns. Recently, replay-free continual learning methods based on Parameter-Efficient Tuning (PET) have gained widespread attention. However, they often rely on ideal settings and initial task data, sacrificing the model’s generalization ability, which limits their applicability in realworld scenarios. To address this, we propose a novel Adaptive PET eXpert meta-learning (APEX) approach for CSP. First, SQL syntax guides the LLM to assist experts in adaptively warming up, ensuring better model initialization. Then, a dynamically expanding expert pool stores knowledge and explores the relationship between experts and instances. Finally, a selection/fusion inference strategy based on sample historical visibility promotes expert collaboration. Experiments on two CSP benchmarks show that our method achieves superior performance without data replay or ideal settings, effectively handling cold-start scenarios and generalizing to unseen tasks, even surpassing performance upper bounds<sup>1</sup>. 

## 1 Introduction

The rapid development of Large Language Models (LLMs) has greatly advanced semantic parsing, providing non-expert users with a convenient interface for querying and analyzing large-scale structured data (Yang et al., 2024; Zhang et al., 2024a; Qu et al., 2024; Li et al., 2024). However, frequent database updates require semantic parsers to adapt to evolving data environments dynamically. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/707ff2e9-62f8-4241-abf7-69952480efc5/a51ab283ee9166b303b4e83ccf48317ba030b6698aeeec15f011138f1bcb7fa8.jpg)



Figure 1: CSP process based on the PET method. Each task in the task stream learns a dedicated PET module and adds it to the pool. Previous method: ① Require manual data collection for warming up parsers. ② Assume known task identifier and load Oracle PET modules. ③ Handle only seen tasks, unable to generalize to new tasks. Our method: ① No need for manual warm up data. ② Model autonomously selects/fuses PET modules. ③ Handle both seen and unseen tasks.


Consequently, task stream-oriented continual semantic parsing has garnered significant attention from academia and industry, presenting three major challenges: (1) Few-shot Challenge: For new databases, obtaining sufficient text-to-SQL pairs in a short time is difficult, which may lead to overfitting of the parser (Chen et al., 2023b). (2) Catastrophic Forgetting: In continual learning, a model may forget previously learned knowledge while acquiring new information (Wang et al., 2024). (3) Knowledge Transfer: Accumulating and generalizing knowledge from task sequences to unseen data is challenging (Zhao et al., 2024). 

To tackle these challenges, recent studies have explored PET methods to prevent forgetting caused by extensive parameter updates (Wang et al., 2022; Chen et al., 2023c; Yadav et al., 2023). As shown in Figure 1, PET modules such as prompt (Lester et al., 2021) are trained to capture task-specific knowledge and stored in a PET pool. During testing, the appropriate PET module is selected from the pool. While these methods have made some progress in reducing training overhead and preventing catastrophic forgetting, they still have limitations: (1) Ideal Setting: Current methods perform well in preventing catastrophic forgetting but rely on a simplified scenario where task identifiers are known during testing, enabling the selection of the appropriate PET module. This assumption ignores the shared SQL syntax across tasks, limiting knowledge transfer and accumulation, and sacrificing the model’s ability to generalize, making it less suitable for real-world applications (Chen et al., 2023c). (2) Initial Task Data Dependency: General language models are pre-trained on large-scale natural language corpora. Previous methods often depend on substantial labeled data for the initial task and su pervised fine-tuning to ensure proper initialization of the model backbone, enabling adaptation to complex and dynamic semantic parsing task streams. Consequently, CSP performance heavily relies on this initial labeled data, which conflicts with the few-shot challenge (Chen et al., 2023b,c). 

Inspired by the Mixture-of-Experts (MoE) theory (Jacobs et al., 1991), we propose a novel CSP framework, named APEX. The framework first builds a dynamically expanding shared expert pool and trains PET experts with domain-specific knowledge for each task. To enable the model to actively select the appropriate expert at the instance level, we assign a meta-representation as an index to each expert and propose a meta-learning approach for CSP to model the matching relationship between experts and instances. During inference, we propose a routing strategy based on the historical visibility of samples to facilitate expert collaboration. This approach effectively prevents catastrophic forgetting and improves generalization without task identifiers. Moreover, to reduce the dependency on initial task data, we propose an SQL syntaxguided adaptive warm-up method. It infers the SQL syntax features of current and future tasks through induction and evolution and then guides LLMs to synthesize pseudo sample pairs. Accuracy and fidelity are enhanced through correction and filtering strategies. Extensive experimental results demonstrate that our method achieves superior performance without replaying historical data, ideal settings, or manual data collection for warmup, outperforming previous rehearsal-based methods by up to 15.8%. Compared to methods with ideal settings, our approach not only adapts well to cold-start scenarios but also exhibits strong forward transfer capabilities, generalizing to unseen tasks and even exceeding the upper performance bound by as much as 8.8%. The main contributions of this work are summarized as follows: 

• We propose a novel CSP framework, APEX, which accumulates knowledge via a dynamically expanding expert pool, without replaying historical data, and supports adaptive warm-up without manual intervention. 

• We introduce a multi-expert meta-learning method and an active selection/fusion strategy based on task visibility to guide expert collaboration for different instances in CSP. 

• Experimental results show that our method surpasses baselines, exceeding performance limits, and handles cold-start scenarios without relying on data replay or ideal settings. 

## 2 Related Work

## 2.1 Continual Semantic Parsing

Previous semantic parsing research primarily focuses on fine-tuning Pre-trained Language Models (PLMs) on large static datasets (Cai et al., 2022; Li et al., 2023a), but high-quality SQL annotations often require domain experts, making it costly. Recently, with the rapid advancement of LLMs (Anil et al., 2023; OpenAI et al., 2024), few-shot in-context learning has made significant progress (Li et al., 2023b; Mao et al., 2024). However, these methods often rely on powerful closed-source LLMs, posing risks of data privacy breaches and high inference costs (Xue et al., 2024; Li et al., 2024; Zhang et al., 2024b). Furthermore, they are still designed for static data, making it difficult to update and iterate models as environments change. Inspired by human knowledge accumulation and adaptive learning paradigms, continuous semantic parsing based on few-shot and dynamic data has become a promising direction (Yadav et al., 2023; Chen et al., 2023b,c). Earlier related works focus on rehearsal-based methods (Li et al., 2021; Chen et al., 2023b), which prevent forgetting by replaying real samples from past tasks during current training. While simple and effective, these methods depend on the number of replayed samples and are unsuitable for scenarios with data privacy concerns or limited storage resources. Recently, PET-based methods that do not rely on replay have gained significant attention (Smith et al., 2023; Liu et al., 2024a; Zhao et al., 2024). 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/707ff2e9-62f8-4241-abf7-69952480efc5/d1397fac23d41be21616d2e1921f1b4837153d57071725ed942ff8bb1951f1d4.jpg)



Figure 2: The overall architecture of APEX. [T], [C], and [V] represent table names, column names, and database values, respectively. For more details, see Section 4.1.


## 2.2 Parameter-Efficient Continual Learning

Techniques like prompt tuning (Lester et al., 2021), Adapter (Poth et al., 2023), and LoRA (Hu et al., 2022) in PET enable LLMs to adapt to lowresource downstream tasks by efficiently updating knowledge without large-scale parameter updates (Liao et al., 2024). These methods help mitigate distribution shifts and prevent catastrophic forgetting in continual learning without relying on historical data replay (Chen et al., 2023c; Zhao et al., 2024). Typically, PET modules are trained separately for each task to decouple task-specific knowledge. During inference, two settings are used: Task-aware, where the task identity of each sample is known, allowing the appropriate PET module to be loaded (Chen et al., 2023c). This ideal setting simplifies the continual learning scenario but struggles to generalize to unseen samples, making it difficult to apply in real-world practice. A more challenging and realistic scenario is Task-agnostic, where the model needs to autonomously organize PET modules (Wang et al., 2022; Razdaibiedina et al., 2023; Zhao et al., 2024; Menabue et al., 2024). 

## 3 Problem Definition

The goal of semantic parsing is to generate an SQL query $y ~ = ~ F ( \mathcal { Q } , \mathcal { D } )$ from a natural language question $\mathcal { Q }$ and a database $\mathcal { D } \ = \ ( \mathcal { L } , \mathcal { C } )$ where $\mathcal { L }$ represents the set of table names, C is the set of corresponding column names, and $F$ is the parser. CSP aims to learn from a sequence of tasks $\{ \mathcal { T } _ { 1 } , \mathcal { T } _ { 2 } , . . . , \mathcal { T } _ { N _ { T } } \}$ , where $N _ { T }$ denotes the number of tasks. Each task involves an independent semantic parsing task $\mathcal { T } _ { t } = \{ ( \mathcal { X } _ { t } ^ { i } , \mathcal { Y } _ { t } ^ { i } ) \} _ { i = 1 } ^ { N _ { t } ^ { - } }$ , with $\mathcal { X } = ( \mathcal { Q } , \mathcal { D } )$ , and $N _ { t }$ is the data size. The database information between tasks is non-overlapping. At time step t, the parser $F$ is trained only on the data from the current task $\mathcal { T } _ { t } ,$ , without access to previous tasks’ data. After training, the model is expected to perform well on all seen tasks and generalize effectively to unseen future tasks. 

## 4 Methodology

## 4.1 Overview

The framework of APEX is shown in Figure 2, consisting of three stages: ①Warm-up preparation (Figure 2(a)): To reduce reliance on large manually labeled data (Chen et al., 2023b,c), we propose a SQL syntax-guided adaptive warm-up method using LLMs. SQL prototypes are extracted from existing samples and evolved for potential future tasks, guiding LLMs to generate question-SQL pairs with accuracy and fidelity ensured through correction and filtering. ②Training (Figure 2(b)): A dynamically expanding expert pool is first established (Figure 2(c)). We then propose a multiexpert meta-learning approach to capture instanceexpert commonalities and differences, enabling the most suitable expert collaboration for each sample. ③Prediction (Figure 2(d)): For task-agnostic samples, we infer task visibility and apply different decoding strategies. For seen tasks, the most relevant expert is selected, for unseen tasks, multiple experts are fused for collaborative inference. 

## 4.2 Adaptive Warm-Up with LLMs

To enhance the model’s initial semantic parsing performance, previous methods rely on large amounts of manually labeled data for supervised fine-tuning. However, in real-world settings, acquiring sufficient labeled data at the start of continual learning is challenging (Chen et al., 2023b). To overcome this issue, inspired by related work (Liu et al., 2024b), we propose a SQL syntax-guided adaptive warm-up method leveraging LLMs’ generation and reasoning abilities, as shown in Figure 2(a). All prompt details are provided in Appendix A. 

SQL Prototype Extraction The model’s understanding of SQL syntax reflects its semantic parsing ability. While databases vary across CSP tasks, they share the same SQL syntax. Early in continual learning, limited labeled data hinders the model’s grasp of diverse SQL. Thus, we extract SQL syntax features from two perspectives: reinforcing current knowledge and anticipating future syntax. Considering SQL’s code-like properties, we use CodeT5 (Wang et al., 2021) as the encoder to jointly represent the question Q, schemas D, and SQL Y from existing samples, and then apply k-means clustering to decouple the complexity of semantic parsing. This allows each cluster to share similar sub-SQL knowledge. Next, database-specific information is removed from the SQL of cluster centers $A _ { c } ,$ to obtain syntax-focused SQL prototypes ${ \mathcal { A } } _ { p } .$ . However, limiting the model to existing syntax may restrict its scope. To address this, we use the extracted prototypes as seeds, combining them with common SQL keywords, and leverage LLMs to infer potential future SQL prototypes to obtain ${ \mathcal { A } } _ { p } ^ { * } .$ 

Pseudo-Sample Synthesis To tackle the scarcity of labeled data, previous works often rely on semisupervised learning (Qin and Joty, 2022), such as self-training (Chen et al., 2023b), which requires extra data and yields low-quality labels due to its one-way generation. We propose a SQL syntaxguided soft synthesis method using LLMs, where LLMs are provided with database schemas and SQL prototypes $A = \{ \mathcal { A } _ { p } , \mathcal { A } _ { p } ^ { \ast } \}$ to generate natural language questions $\hat { \mathcal { Q } }$ and corresponding SQL $\hat { \mathcal { V } }$ simultaneously. This bidirectional generation taps into LLMs’ internal knowledge, avoiding one-way limitations and removing the need for additional semi-supervised data. 

Correction and Filtering Although our method mitigates some challenges in pseudo-sample synthesis, LLMs’ hallucination issues (Qu et al., 2024) can still produce SQL queries with execution errors or deviations from the intended syntax. To enhance accuracy and fidelity, inspired by prior code-related works (Huang et al., 2024; Chen et al., 2023a), we design a two-stage correction-filtering strategy. First, the LLM performs self-correction, retaining only executable samples. Then, we mask the database schema and select the Top- $N _ { g }$ samples with the smallest edit distance to the target syntax. 

## 4.3 Multi-Expert Meta-Learning

After the warm-up phase, our method splits into two parts: multi-expert meta-learning and active selection or fusion, corresponding to the training and prediction stages in CSP, as shown in Figure 2(b)-(d). During training, the matching relationship between the meta-representations of different PET experts and instances is modeled and optimized. In inference, dynamic recognition of a sample’s visibility guides the decoding strategy, mitigating forgetting and enhancing generalization. 

Dynamic Expert Pool Expansion Unlike previous work that assigns an Oracle expert module to test samples (Chen et al., 2023c), this work focuses on real-world continual learning scenarios where the task identifier is unknown. Inspired by prior work (Wang et al., 2022), an expert pool is maintained to encourage the model to actively invite experts to collaborate, as shown in Figure 2(c). We observe that previous methods use a large, fixed expert pool, which makes early expert collaboration challenging (Yadav et al., 2023). To address this, we propose a dynamically expanding expert pool that grows with tasks. For each task $\mathcal { T } _ { t } .$ , we similarly use k-means to cluster the samples into $K$ groups, train a PET module for each cluster as its expert, and assign a learnable meta-representation as a key. The expert pool then expands to include all key-expert paris up to task $\mathcal { T } _ { t } .$ , forming the pool $\mathcal { E } _ { t }$ , represented as: 

$$
\mathcal {E} _ {t} = \{(\mathbf {k} _ {1}, P _ {1}), (\mathbf {k} _ {2}, P _ {2}), \ldots , (\mathbf {k} _ {N _ {p} ^ {t}}, P _ {N _ {p} ^ {t}}) \}\tag{1}
$$

Here, k $\in \mathbb { R } ^ { 1 \times d }$ represents different keys, initialized from the hidden state representation of each cluster center $x _ { c } ,$ i.e., $\mathbf { k } ^ { \mathrm { i n i t } } = F _ { e } ( x _ { c } )$ . d is the encoding dimension. $P _ { i }$ represents the i-th PET expert module, with $i \in [ 1 , N _ { p } ^ { t } ]$ , and $N _ { p } ^ { t }$ denotes the total number of experts in the pool at task $\mathcal { T } _ { t }$ 

During training, the model actively queries keyvalue pairs to select the appropriate expert for each sample. We first use the parser’s frozen encoder as a function $F _ { e }$ to extract query features of the sample: $q ( x ) = F _ { e } ( x )$ , then apply a scoring function $\Phi ( \cdot )$ to find the most relevant key $\mathbf { k } ^ { * }$ in $\mathcal { E } _ { t }$ : 

$$
\mathbf {k} ^ {*} = \underset {\mathbf {k} _ {i} \in \mathcal {E} _ {t}} {\arg \max} \Phi (q (x), \mathbf {k} _ {i})\tag{2}
$$

Where $\begin{array} { r } { \Phi ( \mathbf { u } , \mathbf { v } ) = \frac { \mathbf { u } \cdot \mathbf { v } } { \| \mathbf { u } \| \| \mathbf { v } \| } } \end{array}$ represents the similarity between u and v. This design eliminates the need for real historical data during training or Oracle task identifiers during prediction. 

Meta-Continual Learning In each training step t, the most relevant expert module is selected based on the above strategy, loaded into the frozen backbone model for training, and the match loss between the query vector and the meta-representation is then computed as follows: 

$$
\mathcal {L} _ {\text { match }} = - \sum_ {x _ {t} \in \mathcal {T} _ {t}} \Phi (q (x _ {t}), \mathbf {k} ^ {*})\tag{3}
$$

Although the above calculation helps the model select expert modules by learning keys at the instance level, this approach treats different keys independently (Wang et al., 2022), overlooking the relationships between them. As a result, the model struggles to capture the similarities and differences among various experts effectively. To tackle this, we propose a Meta-Contrastive Loss (MCL) in CSP, aligning the current task with the relevant key while enhancing differentiation from other tasks, and improving expert distinguishability during inference. 

For the sample x<sub>t</sub>, we treat its query feature $q ( x _ { t } )$ and the corresponding task’s key ${ \mathbf { k } } ^ { + }$ as a positive pair, while $q ( x _ { t } )$ and keys from other tasks $\mathbf { k } ^ { - } \in \mathcal { K }$ are negative pairs. We then apply a nonlinear projection $p ( \cdot )$ to map the query feature into the latent space and compute the following: 

$$
\mathbf {h} _ {t} = p (q (x _ {t})) = \mathbf {W} _ {2} \mathrm{ReLU} (\mathbf {W} _ {1} q (x _ {t}))\tag{4}
$$

Here, $\mathbf { W } _ { 1 }$ and $\mathbf { W } _ { 2 }$ are trainable weight matrices. Next, we calculate the meta-contrastive loss: 

$$
\begin{array}{c} \mathcal {L} _ {\mathrm{mcl}} = \\ - \sum_ {t} ^ {| \mathcal {T} _ {t} |} \log \frac {\exp (\Phi_ {t} ^ {+} / \tau)}{\exp (\Phi_ {t} ^ {+} / \tau) + \sum_ {\mathbf {k} ^ {-} \in \mathcal {K} ^ {-}} \exp (\Phi_ {t} ^ {-} / \tau)} \end{array}\tag{5}
$$

Where $\Phi _ { t } ^ { + } = \Phi ( { \bf h } _ { t } , { \bf k } ^ { + } )$ , and similarly for $\Phi _ { t } ^ { - }$ $\tau$ is the temperature factor. We define the training loss of PET parameters $\theta _ { \mathrm { { p e t } } }$ as: $\begin{array} { r l } { \mathcal { L } _ { \mathrm { t a s k } } } & { { } = } \end{array}$ $\textstyle - \sum _ { ( x _ { t } , y _ { t } ) \in \mathcal { T } _ { t } } ^ { \mathcal { T } _ { t } } \log P ( y _ { t } | x _ { t } ; \theta _ { \mathrm { p e t } } )$ . Thus, the total loss with weight factors α and $\beta$ is: 

$$
\mathcal {L} _ {\text { total }} = \mathcal {L} _ {\text { task }} + \alpha \mathcal {L} _ {\text { match }} + \beta \mathcal {L} _ {\text { mcl }}\tag{6}
$$

Active Selection or Fusion During prediction, samples are classified as seen (from previous tasks) or unseen (from new tasks). For seen samples, the expert module from their original task is loaded, while for unseen samples, relevant expert modules are dynamically selected and combined based on SQL skills. We propose a simple but effective routing mechanism to predict sample historical visibility and choose the appropriate inference strategy. It calculates the similarity between the test sample and keys of all PET modules in the expert pool, using the z-score (standard score) (Abdi, 2007) to measure visibility. A high z-score indicates a significant deviation from the mean, suggesting a strong association with a specific expert, while a low z-score indicates a weak association with the current expert. The calculation is as follows: 

$$
Z _ {i} = \frac {\Phi (q (x) , \mathbf {k} _ {i}) - \mu}{\sigma}\tag{7}
$$

In this case, $Z _ { i }$ represents the z-score of sample x with the i-th expert module, where $i \in [ 1 , N _ { p } ]$ . µ is the mean of the similarity scores with all expert modules, and $\sigma$ is the standard deviation. Finally, we select the largest $Z _ { i }$ among all experts as the final target: $Z =$ max $\left( Z _ { 1 } , Z _ { 2 } , \ldots , Z _ { N _ { p } } \right)$ 

We also introduce a threshold $\gamma$ to infer sample historical visibility by comparing it with $Z .$ , which guides the inference strategy. For seen tasks, the 

Top-1 expert module is loaded. For unseen tasks, the Top-N experts are selected, and their scores are normalized to serve as weight proportions for fusing the parameters of these modules. The expert key set $\hat { \boldsymbol { \cal K } } \in \mathcal { E }$ selection process is as follows: 

$$
\hat {\mathcal {K}} = \left\{ \begin{array}{l l} \arg \max \Phi (q (x), \mathbf {k} _ {i}) & \text { if } Z \geq \gamma \\ \operatorname{Top-} N (\Phi (q (x), \mathbf {k} _ {i})) & \text { else } \end{array} \right.\tag{8}
$$

The calculation process of expert module parameter selection or fusion is as follows: 

$$
\hat {P} = \left\{ \begin{array}{l l} V (\hat {\mathcal {K}}) & \text { if } Z \geq \gamma \\ \sum_ {i = 1} ^ {N} w _ {i} V (\hat {\mathcal {K}}) & \text { else } \end{array} \right.\tag{9}
$$

Where $\begin{array} { r } { w _ { i } = \frac { e ^ { \Phi ( q ( x ) , \mathbf { k } _ { i } ) } } { \sum _ { j = 1 } ^ { N } e ^ { \Phi ( q ( x ) , \mathbf { k } _ { j } ) } } . \ V ( \cdot ) } \end{array}$ maps a key to expert module parameters. 

## 5 Experiments

## 5.1 Datasets

We evaluate our method on two benchmarks: Spider-stream and Combined-stream (Chen et al., 2023c). Spider-stream, built from the Spider (Yu et al., 2018), has most tasks with fewer than 500 labeled samples. It assesses the impact of domain shifts in databases on CSP under limited resources. Combined-stream, constructed from Spider and WikiSQL (Zhong et al., 2018), introduces single-table queries from WikiSQL, evaluating performance across both domain shifts and SQL complexity. Notably, in the original setting, large amounts of labeled data are manually collected for the initial task to warm up the model. To better reflect real-world scenarios, we introduce a cold-start setting where the first task does not have the most labeled data, and tasks are shuffled randomly to evaluate the effects of initial performance and task order. More details are in Appendix B.1. 

## 5.2 Evaluation Metrics

Following previous works (Yu et al., 2018; Chen et al., 2023b,c), we adopt four common continual learning metrics and report the official semantic parsing metrics: Exact-set-Match (EM) and EXecution (EX) accuracy. EM checks the correctness of the generated SQL, while EX verifies its execution on the database. (1) Average Accuracy $( \mathrm { A C C _ { a } } ) \mathrm { : }$ : Average performance across all seen tasks. (2) Whole Accuracy $( \mathrm { A C C } _ { \mathrm { w } } ) \colon$ Performance on the combined test set of all seen tasks. (3) Backward Transfer (BWT): Impact on previous tasks after learning the last task. (4) Forward Transfer (FWT): Ability to generalize to unseen tasks. Notably, the previous method (Chen et al., 2023c) fails to generalize to new tasks due to its ideal settings. Metric calculations are detailed in Appendix B.2. 

## 5.3 Implementation Details

Our method is independent of PLMs and PET structures, allowing seamless integration with Transformer-based LLMs. To ensure a fair comparison with prior works, we use the T5 series as the backbone model. For PET, we choose the popular Adapter (Poth et al., 2023) and LoRA (Hu et al., 2022), which are widely used for supervised finetuning of PLMs. During the warm-up phase, we utilize the publicly available Mixtral-8x7B-Instructv0.1, and generate 10 pairs of pseudo-samples per SQL prototype. The expert pool is expanded with K = 5 clusters per task. In training, the temperature factor τ is 0.07, and loss weights α and $\beta$ are 0.3 and 0.1. The inference threshold $\gamma$ is set to 1.6, and the Top-N for fusion is 3. All experiments are run on an A100 GPU with 80GB memory. Further details are provided in Appendix B.3. 

## 5.4 Baselines

We conduct a comprehensive comparison with the following three types of baselines: (1) Original method: Sequential Fine-Tuning (SFT) (Yogatama et al., 2019) reflects the performance of a vanilla model. (2) Rehearsal-based methods: EMAR (Han et al., 2020), SFNET (Chen et al., 2023b). They require replaying real historical examples and using additional unlabeled data. (3) PET-based methods:Fusion (Pfeiffer et al., 2021), PEFT (Chen et al., 2023c), C3 (Chen et al., 2023c), ProgPrompt (Razdaibiedina et al., 2023), L2P (Wang et al., 2022), Soup (Chronopoulou et al., 2023). Some of them rely on ideal settings. As our method uses LLMs to generate pseudo-samples (zero-shot) during the warm-up, we also report its standalone zero-shot performance. The ORACLE setting incrementally trains the model on all task data, representing the upper bound of CSP. More details are in Appendix B.4. 

## 6 Results and Analysis

## 6.1 Overall Results

Table 1 shows the overall results of different methods. Our method demonstrates clear advantages across benchmarks, even exceeding theoretical performance limits on some metrics. Notably, it requires no historical data replay, additional unsupervised data, or idealized settings. More results and analysis are in Appendix C. 

<table><tr><td rowspan="2">Method</td><td rowspan="2">DR</td><td rowspan="2">PET</td><td colspan="5">Spider-stream</td><td colspan="5">Combined-stream</td></tr><tr><td><eq>ACC_a</eq></td><td><eq>ACC_w</eq></td><td>BWT</td><td>FWT</td><td><eq>*ACC_w</eq></td><td><eq>ACC_a</eq></td><td><eq>ACC_w</eq></td><td>BWT</td><td>FWT</td><td><eq>*ACC_w</eq></td></tr><tr><td colspan="13">T5-Base(220M)</td></tr><tr><td rowspan="3">SFT</td><td>X</td><td>Full</td><td>51.9/57.9</td><td>47.1/52.8</td><td>-19.6/-14.9</td><td>29.6/33.2</td><td>45.7</td><td>51.0/36.0</td><td>52.4/38.3</td><td>-11.1/-17.6</td><td>19.9/11.1</td><td>32.0</td></tr><tr><td>X</td><td>Adapter</td><td>40.8/43.5</td><td>38.9/41.7</td><td>-32.3/-31.3</td><td>23.9/27.4</td><td>35.6</td><td>38.8/33.9</td><td>43.0/40.9</td><td>-33.6/-37.8</td><td>11.4/5.4</td><td>18.5</td></tr><tr><td>X</td><td>LoRA</td><td>42.3/47.8</td><td>42.2/47.6</td><td>-28.4/-24.1</td><td>27.2/30.7</td><td>32.6</td><td>46.9/40.6</td><td>48.6/45.1</td><td>-21.0/-26.3</td><td>16.8/11.1</td><td>17.0</td></tr><tr><td rowspan="2">L2P</td><td>X</td><td>Adapter</td><td>22.6/24.3</td><td>22.4/24.2</td><td>-42.2/-41.6</td><td>15.4/17.2</td><td>21.4</td><td>27.0/25.7</td><td>26.8/25.6</td><td>-38.1/-35.0</td><td>8.7/2.4</td><td>12.3</td></tr><tr><td>X</td><td>LoRA</td><td>30.1/32.7</td><td>34.8/38.2</td><td>-33.2/-31.2</td><td>23.0/25.9</td><td>26.0</td><td>27.1/22.0</td><td>26.0/23.2</td><td>-31.0/-35.8</td><td>8.2/3.7</td><td>16.7</td></tr><tr><td>Fusion</td><td>X</td><td>Adapter</td><td>41.1/45.2</td><td>39.7/43.6</td><td>-32.4/-29.7</td><td>24.5/27.4</td><td>23.9</td><td>35.0/29.8</td><td>43.2/40.7</td><td>-33.8/-37.4</td><td>10.2/5.1</td><td>13.5</td></tr><tr><td>Soup</td><td>X</td><td>Adapter</td><td>31.4/34.3</td><td>43.1/46.0</td><td>-1.7/-1.7</td><td>27.5/30.4</td><td>8.1</td><td>25.5/22.7</td><td>40.2/38.9</td><td>-5.2/-3.7</td><td>19.0/15.7</td><td>13.7</td></tr><tr><td>ProgPrompt</td><td>X</td><td>Prompt</td><td>21.0/23.6</td><td>23.5/27.1</td><td>-40.1/-39.7</td><td>16.3/16.9</td><td>7.1</td><td>21.3/19.2</td><td>27.2/26.8</td><td>-43.9/-41.2</td><td>6.1/3.3</td><td>0.1</td></tr><tr><td>EMAR</td><td>√</td><td>Full</td><td>55.6/59.4</td><td>50.6/55.1</td><td>-17.5/-15.0</td><td>28.5/31.1</td><td>53.2</td><td>61.2/59.4</td><td>57.6/57.9</td><td>-7.3/-6.8</td><td>28.4/25.9</td><td>50.1</td></tr><tr><td>SFNET♣</td><td>X</td><td>Full</td><td>51.8/57.8</td><td>47.1/52.2</td><td>-18.9/-13.8</td><td>28.3/32.7</td><td>46.4</td><td>56.2/49.4</td><td>51.8/48.4</td><td>-7.9/-14.4</td><td>21.0/14.6</td><td>24.4</td></tr><tr><td>SFNET♣</td><td>√</td><td>Full</td><td>60.4/62.5</td><td>54.6/57.4</td><td>-11.3/-10.7</td><td>28.8/31.8</td><td>51.3</td><td>61.3/60.9</td><td>57.5/56.7</td><td>-5.4/-5.2</td><td>29.7/27.5</td><td>55.9</td></tr><tr><td rowspan="2">APEX(Ours)</td><td>X</td><td>Adapter</td><td>61.0/62.4</td><td>63.4/64.8</td><td>1.3/1.4</td><td>25.1/28.3</td><td>53.5</td><td>62.7/59.3</td><td>64.9/63.1</td><td>6.8/9.2</td><td>21.9/20.7</td><td>57.7</td></tr><tr><td>X</td><td>LoRA</td><td>63.3/65.3</td><td>62.1/63.9</td><td>0.4/0.5</td><td>24.4/27.5</td><td>57.8</td><td>63.5/61.4</td><td>64.3/63.5</td><td>7.2/10.6</td><td>29.2/25.8</td><td>56.1</td></tr><tr><td colspan="13">Ideal Setting &amp; Upper Bound</td></tr><tr><td rowspan="3">PEFT◇</td><td>X</td><td>Prompt</td><td>65.7/-</td><td>64.5/-</td><td>0.0/0.0</td><td>N/A</td><td>14.9</td><td>63.8/-</td><td>66.2/-</td><td>0.0/0.0</td><td>N/A</td><td>16.6</td></tr><tr><td>X</td><td>Adapter</td><td>69.9/71.1</td><td>70.7/72.0</td><td>0.0/0.0</td><td>N/A</td><td>70.1</td><td>66.7/65.2</td><td>67.6/67.5</td><td>0.0/0.0</td><td>N/A</td><td>64.3</td></tr><tr><td>X</td><td>LoRA</td><td>68.0/69.8</td><td>69.6/71.4</td><td>0.0/0.0</td><td>N/A</td><td>64.7</td><td>63.1/61.4</td><td>67.0/66.7</td><td>0.0/0.0</td><td>N/A</td><td>58.4</td></tr><tr><td>ORACLE</td><td>-</td><td>Full</td><td>75.6/77.0</td><td>76.4/78.4</td><td>4.2/3.3</td><td>29.7/32.0</td><td>-</td><td>70.2/68.2</td><td>71.1/70.8</td><td>5.6/4.7</td><td>29.2/26.6</td><td>-</td></tr><tr><td colspan="13">T5-Large(770M)</td></tr><tr><td rowspan="3">SFT</td><td>X</td><td>Full</td><td>60.6/67.3</td><td>56.2/62.0</td><td>-20.0/-14.6</td><td>39.7/44.4</td><td>56.9</td><td>57.8/48.9</td><td>58.9/53.3</td><td>-7.6/-16.4</td><td>28.3/25.4</td><td>44.3</td></tr><tr><td>X</td><td>Adapter</td><td>47.1/51.8</td><td>48.0/52.8</td><td>-30.5/-27.3</td><td>34.3/37.8</td><td>44.2</td><td>47.3/39.6</td><td>52.0/47.5</td><td>-28.6/-35.0</td><td>16.4/11.0</td><td>22.6</td></tr><tr><td>X</td><td>LoRA</td><td>52.4/56.5</td><td>52.3/56.4</td><td>-25.3/-21.9</td><td>37.6/41.8</td><td>43.8</td><td>47.7/41.2</td><td>46.3/42.6</td><td>-28.7/-35.6</td><td>26.5/21.2</td><td>18.5</td></tr><tr><td>ProgPrompt</td><td>X</td><td>Prompt</td><td>27.9/29.6</td><td>32.8/35.3</td><td>-37.0/-36.5</td><td>17.2/17.2</td><td>11.5</td><td>23.5/20.2</td><td>29.8/27.7</td><td>-46.4/-42.7</td><td>10.2/3.7</td><td>1.4</td></tr><tr><td>EMAR</td><td>√</td><td>Full</td><td>63.9/68.0</td><td>59.2/62.7</td><td>-13.7/-12.0</td><td>36.5/39.5</td><td>59.0</td><td>58.1/55.8</td><td>55.2/54.3</td><td>-12.4/-12.8</td><td>28.1/22.1</td><td>53.0</td></tr><tr><td>SFNET♣</td><td>X</td><td>Full</td><td>54.1/60.2</td><td>50.3/55.6</td><td>-21.0/-17.6</td><td>37.3/41.1</td><td>51.5</td><td>65.0/60.3</td><td>61.5/59.7</td><td>-7.5/-11.9</td><td>28.2/20.4</td><td>32.6</td></tr><tr><td>SFNET♣</td><td>√</td><td>Full</td><td>63.1/66.6</td><td>56.5/59.9</td><td>-10.5/-8.6</td><td>34.3/38.7</td><td>58.6</td><td>66.2/65.2</td><td>61.4/61.3</td><td>-3.9/-3.2</td><td>37.1/35.0</td><td>59.8</td></tr><tr><td rowspan="2">APEX(Ours)</td><td>X</td><td>Adapter</td><td>67.0/68.8</td><td>68.3/70.3</td><td>0.0/4.4</td><td>29.2/32.3</td><td>60.3</td><td>64.3/60.5</td><td>66.0/63.2</td><td>6.8/9.3</td><td>27.1/23.1</td><td>62.9</td></tr><tr><td>X</td><td>LoRA</td><td>69.2/70.9</td><td>70.8/72.6</td><td>-0.4/-0.4</td><td>33.3/37.2</td><td>61.4</td><td>66.3/63.0</td><td>68.3/65.2</td><td>8.4/11.1</td><td>32.8/30.3</td><td>58.5</td></tr><tr><td colspan="13">Ideal Setting &amp; LLMs Performance &amp; Upper Bound</td></tr><tr><td rowspan="3">PEFT◇</td><td>X</td><td>Prompt</td><td>69.8/-</td><td>67.4/-</td><td>0.0/0.0</td><td>N/A</td><td>51.7</td><td>67.3/-</td><td>70.0/-</td><td>0.0/0.0</td><td>N/A</td><td>38.2</td></tr><tr><td>X</td><td>Adapter</td><td>73.3/75.5</td><td>73.8/76.0</td><td>0.0/0.0</td><td>N/A</td><td>74.7</td><td>70.4/68.1</td><td>71.3/70.4</td><td>0.0/0.0</td><td>N/A</td><td>70.1</td></tr><tr><td>X</td><td>LoRA</td><td>75.7/76.8</td><td>76.1/77.5</td><td>0.0/0.0</td><td>N/A</td><td>72.3</td><td>70.6/70.3</td><td>71.8/72.6</td><td>0.0/0.0</td><td>N/A</td><td>66.7</td></tr><tr><td>C3◇</td><td>X</td><td>Prompt</td><td>70.7/-</td><td>68.9/-</td><td>0.0/0.0</td><td>N/A</td><td>-</td><td>69.0/-</td><td>71.2/-</td><td>0.0/0.0</td><td>N/A</td><td>-</td></tr><tr><td>C3◇†</td><td>X</td><td>Prompt</td><td>71.3/-</td><td>69.6/-</td><td>0.0/0.0</td><td>N/A</td><td>-</td><td>67.6/-</td><td>70.0/-</td><td>0.0/0.0</td><td>N/A</td><td>-</td></tr><tr><td>Mixtral-8x7B</td><td>X</td><td>-</td><td>23.9/29.7</td><td>23.1/28.5</td><td>-/-</td><td>-/-</td><td>-</td><td>28.6/24.4</td><td>27.0/26.0</td><td>-/-</td><td>-/-</td><td>-</td></tr><tr><td>ORACLE</td><td>-</td><td>Full</td><td>80.9/82.8</td><td>80.5/82.6</td><td>2.4/2.9</td><td>38.1/41.3</td><td>-</td><td>73.7/73.2</td><td>75.8/76.0</td><td>1.9/2.3</td><td>37.3/34.2</td><td>-</td></tr></table>


Table 1: Results (EM/EX) on Spider-stream and Combined-stream datasets (%). A uses Mixtral-8x7B for warm-up. DR indicates whether historical data replay is used, with a memory size set to 15. PET refers to different training architectures, with FULL being full-parameter tuning. $\sharp \mathsf { A C C } _ { \mathrm { w } }$ represents the $\mathbf { A C C } _ { \mathrm { w } } – \mathbf { E M }$ in the cold start scenario, detailed results can be found in Appendix C.2. ♣ indicates the need for additional unsupervised data, and ♢ means a known task ID, but forward transfer is impossible (N/A). Some PEFT and C3 results are from their original papers or official code repositories. † signifies the use of GPT-3.5 as the teacher. The best results are highlighted in bold, and the second-best results are underlined. Our results are the average of three random runs.


VS. Rehearsal-Based Methods These baselines rely on full parameter fine-tuning, data replay, and additional data, differing from ours and making comparisons less fair. Meanwhile, SFT results show that simply adding PET modules reduces overhead but lowers performance. Despite this, APEX achieves significant improvements without relying on data strategies, outperforming the rehearsal-based SFNET with a BWT increase of up to 15.8% on the Combined-stream. Notably, SFNET’s performance drops below SFT when replay strategies are removed, underscoring its heavy reliance on replay, whereas APEX remains unaffected. More results are in Appendix C.1. 

VS. PET-Based Methods They can be divided into with and without ideal settings. Baselines without ideal settings perform poorly due to task-level PET module combinations lacking fine-grained guidance. L2P, despite instance-level selection, is limited by a fixed prompt pool and single selection strategy, hindering knowledge accumulation and ignoring sample visibility (Yadav et al., 2023). Baselines relying on ideal settings use task identifiers, restricting knowledge sharing and making them unsuitable for new tasks. In contrast, APEX excels without ideal settings by flexibly accumulating knowledge and organizing expert collaboration. It mitigates forgetting, enhances generalization, and ensures efficient parameter utilization. APEX outperforms PET-based methods across most metrics, surpassing ORACLE in BWT by 8.8% on the Combined-stream, proving its effectiveness. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/707ff2e9-62f8-4241-abf7-69952480efc5/c3c685e75fa10075acd57e7f1337e3a4577d8e1eef95c93814ddb75772c6b9cb.jpg)



Figure 3: Comparison of results (EM) across all seen tasks on Spider-stream (T5-base).


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/707ff2e9-62f8-4241-abf7-69952480efc5/0bd2e014799c61639a3fc88fadc1269b7753e51e5e3d9de43778e63a19408052.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/707ff2e9-62f8-4241-abf7-69952480efc5/34a81b56040b40a1edb3c68385a3106cf8b6ee4d50676cb895b3e89d58563cf2.jpg)



Figure 4: t-SNE visualization results without MCL (left) and with MCL (right). • represent samples from the same task. ✗ denotes the specific key related to the current task, and ✗ indicates unrelated keys.


Cold Start Results Table 1 also shows the performance of different methods in the cold-start scenario, where most methods experience a performance drop, highlighting the challenges of this scenario. In particular, PET-based methods suffer a noticeable decline, emphasizing their reliance on manually collected warm-up data. ProgPrompt struggles on the Combined-stream dataset due to increased task diversity and its progressive prompt design, which limits input length. In contrast, our method uses LLMs for adaptive warm-up, adapts well to cold-start conditions, and performs best. See Appendix C.2 for more detailed results. 

## 6.2 Detailed Results and Analysis

Results Till the Seen Tasks Figure 3 shows that our method maintains stable and significant advantages in all tasks, even surpassing ORACLE. Most PET-based methods suffer severe forgetting as tasks increase, while rehearsal-based methods perform slightly better. APEX with LoRA significantly boosts generalization, consistently surpassing ORACLE. More details are in Appendix C.3. 

Impact of Warm-Up and MCL Table 2 shows that our LLM-based warm-up method significantly boosts both forgetting prevention and generalization, with gains up to 9.3%. Additionally, MCL consistently improves model performance. Further results in Figure 4 demonstrate that MCL effectively pulls samples closer to relevant keys and distances them from irrelevant ones, helping to predict the sample’s historical visibility. 

<table><tr><td></td><td><eq>\text{ACC}_{\text{a}}</eq></td><td><eq>\text{ACC}_{\text{w}}</eq></td><td>BWT</td><td>FWT</td></tr><tr><td>APEX</td><td>62.7/59.3</td><td>64.9/63.1</td><td>6.8/9.2</td><td>21.9/20.7</td></tr><tr><td>w/o Warm up</td><td>57.7/50.1</td><td>60.5/57.4</td><td>-0.5/-0.1</td><td>15.8/14.2</td></tr><tr><td>w/o MCL</td><td>60.6/57.0</td><td>61.9/59.7</td><td>8.0/9.2</td><td>20.2/17.1</td></tr><tr><td>S/F(<eq>\gamma</eq>=1.4)</td><td>62.0/61.2</td><td>65.5/63.7</td><td>7.7/10.2</td><td>17.0/13.6</td></tr><tr><td>S/F(<eq>\gamma</eq>=1.5)</td><td>63.4/59.5</td><td>64.5/63.3</td><td>8.7/11.0</td><td>19.4/16.4</td></tr><tr><td>S/F(<eq>\gamma</eq>=1.7)</td><td>60.8/57.4</td><td>61.9/60.3</td><td>5.2/6.8</td><td>23.9/21.2</td></tr><tr><td>S/F(Top-2)</td><td>62.9/59.6</td><td>65.3/63.5</td><td>4.4/5.3</td><td>24.0/21.7</td></tr><tr><td>S/F(Top-4)</td><td>62.7/59.3</td><td>64.9/63.0</td><td>8.7/10.6</td><td>21.3/20.0</td></tr><tr><td>only S</td><td>60.6/57.2</td><td>60.8/59.4</td><td>-1.0/-1.0</td><td>18.2/15.7</td></tr><tr><td>only F(Top-2)</td><td>45.8/38.8</td><td>47.8/42.2</td><td>1.4/0.8</td><td>29.9/23.8</td></tr><tr><td>only F(Top-3)</td><td>33.6/26.9</td><td>39.9/34.3</td><td>0.1/0.8</td><td>23.5/18.1</td></tr><tr><td>only F(Top-4)</td><td>29.3/23.2</td><td>36.3/30.4</td><td>0.4/0.6</td><td>20.7/15.7</td></tr></table>


Table 2: Ablation study on Combined-stream (Adapter, T5-base). S/F indicates prediction using Select/Fuse.


Impact of Selection/Fusion We experiment with different Top-N, γ thresholds, and inference strategies, including selecting only the top expert (treating all samples as seen) and only fusing experts (treating all samples as unseen). γ is crucial in balancing the model’s forgetting and generalization. From Table 2, we observe that the lower γ, the more likely a sample is to be classified as seen, improving forgetting-related metrics $\mathrm { ( A C C _ { a } , }$ $\mathsf { A C C } _ { \mathbf { w } }$ , BWT) but lowering generalization $( \mathrm { F W T } )$ Conversely, increasing γ enhances generalization but increases forgetting. Thus, we search for an optimal $\gamma$ to balance forgetting and generalization. Using only selection or fusion strategies results in a significant overall performance drop, although the fusion-only strategy can improve the model’s generalization. The number of Top-N experts in the fusion phase also impacts performance. In APEX, varying the number of experts consistently yields good results, while in the fusion-only strategy, increasing experts degrades performance. 

## 7 Conclusion

This work presents a novel parameter-efficient approach for continual semantic parsing. It uses LLMs for adaptive warm-up without manual intervention and builds a dynamically expanding expert pool to accumulate task-specific knowledge. Through meta-continual learning and a routing module, the model actively selects appropriate experts for samples with unknown task identifiers. Importantly, it requires no real historical data or ideal settings, while delivering strong performance, making it well-suited for real-world applications. 

## Limitations

This study has several limitations that warrant further exploration in future work: (1) Optimization of the expert pool: although our PET-based method effectively reduces training costs, the growing number of tasks in the task stream leads to an accumulation of PET modules in the expert pool. This results in knowledge redundancy and increased storage overhead. Streamlining the expert pool through pruning and refinement will be essential for managing longer task streams in practical applications. (2) Balancing past and future: our research shows that combining sample historical visibility with flexible selection/fusion inference strategies can significantly improve model performance. However, this improvement largely depends on the accuracy of predicting historical visibility. Exploring better approaches to distinguish between past and future tasks remains a promising direction. (3) Scaling to more LLMs and tasks: while this work primarily focuses on CSP tasks, our approach can also be applied to continual learning in other domains, such as code generation and complex reasoning. Moreover, it has the potential to integrate with more open-source LLMs and PET technologies, broadening its applicability. 

## Ethics Statement

In this paper, we propose a novel continual semantic parsing framework called APEX. We conduct experiments on two publicly available benchmark datasets, Spider-stream and Combined-stream, ensuring no ethical concerns. 

## Acknowledgments

We would like to thank the anonymous reviewers for their valuable comments. This work is supported by the National Natural Science Foundation of China (No. 62476066 and No. 62277002). 

## References



Hervé Abdi. 2007. Z-scores. Encyclopedia ofmeasurement and statistics, 3:1055–1058. 





Rohan Anil, Andrew M. Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, and Eric Chu et al. 2023. Palm 2 technica report. Preprint, arXiv:2305.10403. 





Zefeng Cai, Xiangyu Li, Binyuan Hui, Min Yang, Bowen Li, Binhua Li, Zheng Cao, Weijie Li, Fei Huang, Luo Si, and Yongbin Li. 2022. STAR: SQL guided pre-training for context-dependent text-to-SQL parsing. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2022, pages 1235– 1247, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics. 





Bei Chen, Fengji Zhang, Anh Nguyen, Daoguang Zan, Zeqi Lin, Jian-Guang Lou, and Weizhu Chen. 2023a. Codet: Code generation with generated tests. In The Eleventh International Conference on Learning Representations. 





Yongrui Chen, Xinnan Guo, Tongtong Wu, Guilin Qi, Yang Li, and Yang Dong. 2023b. Learn from yesterday: a semi-supervised continual learning method for supervision-limited text-to-sql task streams. In Pro ceedings ofthe Thirty-Seventh AAAI Conference on Artificial Intelligence and Thirty-Fifth Conference on Innovative Applications ofArtificial Intelligence and Thirteenth Symposium on Educational Advances in Artificial Intelligence, AAAI’23/IAAI’23/EAAI’23. AAAI Press. 





Yongrui Chen, Shenyu Zhang, Guilin Qi, and Xinnan Guo. 2023c. Parameterizing context: Unleashing the power of parameter-efficient fine-tuning and incontext tuning for continual table semantic parsing. In Advances in Neural Information Processing Systems, volume 36, pages 17795–17810. Curran Associates, Inc. 





Alexandra Chronopoulou, Matthew Peters, Alexander Fraser, and Jesse Dodge. 2023. AdapterSoup: Weight averaging to improve generalization of pretrained language models. In Findings of the Associationfor Computational Linguistics: EACL 2023, pages 2054–2063, Dubrovnik, Croatia. Association for Computational Linguistics. 





Xu Han, Yi Dai, Tianyu Gao, Yankai Lin, Zhiyuan Liu, Peng Li, Maosong Sun, and Jie Zhou. 2020. Continual relation learning via episodic memory activation and reconsolidation. In Proceedings ofthe 58th Annual Meeting ofthe Associationfor Computational Linguistics, pages 6429–6440, Online. Association for Computational Linguistics. 





Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations. 





Baizhou Huang, Shuai Lu, Xiaojun Wan, and Nan Duan. 2024. Enhancing large language models in coding through multi-perspective self-consistency. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 1429–1450, Bangkok, Thailand. Association for Computational Linguistics. 





Robert A. Jacobs, Michael I. Jordan, Steven J. Nowlan, and Geoffrey E. Hinton. 1991. Adaptive mixtures of local experts. Neural Computation, 3(1):79–87. 





Diederik P. Kingma and Jimmy Ba. 2017. Adam: A method for stochastic optimization. Preprint, arXiv:1412.6980. 





Brian Lester, Rami Al-Rfou, and Noah Constant. 2021. The power of scale for parameter-efficient prompt tuning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 3045–3059, Online and Punta Cana, Dominican Republic. Association for Computational Lin guistics. 





Haoyang Li, Jing Zhang, Cuiping Li, and Hong Chen. 2023a. Resdsql: decoupling schema linking and skeleton parsing for text-to-sql. In Proceedings of the Thirty-Seventh AAAI Conference on Artificial Intelligence and Thirty-Fifth Conference on Innovative Applications ofArtificial Intelligence and Thirteenth Symposium on Educational Advances in Artificial Intelligence, AAAI’23/IAAI’23/EAAI’23. AAAI Press. 





Haoyang Li, Jing Zhang, Hanbing Liu, Ju Fan, Xiaokang Zhang, Jun Zhu, Renjie Wei, Hongyan Pan, Cuiping Li, and Hong Chen. 2024. Codes: Towards building open-source language models for text-to-sql. Proc. ACM Manag. Data, 2(3). 





Jinyang Li, Binyuan Hui, Ge Qu, Jiaxi Yang, Binhua Li, Bowen Li, Bailin Wang, Bowen Qin, Ruiying Geng, Nan Huo, Xuanhe Zhou, Ma Chenhao, Guoliang Li, Kevin Chang, Fei Huang, Reynold Cheng, and Yongbin Li. 2023b. Can llm already serve as a database interface? a big bench for largescale database grounded text-to-sqls. In Advances in Neural Information Processing Systems, volume 36, pages 42330–42357. Curran Associates, Inc. 





Zhuang Li Lizhen Ou, and Gholamreza Haffari, 2021 l ll i d i l l i h d 





for neural semantic parsers. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 3816–3831, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics. 





Huanxuan Liao, Shizhu He, Yao Xu, Yuanzhe Zhang, Yanchao Hao, Shengping Liu, Kang Liu, and Jun Zhao. 2024. From instance training to instruction learning: Task adapters generation from instructions. Preprint, arXiv:2406.12382. 





Jialin Liu, Jianhua Wu, Jie Liu, and Yutai Duan. 2024a. Learning attentional mixture of loras for language model continual learning. Preprint, arXiv:2409.19611. 





Ruiheng Liu, Jinyu Zhang, Yanqi Song, Yu Zhang, and Bailong Yang. 2024b. Filling memory gaps: Enhancing continual semantic parsing via sql syntax variance-guided llms without real data replay. Preprint, arXiv:2412.07246. 





Wenxin Mao, Ruiqi Wang, Jiyu Guo, Jichuan Zeng, Cuiyun Gao, Peiyi Han, and Chuanyi Liu. 2024. Enhancing text-to-SQL parsing through question rewriting and execution-guided refinement. In Findings of the Associationfor Computational Linguistics ACL 2024, pages 2009–2024, Bangkok, Thailand and virtual meeting. Association for Computational Linguistics. 





Martin Menabue, Emanuele Frascaroli, Matteo Boschini, Enver Sangineto, Lorenzo Bonicelli, Angelo Porrello, and Simone Calderara. 2024. Semantic residual prompts for continual learning. In Computer Vision – ECCV 2024, pages 1–18, Cham. Springer Nature Switzerland. 





OpenAI, :, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, and et al. Shyamal Anadkat. 2024. Gpt-4 technical report. Preprint, arXiv:2303.08774. 





Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Köpf, Edward Yang, Zach DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. 2019. PyTorch: an imperative style, high-performance deep learning library. Curran Associates Inc., Red Hook, NY, USA. 





Jonas Pfeiffer, Aishwarya Kamath, Andreas Rücklé, Kyunghyun Cho, and Iryna Gurevych. 2021. AdapterFusion: Non-destructive task composition for transfer learning. In Proceedings ofthe 16th Conference ofthe European Chapter ofthe Association for Computational Linguistics: Main Volume, pages 487–503, Online. Association for Computational Linguistics. 





Clifton Poth, Hannah Sterz, Indraneil Paul, Sukannya Purkayastha, Leon Engländer, Timo Imhof, Ivan 





Vulic, Sebastian Ruder, Iryna Gurevych, and Jonas´ Pfeiffer. 2023. Adapters: A unified library for parameter-efficient and modular transfer learning. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 149–160, Singapore. Association for Computational Linguistics. 





Chengwei Qin and Shafiq Joty. 2022. LFPT5: A unified framework for lifelong few-shot language learning based on prompt tuning of t5. In International Conference on Learning Representations. 





Ge Qu, Jinyang Li, Bowen Li, Bowen Qin, Nan Huo, Chenhao Ma, and Reynold Cheng. 2024. Before generation, align it! a novel and effective strategy for mitigating hallucinations in text-to-SQL generation. In Findings of the Association for Computational Linguistics ACL 2024, pages 5456–5471, Bangkok, Thailand and virtual meeting. Association for Com putational Linguistics. 





Anastasia Razdaibiedina, Yuning Mao, Rui Hou, Madian Khabsa, Mike Lewis, and Amjad Almahairi. 2023. Progressive prompts: Continual learning for language models. In The Eleventh International Con ference on Learning Representations. 





James Seale Smith, Leonid Karlinsky, Vyshnavi Gutta, Paola Cascante-Bonilla, Donghyun Kim, Assaf Arbelle, Rameswar Panda, Rogerio Feris, and Zsolt Kira. 2023. Coda-prompt: Continual decomposed attention-based prompting for rehearsal-free continual learning. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11909–11919. 





Liyuan Wang, Xingxing Zhang, Hang Su, and Jun Zhu. 2024. A comprehensive survey of continual learning: Theory, method and application. IEEE Trans. Pattern Anal. Mach. Intell., 46(8):5362–5383. 





Yue Wang, Weishi Wang, Shafiq Joty, and Steven C.H. Hoi. 2021. CodeT5: Identifier-aware unified pretrained encoder-decoder models for code understanding and generation. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 8696–8708, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics. 





Zifeng Wang, Zizhao Zhang, Chen-Yu Lee, Han Zhang, Ruoxi Sun, Xiaoqi Ren, Guolong Su, Vincent Perot, Jennifer Dy, and Tomas Pfister. 2022. Learning to prompt for continual learning. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 139–149. 





Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander Rush. 2020. Transformers: State-of-the-art natural language processing. 





In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38–45, Online. Association for Computational Linguistics. 





Siqiao Xue, Caigao Jiang, Wenhui Shi, Fangyin Cheng, Keting Chen, Hongjun Yang, Zhiping Zhang, Jianshan He, Hongyang Zhang, Ganglin Wei, Wang Zhao, Fan Zhou, Danrui Qi, Hong Yi, Shaodong Liu, and Faqiang Chen. 2024. Db-gpt: Empowering database interactions with private large language models. Preprint, arXiv:2312.17449. 





Prateek Yadav, Qing Sun, Hantian Ding, Xiaopeng Li, Dejiao Zhang, Ming Tan, Parminder Bhatia, Xiaofei Ma, Ramesh Nallapati, Murali Krishna Ramanathan, Mohit Bansal, and Bing Xiang. 2023. Exploring continual learning for code generation models. In Proceedings of the 61st Annual Meeting of the Associationfor Computational Linguistics (Volume 2: Short Papers), pages 782–792, Toronto, Canada. Association for Computational Linguistics. 





Jiaxi Yang, Binyuan Hui, Min Yang, Jian Yang, Junyang Lin, and Chang Zhou. 2024. Synthesizing text-to-SQL data from weak and strong LLMs. In Proceedings ofthe 62nd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), pages 7864–7875, Bangkok, Thailand. Association for Computational Linguistics. 





Dani Yogatama, Cyprien de Masson d’Autume, Jerome Connor, Tomas Kocisky, Mike Chrzanowski, Lingpeng Kong, Angeliki Lazaridou, Wang Ling, Lei Yu, Chris Dyer, and Phil Blunsom. 2019. Learning and evaluating general linguistic intelligence. Preprint, arXiv:1901.11373. 





Tao Yu, Chien-Sheng Wu, Xi Victoria Lin, bailin wang, Yi Chern Tan, Xinyi Yang, Dragomir Radev, richard socher, and Caiming Xiong. 2021. Grappa: Grammar-augmented pre-training for table semantic parsing. In International Conference on Learning Representations. 





Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga, Dongxu Wang, Zifan Li, James Ma, Irene Li, Qingning Yao, Shanelle Roman, Zilin Zhang, and Dragomir Radev. 2018. Spider: A large-scale human-labeled dataset for complex and cross-domain semantic parsing and text-to-SQL task. In Proceedings ofthe 2018 Conference on Empirical Methods in Natural Language Processing, pages 3911–3921, Brussels, Belgium. Association for Computational Linguistics. 





Bin Zhang, Yuxiao Ye, Guoqing Du, Xiaoru Hu, Zhishuai Li, Sun Yang, Chi Harold Liu, Rui Zhao, Ziyue Li, and Hangyu Mao. 2024a. Benchmarking the text-to-sql capability of large language models: A comprehensive evaluation. Preprint, arXiv:2403.02951. 





Xiaokang Zhang, Jing Zhang, Zeyao Ma, Yang Li, Bohan Zhang, Guanlin Li, Zijun Yao, Kangli Xu, Jinchang Zhou, Daniel Zhang-Li, Jifan Yu, Shu Zhao, 



Juanzi Li, and Jie Tang. 2024b. Tablellm: Enabling tabular data manipulation by llms in real office usage scenarios. Preprint, arXiv:2403.19318. 

Weixiang Zhao, Shilong Wang, Yulin Hu, Yanyan Zhao, Bing Qin, Xuanyu Zhang, Qing Yang, Dongliang Xu, and Wanxiang Che. 2024. SAPT: A shared attention framework for parameter-efficient continual learning of large language models. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 11641–11661, Bangkok, Thailand. Association for Computational Linguistics. 

Victor Zhong, Caiming Xiong, and Richard Socher. 2018. Seq2SQL: Generating structured queries from natural language using reinforcement learning. 

## A Method Details

## A.1 SQL Prototype Extraction

To prevent the model from being constrained by existing SQL syntax prototypes, we first utilize LLM to evolve and expand them, preparing for potential future tasks. The prompt flow is illustrated in Figure 5, where {database_schema} represents the SQL-formatted database schema information, {sql_keywords} are the given SQL syntax keywords, such as [SELECT, FROM, JOIN, ...]. This allows our method to generalize to semantic parsing tasks that use different types of SQL syntax. {existing_sql_queries} refers to the set of SQL queries obtained from clustering, denoted as $\boldsymbol { A } _ { c }$ In practice, we first randomly sample a subset of SQL queries from $\mathcal { A } _ { c }$ as input. Then, using the prompt shown in Figure 5, we generate $N _ { s }$ new SQL queries. Next, we retain only the executable SQL queries. This process is repeated $N _ { e }$ times, and finally, all the newly generated SQL queries are clustered and stripped of database schema information to obtain a new set of SQL prototypes $\mathcal { A } _ { p } ^ { \ast } .$ We set $N _ { s }$ to 20 and $N _ { e }$ to 5, and cluster centers are set to 80 for warm start and 60 for cold start. 

## A.2 Pseudo-Sample Synthesis

We use the existing SQL prototype A to guide the LLM in generating appropriate pseudo-sample pairs, with the prompt design shown in Figure 6. Specifically, the LLM is provided with an SQL prototype and database schema and instructed to generate corresponding question-SQL pairs. It is important to note that we do not directly use SQL prototypes with placeholders such as [T], [C], and [V], as we found that most open-source LLMs struggle to generate sample pairs that strictly adhere to the prototype, while some closed-source 

LLMs perform better. To accommodate different LLMs and reduce the complexity of the task, we follow previous work (Liu et al., 2024b) and simplify the prototype into a set of SQL keywords. To reduce noise in generation and improve fidelity to the SQL prototype, we introduce a reflection phase for sample correction, with the prompt shown in Figure 7. Finally, we replace the database schema in the generated SQL with placeholders, compute the edit distance from the SQL prototype, and select the Top- $N _ { g }$ most similar pseudo-sample. A total of 10 pseudo-samples are generated for each SQL prototype, and $N _ { g }$ is set to 1. 

## B Experimental Details

## B.1 Datasets

Spider-stream consists of 11 tasks, while Combined-stream contains 7 tasks, with statistics for both datasets shown in Figure 8. In the coldstart scenario, the original task order IDs we used for the two datasets are [5, 2, 9, 0, 7, 4, 10, 1, 8, 3, 6] (Spider-stream) and [3, 2, 4, 0, 6, 5, 1] (Combined-stream). It is important to note that SFNET requires additional semi-supervised data, so we followed the approach from prior work (Chen et al., 2023c) by using unused data from the corresponding databases in Spider and WikiSQL as semi-supervised data. Figure 8 also shows the amount of such data, while the other methods do not require this data. During the warm-up with LLMs phase, Table 3 lists the amount of synthesized data for our method’s initial tasks in Spiderstream and Combined-stream. 

<table><tr><td></td><td>Spider-stream</td><td>Combined-stream</td></tr><tr><td>Warm</td><td>3,393</td><td>3,430</td></tr><tr><td>Cold</td><td>1,445</td><td>1,828</td></tr></table>


Table 3: The number of pseudo samples synthesized on the initial task. Warm and Cold represent the original task order and the cold-start task order, respectively.


## B.2 Evaluation Metrics

We define $a _ { i , j } .$ , where $i , j \in [ 1 , | T | ]$ , as the test accuracy (including EM and EX) of the parser on task $\tau _ { j }$ after training on task T<sub>i</sub>. |T| denotes the total number of tasks. The four metrics are calculated as follows: 

$$
\bullet \mathrm{ACC} _ {\mathrm{a}} = \frac {1}{| \mathcal {T} |} \sum_ {i = 1} ^ {| \mathcal {T} |} a _ {| \mathcal {T} |, i}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/707ff2e9-62f8-4241-abf7-69952480efc5/56f3c6e009ed8aad4f598000e01c740008ea5350922c9103c4ea3a5c22b79876.jpg)



Figure 5: The prompt of SQL evolution.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/707ff2e9-62f8-4241-abf7-69952480efc5/2b11de916e7329083b8169ba4129adef46410994b55b3581fdd13c11dab25acc.jpg)



Figure 6: The prompt of pseudo-sample synthesis.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/707ff2e9-62f8-4241-abf7-69952480efc5/a25c11ebc21df92ee3cd82de72972085f44c64ee4ca3efd1a144cc68c4b5b5b5.jpg)



Figure 7: The prompt of self-correction.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/707ff2e9-62f8-4241-abf7-69952480efc5/d905f94cd262f1300268a6a61427ceb6e223a58c762fc4511ed73c9cdf489aa6.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/707ff2e9-62f8-4241-abf7-69952480efc5/5e116333c13f1f5f847f62697e0911c17118ac19d9a07efb470f667b9050ebf7.jpg)



Figure 8: Number statistics of Spider-stream and Combined-stream.


$\begin{array} { r } { \mathrm { A C C } _ { \mathrm { w } } = a _ { \mathcal { T } _ { \mathrm { t e s t } } ^ { ( 1 : | T | ) } } , \mathcal { T } _ { \mathrm { t e s t } } ^ { ( 1 : | T | ) } = \cup _ { i = 1 } ^ { | T | } \mathcal { T } _ { \mathrm { t e s t } } ^ { i } } \end{array}$ represents the combined test set from task $\mathcal { T } _ { 1 }$ to $\tau _ { | T | }$ 

$$
\bullet \mathrm{BWT} = \frac {1}{| \mathcal {T} | - 1} \sum_ {i = 1} ^ {| \mathcal {T} | - 1} (a _ {| \mathcal {T} |, i} - a _ {i, i})
$$

• F $\begin{array} { r } { \mathrm { W T } = \frac { 1 } { | \mathcal { T } | - 1 } \sum _ { i = 2 } ^ { | \mathcal { T } | } ( a _ { i - 1 , i } - \hat { a } _ { i } ) } \end{array}$ , where $\hat { a } _ { i }$ represents the accuracy of a randomly initialized model on task $\tau _ { i } .$ 

## B.3 Implementation Details

Our experiments are conducted using the Pytorch framework (Paszke et al., 2019) with the Adam optimizer (Kingma and Ba, 2017). Following previous work (Chen et al., 2023c), we use T5-base<sup>2</sup> and T5-large<sup>3</sup> as backbone models. For full fine-tuning, batch sizes are set to 12 and 8, with a learning rate of 1e-4. The maximum training epoch is 150, and the evaluation delay is set to 50 epochs. The input and output length limits are set to 512 and 256, respectively. In parameter-efficient fine-tuning, we experiment with two representative methods: Adapter and LoRA, both widely used for updating LLM parameters, with a learning rate of 1e-3 and an evaluation delay of 50 epochs. Our method is implemented using the Transformer (Wolf et al., 2020) library and AdapterHub<sup>4</sup> repository. 

Adapter (Poth et al., 2023) adapts to new tasks by inserting additional small networks (usually feed-forward neural networks) between the layers of the model, without modifying the existing weights. Each Adapter layer typically consists of a dimensionality reduction and restoration through linear transformations, along with a nonlinear activation function. We use Bottleneck Adapters, which introduce bottleneck feed-forward layers in each layer of the transformer model. The forward computation process is as follows: 

$$
\mathbf {h} = \mathbf {W} _ {\mathrm{up}} \cdot f (\mathbf {W} _ {\mathrm{down}} \cdot \mathbf {x}) + \mathbf {r}\tag{10}
$$

Where $\mathbf { W } _ { \mathrm { u p } }$ and $\mathbf { W } _ { \mathrm { d o w n } }$ represent the projection matrices. r denotes the residual connection. We set the reduction factor to 16, and the nonlinear activation function is ReLU. 

LoRA (Low-Rank Adaptation) (Hu et al., 2022) is an efficient fine-tuning technique that adjusts the weights of certain layers in a pre-trained model by introducing low-rank matrices. These low-rank matrices serve as modifiers to the pre-trained weights rather than replacements. For the output of a linear layer $\mathbf { h } = \mathbf { W } _ { 0 } \mathbf { x }$ , the reparameterization process is as follows: 

$$
\mathbf {h} = \mathbf {W} _ {0} \mathbf {x} + \frac {\alpha}{r} \mathbf {B A x}\tag{11}
$$

Here, $\mathbf { A } \in \mathbb { R } ^ { r \times k }$ and $\mathbf { B } \in \mathbb { R } ^ { d \times r }$ $\mathbf { W } _ { 0 }$ remains fixed during training. We set the low rank to 8. In summary, LoRA modifies existing weights, while Adapter adds new layers without altering the original weights. 

During the warm-up phase, we utilize different types and sizes of LLMs, including Mistral-7B-Instruct- $- \mathrm { v } 0 . 3 ^ { 5 }$ , Meta-Llama-3-8B-Instruct<sup>6</sup>, and Mixtral-8x7B-Instruct-v0.1<sup>7</sup>. All models were run using the $\mathrm { v L L M ^ { 8 } }$ inference framework with the fol lowing settings: temperature of 0.6, top_p of 0.95, and max_token set to 256. 

## B.4 Baselines

Following the setting of previous works (Chen et al., 2023b,c; Liu et al., 2024b), we compare our approach with three types of baseline methods. In addition to citing results from the original papers, certain baselines using different backbone models and evaluation approaches are reimplemented using their official code. Four key metrics are reported: ACC<sub>a</sub>, ACC<sub>w</sub>, BWT, and FWT, to provide deeper insights into the performance of existing methods on CSP. The details of each baseline are as follows: 

(1) Sequential Fine-Tuning (SFT) (Yogatama et al., 2019) is a naive method of sequentially finetuning tasks in a stream, typically suffering from severe catastrophic forgetting and limited generalization, representing the lower bound of continual learning performance. We use the same backbone parameter settings as our method. 

(2) Rehearsal-based Methods mitigate forgetting by repeatedly replaying real data from previous tasks during the current task. They allocate memory for each task to store past data, which is replayed in every training batch to directly enhance performance in continual learning tasks. Some approaches, such as SFNET, also incorporate manually collected semi-supervised data. While comparisons with these methods may not be entirely fair, as our approach does not require replaying historical data or using external manual data, it still shows strong competitive performance. 

EMAR (Han et al., 2020) enhances memory consolidation by combining prototype replay of both new and old relational data, allowing the model to retain prior knowledge while learning new relations, thus preventing catastrophic forgetting through memory reconsolidation. We do not directly reference the results from Chen et al. (2023c) because the original paper does not specify the task sequence order, so we reproduce the results using the official code. 

SFNET (Chen et al., 2023b) is a semisupervised continual learning method that integrates self-training (for pseudo-label generation) and episodic memory replay (to retain information from previous tasks), leveraging both supervised and unsupervised data. It employs a teacherstudent framework where the teacher optimizes the current task through self-training, and the student learns from pseudo-labeled data across all tasks to ensure overall task performance. Notably, the original SFNET uses GRAPPA (Yu et al., 2021) (encoder-only), which is pre-trained on semantic parsing, as its backbone model and evaluates a simplified custom intermediate representation. However, GRAPPA utilizes database information from the Spider dataset during pre-training, which raises concerns about potential data leakage. Moreover, its generated intermediate representation cannot be directly executed on the database, and converting it to SQL results in significant performance loss. To ensure fair comparisons with other methods, we follow previous work (Chen et al., 2023c) and use the T5 series models based on a seq2seq architecture. We carefully replace the backbone model while keeping all other parameters from the original code unchanged and report the updated results. 

(3) PET-based Methods typically require training on supervised data, and then the model’s backbone parameters are frozen. Efficient parameter tuning is then applied to learn a PET for each task in the task stream. During inference, these methods generally operate under two configurations: 

① Task-aware: This is an ideal setting that focuses solely on the CSP training phase, bypassing the task selection process during inference, thereby simplifying continual learning. Specifically, the task identifier for each sample is known during inference, so only the corresponding PET module needs to be loaded. However, this setting limits the model’s generalization ability as it cannot handle samples from unseen tasks. Such methods include: 

• PEFT (Chen et al., 2023c) is a vanilla method based on the ideal setting, where it learns a specific PET module for each task and loads it during inference. This approach makes the results independent of task order, so we report the results from the original paper. 

• C3 (Chen et al., 2023c) builds upon PEFT by introducing a teacher-student architecture and using in-context tuning to strengthen contextual information. Since the original paper does not specify the task stream order, we report the results provided in the official code repository. 

② Task-agnostic: This challenging and more realistic setting requires the model to infer without knowing the task identifier during inference. The model must autonomously select the appropriate PET module, allowing it to naturally generalize to unseen tasks. This paper focuses on this scenario. 

• Fusion (Pfeiffer et al., 2021) improves multitask learning by separating knowledge extraction (Adapter training) from knowledge combination (fusion), preventing catastrophic forgetting and task interference, while enabling effective cross-task information transfer. Specifically, it trains an additional layer on each task to fuse all previously learned Adapter modules. 

• ProgPrompt (Razdaibiedina et al., 2023) progressively accumulates learned prompts over the task stream, so the prompt length grows with the number of tasks. We set the prompt length to 50 and prompt learning rate to 0.3. 

• L2P (Wang et al., 2022) introduces a promptbased continual learning method that predefines a subset of learnable prompt parameters in a prompt pool, allowing dynamic prompt selection and task-specific learning without modifying the large model parameters. Since the original method is designed for classification tasks, we adapt it to sequence generation based on prior work (Yadav et al., 2023). The initial prompt pool size is set to 5 × the number of tasks, and the Top-1 PET module is selected for each instance. 

• Soup (Chronopoulou et al., 2023) is a simple yet effective method that improves domain adaptation by performing weight-space averaging of domain-specific adapters, achieving better performance on new domains without additional training. 

## C More Results and Analysis

## C.1 Impact of Memory Size and Ideal Settings

Table 4 presents the impact of different memory sizes and ideal settings on baseline performance. It is evident that rehearsal-based methods heavily depend on memory size. When memory size is reduced to 1, EMAR’s performance on Spider-stream even falls below that of SFT. SFNET, which is supported by replay strategies and additional unlabeled data, achieves better results. However, our method consistently outperforms without relying on any replay or external data. 

For baselines based on ideal settings, we examine their performance in real-world scenarios with unknown task identifiers by employing random loading of pre-trained PET modules in the core PEFT approach. The results show a significant decline in $\mathrm { { A C C } _ { a } }$ and $\mathsf { A C C } _ { \mathrm { w } }$ , indicating that PEFT methods experience severe catastrophic forgetting when not relying on ideal settings. Interestingly, random loading reduces task interference, resulting in better BWT and FWT performance on Spider-stream. However, on Combined-stream, the diversity of task types leads to a drop in BWT. In contrast, our method remains stable and unaffected by these external conditions. 

## C.2 Cold Start Results

Table 5 shows the performance of different methods in the cold-start scenario, where only a small amount of labeled data is available for the first task. Nearly all methods experience performance degradation, highlighting the challenge of the cold-start scenario. Our findings include: 

Dependence on Initialization for PET-based Methods. From the results of L2P, Fusion, Soup, ProgPrompt, and PEFT, we observe a sharp performance drop, even below that of SFT. This is due to the requirement for strong initial performance in these methods. 

Differences among PET Methods. In ideal settings, prompt-based methods perform worse than others, indicating greater reliance on supervised data in the initial task. This may be due to the fixed prefix design in prompt methods, which, while minimally invasive to the model, struggles with complex sequence generation tasks (Qin and Joty, 2022). The prefix also consumes input space, truncating long inputs. This issue is exacerbated in ProgPrompt, where the growing prefix severely impacts performance on later tasks. 

Impact of Replay and External Data. Fullparameter fine-tuning methods like EMAR and SFNET, which utilize historical data replay and incorporate additional semi-supervised data, perform well in preventing catastrophic forgetting and enhancing generalization. As seen in Appendix C.1, their performance is highly dependent on the memory size allocated for replay data (the more data replayed, the slower the forgetting). In contrast, PET-based methods, like ours, which do not rely on data replay, show limited effectiveness in generalization. For example, in Spider-stream, even though Soup is designed for task generalization and uses a fusion of Adapters similar to our approach, its FWT performance is only 4.5/4.8. 

<table><tr><td rowspan="2">Method</td><td rowspan="2">Memory Size</td><td colspan="4">Spider-stream</td><td colspan="4">Combined-stream</td></tr><tr><td><eq>ACC_a</eq></td><td><eq>ACC_w</eq></td><td>BWT</td><td>FWT</td><td><eq>ACC_a</eq></td><td><eq>ACC_w</eq></td><td>BWT</td><td>FWT</td></tr><tr><td colspan="10">Original Method</td></tr><tr><td>SFT</td><td>0</td><td>51.9/57.9</td><td>47.1/52.8</td><td>-19.6/-14.9</td><td>29.6/33.2</td><td>51.0/36.0</td><td>52.4/38.3</td><td>-11.1/-17.6</td><td>19.9/11.1</td></tr><tr><td colspan="10">Rehearsal-Based Method</td></tr><tr><td rowspan="3">EMAR</td><td>1</td><td>46.7/51.6</td><td>43.6/47.8</td><td>-26.7/-23.5</td><td>26.4/30.7</td><td>53.7/51.2</td><td>50.8/49.4</td><td>-15.8/-16.5</td><td>23.1/19.0</td></tr><tr><td>10</td><td>55.0/57.4</td><td>50.0/52.5</td><td>-17.8/-17.1</td><td>28.4/32.1</td><td>60.6/58.9</td><td>55.7/55.3</td><td>-7.9/-7.1</td><td>28.7/25.4</td></tr><tr><td>15</td><td>55.6/59.4</td><td>50.6/55.1</td><td>-17.5/-15.0</td><td>28.5/31.1</td><td>61.2/59.4</td><td>57.6/57.9</td><td>-7.3/-6.8</td><td>28.4/25.9</td></tr><tr><td colspan="10">Rehearsal-Based Method Requiring Unsupervised Data</td></tr><tr><td rowspan="4">SFNET</td><td>0</td><td>51.8/57.8</td><td>47.1/52.2</td><td>-18.9/-13.8</td><td>28.3/32.7</td><td>56.2/49.4</td><td>51.8/48.4</td><td>-7.9/-14.4</td><td>21.0/14.6</td></tr><tr><td>1</td><td>50.9/54.2</td><td>50.3/54.1</td><td>-20.2/-19.1</td><td>24.8/27.9</td><td>59.1/54.1</td><td>53.3/50.8</td><td>-6.3/-9.7</td><td>25.6/19.9</td></tr><tr><td>10</td><td>58.1/61.7</td><td>52.7/56.8</td><td>-12.4/-9.9</td><td>28.7/32.2</td><td>60.7/57.7</td><td>55.2/53.9</td><td>-6.0/-6.7</td><td>28.7/26.0</td></tr><tr><td>15</td><td>60.4/62.5</td><td>54.6/57.4</td><td>-11.3/-10.7</td><td>28.8/31.8</td><td>61.3/60.9</td><td>57.5/56.7</td><td>-5.4/-5.2</td><td>29.7/27.5</td></tr><tr><td colspan="10">Random PET</td></tr><tr><td><eq>\text{RandomAdapter}</eq></td><td>0</td><td>37.8/40.1</td><td>42.6/45.5</td><td>4.8/4.2</td><td>25.5/28.2</td><td>21.8/17.1</td><td>28.1/26.3</td><td>-17.4/-20.3</td><td>25.2/21.0</td></tr><tr><td><eq>\text{RandomLoRA}</eq></td><td>0</td><td>36.9/40.2</td><td>42.3/45.8</td><td>2.6/2.6</td><td>25.9/29.1</td><td>30.5/25.0</td><td>35.1/32.4</td><td>-8.0/-10.8</td><td>32.8/30.6</td></tr><tr><td colspan="10">Without Rehearsal and Unsupervised Data</td></tr><tr><td><eq>\text{APEXAdapter}</eq></td><td>0</td><td>61.0/62.4</td><td>63.4/64.8</td><td>1.3/1.4</td><td>25.1/28.3</td><td>62.7/59.3</td><td>64.9/63.1</td><td>6.8/9.2</td><td>21.9/20.7</td></tr><tr><td><eq>\text{APEXLoRA}</eq></td><td>0</td><td>63.3/65.3</td><td>62.1/63.9</td><td>0.4/0.5</td><td>24.4/27.5</td><td>63.5/61.4</td><td>64.3/63.5</td><td>7.2/10.6</td><td>29.2/25.8</td></tr><tr><td colspan="10">Upper Bound</td></tr><tr><td>ORACLE</td><td></td><td>75.6/77.0</td><td>76.4/78.4</td><td>4.2/3.3</td><td>29.7/32.0</td><td>70.2/68.2</td><td>71.1/70.8</td><td>5.6/4.7</td><td>29.2/26.6</td></tr></table>


Table 4: Effect of memory size and ideal settings on baseline performance (T5-base).


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/707ff2e9-62f8-4241-abf7-69952480efc5/21e93f8d7fe0624a9750a823a8869cce86e35d76a5024dbadd3288c10beb7f73.jpg)



(a) ACC


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/707ff2e9-62f8-4241-abf7-69952480efc5/b9b003d6817f855456141ffa3e2e1a45c907d25d47a1d2e4b3762ccfc35c257d.jpg)



(b) ACC


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/707ff2e9-62f8-4241-abf7-69952480efc5/f4169dc89d81c86634f04ed835e2f716d50f28e2ad21b2555437a93695629c53.jpg)



(c) BWT


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/707ff2e9-62f8-4241-abf7-69952480efc5/5063d0458e1641f87e6d6e3e31e613b3ef505825dd0202ffa8deb1bf6786319c.jpg)



(d) FWT



Figure 9: Comparison of results (EM) across all seen tasks on Combined-stream (T5-base).


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/707ff2e9-62f8-4241-abf7-69952480efc5/bcc7ca12450a1f24e4ccb4423b343706905da380ce2a40b668bc9c28032a79bb.jpg)



(a) ACC<sub>a</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/707ff2e9-62f8-4241-abf7-69952480efc5/e49df3a0bd24ed755f41884487a4fa42377392607bc0e93710b7c0534bcb9623.jpg)



(b) ACC<sub>w</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/707ff2e9-62f8-4241-abf7-69952480efc5/205fe4d00a738a6276591c3a2b2c8644bc6cf457bfed7da780d758e19729f815.jpg)



(c) BWT


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/707ff2e9-62f8-4241-abf7-69952480efc5/2ba8ee4b70d8b214302eab730b0fa1678f94597dcc0db8850193a19e46a62855.jpg)



(d) FWT



Figure 10: Comparison of results (EM) across all seen tasks on Spider-stream (T5-large).


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/707ff2e9-62f8-4241-abf7-69952480efc5/e8af37fee77d75021256637041c6c22757c6c2e71f184e8ac98fd41c0ec911ad.jpg)



(a) ACC<sub>a</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/707ff2e9-62f8-4241-abf7-69952480efc5/87083053200628e69bdcfb879212fa03347c7aaf89c57e9cc455f7c096044b44.jpg)



(b) ACC<sub>w</sub>


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/707ff2e9-62f8-4241-abf7-69952480efc5/e96331d33ef886e1af27f1d9daf54cdeb8f6fb9b802c1f043ac158e6ddf11b43.jpg)



(c) BWT


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/707ff2e9-62f8-4241-abf7-69952480efc5/67c808b2b374837107988b4aea534b9e71cf8c28c754c86099e8d739be8d348d.jpg)



(d) FWT



Figure 11: Comparison of results (EM) across all seen tasks on Combined-stream (T5-large).


<table><tr><td rowspan="2">Method</td><td rowspan="2">DR</td><td rowspan="2">PET</td><td colspan="4">Spider-stream (cold start)</td><td colspan="4">Combined-stream (cold start)</td></tr><tr><td><eq>ACC_a</eq></td><td><eq>ACC_w</eq></td><td>BWT</td><td>FWT</td><td><eq>ACC_a</eq></td><td><eq>ACC_w</eq></td><td>BWT</td><td>FWT</td></tr><tr><td colspan="11">T5-Base(220M)</td></tr><tr><td rowspan="3">SFT</td><td>X</td><td>Full</td><td>49.6/55.4</td><td>45.7/52.0</td><td>-20.8/-16.3</td><td>26.7/27.8</td><td>39.9/36.4</td><td>32.0/29.4</td><td>-25.9/-28.5</td><td>24.5/20.5</td></tr><tr><td>X</td><td>Adapter</td><td>40.4/44.9</td><td>35.6/39.8</td><td>-33.2/-29.4</td><td>23.0/23.6</td><td>28.9/25.6</td><td>18.5/16.4</td><td>-40.1/-43.2</td><td>23.4/19.4</td></tr><tr><td>X</td><td>LoRA</td><td>36.4/39.6</td><td>32.6/36.0</td><td>-30.2/-28.8</td><td>21.4/22.9</td><td>27.7/25.7</td><td>17.0/15.8</td><td>-35.6/-36.3</td><td>20.6/17.4</td></tr><tr><td rowspan="2">L2P</td><td>X</td><td>Adapter</td><td>25.7/28.4</td><td>21.4/23.7</td><td>-41.0/-39.9</td><td>15.9/16.4</td><td>20.1/16.5</td><td>12.3/10.1</td><td>-47.5/-49.6</td><td>15.0/12.8</td></tr><tr><td>X</td><td>LoRA</td><td>29.6/33.5</td><td>26.0/30.5</td><td>-39.2/-37.4</td><td>18.8/19.7</td><td>27.1/24.0</td><td>16.7/14.8</td><td>-40.7/-42.6</td><td>20.7/16.8</td></tr><tr><td>Fusion</td><td>X</td><td>Adapter</td><td>25.7/30.1</td><td>23.9/28.3</td><td>-14.1/-13.3</td><td>23.4/24.7</td><td>22.0/20.3</td><td>13.5/12.4</td><td>-28.1/-29.4</td><td>18.1/15.3</td></tr><tr><td>Soup</td><td>X</td><td>Adapter</td><td>9.0/10.2</td><td>8.1/9.7</td><td>-2.5/-2.5</td><td>4.5/4.8</td><td>22.5/19.8</td><td>13.7/12.0</td><td>-0.3/0.3</td><td>17.9/15.6</td></tr><tr><td>ProgPrompt</td><td>X</td><td>Prompt</td><td>8.7/11.0</td><td>7.1/9.4</td><td>-34.7/-35.3</td><td>8.4/8.5</td><td>0.2/0.6</td><td>0.1/0.3</td><td>-10.2/-11.2</td><td>0.1/0.0</td></tr><tr><td>EMAR</td><td>√</td><td>Full</td><td>57.2/61.4</td><td>53.2/58.1</td><td>-15.6/-12.2</td><td>27.4/28.0</td><td>54.4/51.5</td><td>50.1/50.0</td><td>-10.0/-12.1</td><td>17.8/15.9</td></tr><tr><td>SFNET♣</td><td>X</td><td>Full</td><td>50.1/55.3</td><td>46.4/51.9</td><td>-21.0/-17.9</td><td>25.9/27.5</td><td>34.5/31.9</td><td>24.4/23.3</td><td>-33.7/-34.6</td><td>22.5/19.2</td></tr><tr><td>SFNET♣</td><td>√</td><td>Full</td><td>55.4/59.3</td><td>51.3/55.5</td><td>-13.4/-11.4</td><td>24.7/25.6</td><td>58.4/56.7</td><td>55.9/56.3</td><td>-6.8/-7.0</td><td>24.2/20.0</td></tr><tr><td rowspan="2">APEX(Ours)</td><td>X</td><td>Adapter</td><td>54.1/55.6</td><td>53.5/55.5</td><td>2.9/2.7</td><td>9.1/9.4</td><td>58.2/55.5</td><td>57.7/57.1</td><td>7.6/9.3</td><td>26.3/24.3</td></tr><tr><td>X</td><td>LoRA</td><td>57.2/59.3</td><td>57.8/60.1</td><td>3.5/3.7</td><td>13.0/13.3</td><td>57.3/55.3</td><td>56.1/55.7</td><td>2.5/2.2</td><td>27.6/23.4</td></tr><tr><td colspan="11">Ideal Setting</td></tr><tr><td rowspan="3">PEFT◇</td><td>X</td><td>Prompt</td><td>11.5/14.5</td><td>14.9/18.6</td><td>0.0/0.0</td><td>N/A</td><td>25.0/23.1</td><td>16.6/18.2</td><td>0.0/0.0</td><td>N/A</td></tr><tr><td>X</td><td>Adapter</td><td>69.6/71.0</td><td>70.1/72.1</td><td>0.0/0.0</td><td>N/A</td><td>61.7/62.0</td><td>64.3/65.7</td><td>0.0/0.0</td><td>N/A</td></tr><tr><td>X</td><td>LoRA</td><td>63.6/65.8</td><td>64.7/67.4</td><td>0.0/0.0</td><td>N/A</td><td>55.3/55.0</td><td>58.4/59.0</td><td>0.0/0.0</td><td>N/A</td></tr><tr><td colspan="11">T5-Large(770M)</td></tr><tr><td rowspan="3">SFT</td><td>X</td><td>Full</td><td>60.9/67.7</td><td>56.9/63.5</td><td>-16.8/-11.0</td><td>34.6/36.3</td><td>49.5/47.2</td><td>44.3/41.7</td><td>-22.1/-23.0</td><td>32.6/27.0</td></tr><tr><td>X</td><td>Adapter</td><td>48.2/53.2</td><td>44.2/48.9</td><td>-28.7/-24.9</td><td>30.5/31.4</td><td>32.5/30.7</td><td>22.6/21.4</td><td>-40.4/-42.0</td><td>29.4/23.9</td></tr><tr><td>X</td><td>LoRA</td><td>48.0/53.7</td><td>43.8/49.0</td><td>-27.3/-22.4</td><td>32.1/33.5</td><td>30.1/27.8</td><td>18.5/17.1</td><td>-41.2/-43.4</td><td>28.3/23.4</td></tr><tr><td>ProgPrompt</td><td>X</td><td>Prompt</td><td>13.9/16.5</td><td>11.5/13.8</td><td>-38.3/-37.1</td><td>10.2/10.7</td><td>1.1/1.5</td><td>1.4/1.5</td><td>-13.5/-14.1</td><td>0.3/0.2</td></tr><tr><td>EMAR</td><td>√</td><td>Full</td><td>61.2/65.3</td><td>59.0/63.2</td><td>-14.1/-11.7</td><td>32.6/33.3</td><td>55.0/51.5</td><td>53.0/50.4</td><td>-11.9/-14.4</td><td>25.2/19.6</td></tr><tr><td>SFNET♣</td><td>X</td><td>Full</td><td>56.4/61.4</td><td>51.5/56.0</td><td>-18.3/-14.7</td><td>31.2/32.1</td><td>40.9/37.0</td><td>32.6/29.7</td><td>-29.9/-33.5</td><td>28.8/25.0</td></tr><tr><td>SFNET♣</td><td>√</td><td>Full</td><td>63.4/67.3</td><td>58.6/62.6</td><td>-10.6/-8.8</td><td>32.8/34.4</td><td>62.2/61.9</td><td>59.8/60.6</td><td>-6.5/-5.1</td><td>31.3/27.2</td></tr><tr><td rowspan="2">APEX(Ours)</td><td>X</td><td>Adapter</td><td>58.9/59.7</td><td>60.3/61.9</td><td>4.7/5.0</td><td>16.6/17.8</td><td>61.9/57.4</td><td>62.9/61.1</td><td>-0.1/-0.1</td><td>27.6/25.3</td></tr><tr><td>X</td><td>LoRA</td><td>61.2/62.9</td><td>61.4/63.6</td><td>2.5/2.5</td><td>18.1/18.3</td><td>61.4/60.8</td><td>58.5/58.9</td><td>4.6/4.4</td><td>35.2/32.7</td></tr><tr><td colspan="11">Ideal Setting</td></tr><tr><td rowspan="3">PEFT◇</td><td>X</td><td>Prompt</td><td>42.6/46.2</td><td>51.7/55.1</td><td>0.0/0.0</td><td>N/A</td><td>43.9/45.5</td><td>38.2/40.8</td><td>0.0/0.0</td><td>N/A</td></tr><tr><td>X</td><td>Adapter</td><td>74.1/75.6</td><td>74.7/76.6</td><td>0.0/0.0</td><td>N/A</td><td>65.4/65.9</td><td>70.1/71.2</td><td>0.0/0.0</td><td>N/A</td></tr><tr><td>X</td><td>LoRA</td><td>72.1/73.4</td><td>72.3/74.2</td><td>0.0/0.0</td><td>N/A</td><td>63.3/63.8</td><td>66.7/68.5</td><td>0.0/0.0</td><td>N/A</td></tr></table>


Table 5: Results on Spider-stream and Combined-stream datasets under cold start settings (%). DR indicates whether historical data replay is used, with a memory size set to 15. PET refers to different training architectures, with FULL being full-parameter tuning. ♣ indicates the use of additional unsupervised data, and ♢ means a known task ID, but forward transfer is impossible. The best results are highlighted in bold, and the second-best results are underlined. Our results are from three random runs.


Robustness and Effectiveness of Our Method. APEX achieves consistent performance without requiring historical data replay or external semisupervised data. It significantly outperforms PETbased baselines and rivals or surpasses replaybased methods, with a 16.9% improvement over SFNET in BWT, further proving its effectiveness. 

## C.3 Results Till the Seen Tasks

Figures 9 to 11 provide further comparison results on previously seen tasks. It can be observed that our method shows significant advantages across most metrics, even surpassing ORACLE performance in some cases. We also notice that methods based on full-parameter fine-tuning, such as SFT-Full, EMAR, and SFNET, generally perform better on FWT compared to PET-based methods, indicating that a larger parameter update range helps the model generalize to new tasks more effectively. 

## C.4 Impact of Threshold γ

To further explore the impact of threshold γ, we evaluate sample visibility prediction accuracy using known task identifiers in the test set across different threshold values, as shown in Figure 12. The results indicate that as the threshold increases, the accuracy for seen and unseen samples becomes inversely related. The threshold of 1.6, used in Section 6.2, is located near the intersection of the two curves on the x-axis, which further demonstrates that balancing the prediction accuracy for seen and unseen samples is key to balancing the model’s 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/707ff2e9-62f8-4241-abf7-69952480efc5/11963c38e2254de212b04f242019193bf2d38aa938dc68d15270c5abf905b57c.jpg)



Figure 12: Prediction accuracy of sample history visibility (seen/unseen) under different γ settings.



resistance to forgetting and its generalization.


## C.5 Warm Up with Different LLMs

To further explore the impact of different types and scales of LLM warm-up on model performance, in addition to the previously mentioned Mixtral-8x7B-Instruct-v0.1, we also used Mistral-7B-Instruct-v0.3 and Meta-Llama-3-8B-Instruct. Table 6 presents the results of our method under different LLM warm-up settings, as well as the performance of LLMs in zero-shot inference. It is evident that the zero-shot results of LLMs alone are not satisfactory, further highlighting the importance of continual learning. Notably, our method, when combined with various types and scales of LLMs, continues to demonstrate strong performance, especially in the BWT metric, consistently exceeding the upper performance bound, which further validates the effectiveness of our approach. 

<table><tr><td>Model</td><td><eq>\text{ACC}_{\text{a}}</eq></td><td><eq>\text{ACC}_{\text{w}}</eq></td><td>BWT</td><td>FWT</td></tr><tr><td>SFT-Full</td><td>51.0/36.0</td><td>52.4/38.3</td><td>-11.1/-17.6</td><td>19.9/11.1</td></tr><tr><td>SFT-LoRA</td><td>46.9/40.6</td><td>48.6/45.1</td><td>-21.0/-26.3</td><td>16.8/11.1</td></tr><tr><td>Mistral-7B</td><td>27.3/22.1</td><td>24.9/23.2</td><td>-/-</td><td>-/-</td></tr><tr><td>+APEX</td><td>64.3/62.8</td><td>66.7/66.9</td><td>8.1/9.6</td><td>23.4/22.1</td></tr><tr><td>Llama-3-8B</td><td>33.5/26.2</td><td>27.7/24.9</td><td>-/-</td><td>-/-</td></tr><tr><td>+APEX</td><td>64.9/63.1</td><td>66.9/65.9</td><td>7.8/9.5</td><td>24.0/21.5</td></tr><tr><td>Mixtral-8x7B</td><td>28.6/24.4</td><td>27.0/26.0</td><td>-/-</td><td>-/-</td></tr><tr><td>+APEX</td><td>63.5/61.4</td><td>64.3/63.5</td><td>7.2/10.6</td><td>29.2/25.8</td></tr><tr><td colspan="5">Ideal Setting &amp; ORACLE</td></tr><tr><td>PEFT-LoRA</td><td>63.1/61.4</td><td>67.0/66.7</td><td>0.0/0.0</td><td>N/A</td></tr><tr><td>ORACLE</td><td>70.2/68.2</td><td>71.1/70.8</td><td>5.6/4.7</td><td>29.2/26.6</td></tr></table>


Table 6: Comparison of LLM zero-shot results and the performance of different LLMs warming up APEX on Combined-stream (T5-base).
