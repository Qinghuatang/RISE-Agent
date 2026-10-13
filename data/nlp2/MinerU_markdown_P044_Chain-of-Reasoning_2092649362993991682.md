# Chain-of-Reasoning: Towards Unified Mathematical Reasoning in Large Language Models via a Multi-Paradigm Perspective

Yiyao Yu<sup>1</sup>* Yuxiang Zhang<sup>1</sup>* Dongdong Zhang<sup>2†</sup> Xiao Liang<sup>1</sup> Hengyuan Zhang<sup>1</sup> Xingxing Zhang<sup>2</sup> Mahmoud Khademi<sup>2</sup> Hany Awadalla<sup>2</sup> Junjie Wang<sup>1†</sup> Yujiu Yang<sup>1†</sup> Furu Wei<sup>2</sup> 

<sup>1</sup>Tsinghua University <sup>2</sup>Microsoft 

yuyy23@mails.tsinghua.edu.cn, joell070408@gmail.com, wangjunjie@sz.tsinghua.edu.cn {dozhang,xizhang,mkhademi,hanyh,fuwei}@microsoft.com 

## Abstract

Large Language Models (LLMs) have made notable progress in mathematical reasoning, yet they often rely on single-paradigm reasoning that limits their effectiveness across diverse tasks. In this paper, we introduce Chain-of-Reasoning (CoR), a novel unified framework that integrates multiple reasoning paradigms — Natural Language Reasoning (NLR), Algorithmic Reasoning (AR), and Symbolic Reasoning (SR) — to enable synergistic collaboration. CoR generates multiple potential answers using different reasoning paradigms and synthesizes them into a coherent final solution. We propose a Progressive Paradigm Training (PPT) strategy that allows models to progressively master these paradigms, culminating in the development of CoR-Math-7B. Experimental results demonstrate that CoR-Math-7B significantly outperforms current SOTA models, achieving up to a 41.0% absolute improvement over GPT-4o in theorem proving tasks and a 15% improvement over RL-based methods on the MATH benchmark in arithmetic tasks. These results show the enhanced mathematical comprehensive ability of our model, enabling zero-shot generalization across tasks. The code is available at https://github. com/microsoft/CoR. 

## 1 Introduction

While LLMs have shown strong performance in solving mathematical tasks (Feigenbaum et al., 1963; Hosseini et al., 2014), advanced opensource reasoners still struggle with solving comprehensive mathematical problems, including both arithmetic computation and theorem proving. 

Existing works (Xin et al., 2024; Yang et al., 2024; Wu et al., 2024; Zhang et al., 2024) are often trained on specific tasks, aiming to enhance their ability to independently derive answers based on specific structured knowledge representation. This representation is known as the reasoning paradigm, involving Natural Language Reasoning (NLR), Algorithmic Reasoning (AR), and Symbolic Reasoning (SR), as depicted in Fig. 2 (a). Specifically, NLR uses natural language for reasoning via common sense and semantic context, providing explicit step-by-step explanations (Wei et al., 2022). AR leverages code to emphasize computational operations and execution processes, such as generating Python code for execution (Chen et al., 2023; Gao et al., 2023). SR uses logical symbols and axiomatic systems for formalized reasoning, with recent methods (Xin et al., 2024; Huang et al., 2024; Wu et al., 2024) considering numerous symbolic trajectories via tree-based search for theorem proving. However, these methods mainly focus on optimizing singleparadigm reasoning, creating models that demonstrate asymmetrical performance across different mathematical tasks. For instance, models specialized in NLR may exhibit deficiencies in theorem proving and vice versa. This fragmented paradigm constrains the upper-bound performance on individual tasks and undermines the model’s capacity for cross-paradigm generalization. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/db771bc9-d734-41ba-ac35-ccfc42738592/0ad836dfc0cfe9de392166fbe156da44e433e5cf7f5398de0c9e383e37266ffa.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/db771bc9-d734-41ba-ac35-ccfc42738592/6bd783052bcfc1996d7c5cf5fadbea8f9d6cf0c953df54e03006ceabb9879d53.jpg)



(b) Arithmetic Computation (zero-shot)



Figure 1: A comprehensive comparative analysis of CoR-Math-7B and baseline models across mathematical tasks. (a) shows the effectiveness of CoR-Math-7B (zero-shot) in theorem proving tasks. (b) shows a resource-efficiency analysis for arithmetic computation tasks, where CoR-Math-7B achieves optimal resource efficiency and near-optimal zero-shot performance.


Researchers have explored various strategies to tackle these challenges. To improve the single-task performance, some works integrate tools to overcome the limitations of single-paradigm reasoning (Gou et al., 2024; LI et al., 2024), as shown in Fig. 2 (b). While recognizing the benefits of combining reasoning paradigms, they still rely on a single paradigm, neglecting the possibility that the second paradigm could independently complete the reasoning, thereby constraining overall potential. Besides, to improve cross-task generalization, several studies (Shao et al., 2024; Huang et al., 2024) incorporate diverse task samples into large-scale datasets, such as those drawn from theorem proving tasks that focus exclusively on SR solutions, or from arithmetic problems that emphasize NLR solutions. Although models trained on such data are capable of cross-task reasoning, they still rely on demonstrations for effective transfer. 

To address these limitations, we propose Chainof-Reasoning (CoR), a unified framework integrating NLR, AR, and SR to produce synergistic benefits. As shown in Fig. 2 (c), CoR enables multiparadigm reasoning by applying different reasoning paradigms to derive multiple potential answers, which are then summarized into a final solution. This framework allows iterative reasoning across paradigms, leveraging prior results to enhance single-task performance. Moreover, CoR unifies multi-paradigm reasoning across tasks, enabling zero-shot generalization. Adjusting prompts to control reasoning depth improves adaptability to diverse tasks. As a result, we introduce Multi-Paradigm Mathematical (MPM), a dataset comprising 167k reasoning paths, and propose Progressive Paradigm Training (PPT), a method enabling models to progressively master multiple reasoning paradigms, leading to CoR-Math-7B. 

We evaluate CoR-Math-7B on five challenging mathematical reasoning benchmarks, covering both arithmetic computation and theorem proving. Our results show that LLMs equipped with CoR significantly surpass current SOTA baselines. In theorem proving (Fig. 1 (a)), CoR-Math-7B improves zero-shot performance over GPT-4o (Hurst et al., 2024) by 41.0%, outperforming all fewshot reasoners. For arithmetic tasks, CoR-Math-7B achieves a 24.2% absolute improvement over GPT-4 on MATH. Furthermore, as shown in Fig. 1 (b), CoR-Math-7B efficiently utilizes resources to surpass the optimal performance curve of singleparadigm approaches. Unlike mainstream methods that conduct extensive searches within a single paradigm, CoR enhances test-time inference in multiple paradigms. These results show that CoR can solve comprehensive mathematical problems through multi-paradigm reasoning, requiring less training data and lowering reasoning costs. 

## 2 Related Work

Reasoning Paradigms in LLMs. Recent advancements in LLMs focus on single-paradigm reasoning, with each paradigm representing a distinct method for knowledge representation and logical inference. NLR uses human-readable language for commonsense reasoning and step-by-step deductions (Wei et al., 2022; Yao et al., 2023; Zhou et al., 2023; Besta et al., 2024; Sel et al., 2024), AR generates executable code for precise calculations (Chen et al., 2023; Rozière et al., 2023), and SR utilizes logical symbols for theorem proving (Xin et al., 2024). Some methods enhance reasoning by integrating external tools like calculators or code interpreters, yet they remain within a single paradigm (Gou et al., 2024). While effective in specific tasks, they struggle with crossdomain generalization and dynamic environments. To address these limitations, the CoR framework enables collaboration across reasoning paradigms, enabling zero-shot multitask generalization. 

Mathematical Problem Solving with LLMs. Advanced research highlights the potential of LLMs in solving mathematical problems (Zhu et al., 2023; Lin et al., 2024b; Luo et al., 2025). Several studies have developed unified solvers by synthesizing mathematical data to address challenges like arithmetic computation and theorem proving (Huang et al., 2024; Shao et al., 2024). However, most approaches focus on optimizing a single paradigm. For instance, in arithmetic computation, tools are integrated to assist natural language reasoning. (Gou et al., 2024). In theorem proving, models rely on specialized data like pretraining data and Reinforcement Learning (RL) reward data, and tree-based search methods to generate numerous possible solutions (Ying et al., 2024b; Xin et al., 2024). These approaches either neglect complete reasoning or depend on largescale search within the solution space, limiting performance gains within a single paradigm. To address these, we introduce the CoR-Math-7B model with complete reasoning processes to explore an expanded multi-paradigm solution space. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/db771bc9-d734-41ba-ac35-ccfc42738592/275ef57a08487919605eda462e53770ba659214c8407c7d1d14bf10ec0ed2667.jpg)



Figure 2: The reasoning process under different paradigms: (a) In single-paradigm reasoning, each reasoning step relies on the same knowledge medium, such as Natural Language (NL), algorithms, or symbols. (b) In tool integrated single-paradigm, NL is used for reasoning, while code assists in solving specific sub-problems. After obtaining the execution results, the reasoning continues using NL. (c) The proposed CoR reasoning framework, along with several examples, shows that multi-paradigm reasoning allows for varying reasoning depths to address different types of problems.


## 3 Chain-of-Reasoning Framework

## 3.1 Overview

CoR aims to enable LLMs to perform a series of multi-paradigm reasoning on any type of mathematical problem, ultimately arriving at a solution. Specifically, given a mathematical problem x, LLMs (P) can infer the result y by following multiple reasoning paradigms, where each reasoning paradigm τ includes n reasoning paths $\{ r p _ { 1 } , . . . , r p _ { n } \}$ . We represent this single-paradigm scenario as $y \sim \mathbb { P } ( y | x , \tau )$ . To simplify the process for each reasoning paradigm, we set n = 1 as default (Details in Appendix A). 

Inspired by recent works (Wei et al., 2022) in encouraging step-by-step reasoning, we introduce CoR, which extends from a single paradigm to three paradigms $\Gamma ~ = ~ \left( \tau _ { 1 } , \tau _ { 2 } , \tau _ { 3 } \right)$ . For generating multiple chained reasoning paradigms for a given problem, CoR follows the steps outlined below. The reasoning process begins with the problem x and the first reasoning paradigm τ<sub>1</sub>. Subsequently, each paradigm τ in the sequence Γ is generated based on the problem x and the previously established paradigms $\tau _ { 1 } , . . . , \tau _ { i - 1 }$ , as represented by $\tau _ { i } \sim \mathbb { P } ( \tau _ { i } | x , \tau _ { 1 } , . . . , \tau _ { i - 1 } )$ . Finally, the outcomes of all the reasoning paradigms are aggregated to derive the final result y, expressed as $y \sim \mathbb { P } ( y | x , \tau _ { \mathrm { N L R } } , \tau _ { \mathrm { S R } } , \tau _ { \mathrm { A R } } )$ ), considering three paradigms: NLR, SR, and AR. In detail, τ<sub>SR</sub> and τ include both the complete reasoning paths and the interaction processes with tool outputs. In this study, τ<sub>SR</sub> uses the Lean prover, and τ<sub>AR</sub> applies a Python compiler to obtain reasoning results. 

In our workflow (Fig. 3), the training pipeline includes: (a) collecting the MPM dataset with deep reasoning paths; (b) using progressive paradigm training to enhance reasoning across paradigms; and (c) applying the trained LLMs for zero-shot inference with sequential multi-paradigm sampling to explore diverse solutions. 

## 3.2 Collecting Dataset

To train CoR model, we extend the singleparadigm datasets to incorporate multiple reasoning paradigms, denoted $\tt a s < x , N L R , S R , A R , y >$ As presented in Fig. 3 (a), the training data collection process involves two stages: (a.1) Reconstructing and Extending, and (a.2) Revising. In the first stage, we reconstruct high-quality opensource mathematical data as seed samples and synthesize additional reasoning paradigms to form the MPM-raw dataset. In the second stage, a theorem prover examines MPM-raw samples, and a mathematical reasoner revises those that fail, compiling all completed data into the final MPM dataset. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/db771bc9-d734-41ba-ac35-ccfc42738592/65c1ce42b5d333af3ad1b1bd384ce74e7a88e2fd44e42301e62c63c6c58bd1bc.jpg)



Figure 3: An overview of (a) the Multi-Paradigm Math (MPM) dataset construction process, involving reconstruction, extension, and theorem prover verification, and (b) the Progressive Paradigm Training (PPT) method, where the model is trained with increasing reasoning paradigms in stages.


Stage 1: Reconstructing and Extending. We introduce a universal text template for multiparadigm reasoning (details in Appendix B.1), which standardizes the placement and relationships of reasoning paradigms. It supports various reasoning depths and flexible combinations of different reasoning paradigms. As shown in Fig. 2 (c), Problem 1 shows an instance incorporating both NLR and SR paradigms, and Problem 2 integrates three reasoning paradigms. To ensure data integrity and prevent potential biases, we preprocess the Numina-TIR (LI et al., 2024) and Lean-Workbook (Ying et al., 2024a) datasets in two steps. First, samples without corresponding solutions are removed. Second, we further recon struct and extend these datasets by leveraging powerful LLMs , such as GPT-4o (Hurst et al., 2024). These models generate missing reasoning paradigms $\tau _ { g }$ and refine existing ones $\tau ^ { \prime } { \mathrm { . } }$ , ensuring comprehensive coverage and logical consistency. To effectively guide the processes of augmentation and refinement, we develop tailored prompts p for each seed dataset (examples in Appendix B.2). 

The procedure can be described in: 

$$
\tau_ {g} \sim \mathbb {P} _ {\mathcal {G}} (\tau_ {g} \mid p _ {s} \oplus x \oplus y \oplus \tau^ {\prime}), \tau^ {\prime} \in \{\tau_ {\mathrm{NLR}}, \tau_ {\mathrm{SR}}, \tau_ {\mathrm{AR}} \},\tag{1}
$$

where means concatenation. After that, we conduct a manual review of all samples. Given that alternative approaches are readily verified through external tools, we focus on the accuracy of the NLR and AR. This approach considerably lowers the requisite skill level of annotators. To prevent data leakage, we compute the Levenshtein distance (Miller et al., 2009) between training and test problems, as detailed in Appendix C.1. As a result, this phase yields the MPM-raw dataset, comprising approximately 285, 000 synthetic samples. Stage 2: Revising. The MPM-raw dataset interacts with the Lean prover to verify proof steps, guiding the filtering and modification of reasoning paths. The proof $\tau _ { \mathrm { S R } }$ is submitted to the prover, and if verification is successful without errors, the entire multi-paradigm reasoning path is collected to the MPM dataset. Otherwise, the error information ε returned by the prover is fed into a revising model $\mathbb { P } _ { R } .$ Furthermore, this model generates a revised proof $\tilde { \tau } _ {  { \mathrm { S R } } }$ based on a prompt $p _ { \varepsilon }$ . The relationship can be expressed as: 

$$
\tilde {\tau} _ {\mathrm{SR}} \sim \mathbb {P} _ {R} (\tilde {\tau} _ {\mathrm{SR}} \mid p _ {\varepsilon} \oplus x \oplus y \oplus \tau_ {\mathrm{SR}}),\tag{2}
$$

where the revised proof $\tilde { \tau } _ {  { \mathrm { S R } } }$ is then resubmitted to the Lean prover for verification. This iterative process continues for up to 64 iterations or until $\tilde { \tau } _ {  { \mathrm { S R } } }$ is verified as correct. In detail, we employ DeepSeek-Prover-V1.5 (Xin et al., 2024) as the $\mathbb { P } _ { R }$ 

Consequently, the MPM dataset comprises 82, 770 problems and 167, 412 multi-paradigm 

reasoning solutions. 

## 3.3 Training

Inspired by recent advances (LI et al., 2024), we introduce the Progressive Paradigm Training (PPT) strategy, enabling LLMs to gradually master diverse reasoning paradigms. As shown in Fig. 3 (b), this framework expands the model’s reasoning abilities by sequentially introducing different paradigms at each stage. Each training stage uses a different combination of reasoning paradigms. In stage ➀, given the dominance of NL in the pretraining data of language models (Dubey et al., 2024), we create Numina-CoT∗ as an initialized teaching stage. This is done by modifying the original Numina-CoT dataset (LI et al., 2024) according to our universal text template, enabling the model to learn to use NL to solve complex mathematical problems. Based on the question $x ,$ the model performs reasoning τ<sub>NLR</sub> and generates the answer $y .$ The generated sequence is $z = [ x ] \tau _ { \mathrm { N L R } } y$ , where [x] represents the inputs after tokenization. For simplicity, we first consider the loss function for a single sample: 

$$
\mathcal {L} _ {\text { sample }} = - \sum_ {t = 1} ^ {| z |} \log \mathbb {P} _ {\theta} (z _ {t} \mid z _ {<   t}),\tag{3}
$$

where θ represents the model parameters, $z _ { t }$ is the t-th token in the generated sequence, and $z _ { < t }$ indicates all tokens before the t-th token in the generated sequence. In stage ➁, considering a certain proportion of code corpora in the pre-training data, we expand the training dataset to include two paradigms: NLR and AR. Similar to Numina-CoT∗, we modify the original Numina-TIR to create Numina-TIR∗. With this modified dataset, the generated sequence is $z = [ x ] \tau _ { \mathrm { N L R } } \tau _ { \mathrm { A R } } y$ . After this stage, the model can handle problems that require precise answers. In stage ➂, we further expand the training data to three reasoning paradigms by utilizing the MPM dataset, where z is [x]τ τ τ y. After full stages, the trained CoR-Math-7B model not only masters NLR and AR but also can perform rigorous logical SR. 

Unlike traditional curriculum and incremental learning, which sequencing tasks by increasing difficulty within a single paradigm or accumulating knowledge, PPT progressively integrates fundamentally distinct reasoning paradigms, transitioning from familiar to unfamiliar, while fostering synergy across paradigms. 

## 3.4 Inference

We present an inference method combining variable reasoning depth and multi-paradigm sampling to solve comprehensive mathematical tasks. Prompts with variable reasoning depth. In the zero-shot inference phase, CoR-Math-7B exhibits proficiency in multi-paradigm reasoning, enabling adjustable reasoning depths by modifying prompts based on task requirements. Initially, the model is prompted to conduct NLR, thereby activating a wide range of knowledge patterns associated with natural language from the pre-training corpus. Subsequently, the inference method is tailored to the specific problem type. For example, in theorem proving, the model switches to SR, which is more suitable for formal deduction. Since the SR output is structured in Lean 4, the relevant proof segment can be extracted as the final solution. In arithmetic computations, the model first employs NLR, followed by SR for logical coherence, and concludes with AR for precise calculations. A summary box is utilized to present the final results. The paradigm-switching behavior depends on the presence of prompts and can be categorized into instruction-followed reasoning and instruction-free reasoning. This section primarily presents results from the former. Instructionfree reasoning, which explores how the model autonomously switches paradigms without prompt guidance, will be demonstrated in Appendix D.1. This adaptable paradigm effectively captures the distinct reasoning patterns inherent in different paradigms and demonstrates flexibility in accommodating a variety of scenarios. 

Sequential Multi-Paradigm Sampling (SMPS). Instead of token-level sampling based on tree structures (Qiu et al., 2024; Xiong et al., 2024) within single-paradigm reasoning paths, we purpose sequential paradigm-level sampling method, named SMPS. This allows the model to generate outputs by sampling across different paradigms. For instance, in a two-paradigm reasoning scenario, the model first instantiates J distinct paths for the initial reasoning paradigm $\tau _ { 1 }$ 

$$
\tau_ {1 j} \sim \mathbb {P} (\tau_ {1 j} \mid x), \quad \forall j \in 1, \ldots , J\tag{4}
$$

Subsequently, for each of these τ paths, the model instantiates K possible paths for the secondary reasoning paradigm $\tau _ { 2 }$ . 

$$
\tau_ {2 k} \sim \mathbb {P} (\tau_ {2 k} \mid x, \tau_ {1 j}), \quad \forall k \in 1, \dots , K, \forall j\tag{5}
$$

This hierarchical sampling process yields a total of $J \times K$ potential responses, denoted as y. 

$$
y _ {j k} \sim \mathbb {P} (y _ {j k} \mid x, \tau_ {1 j}, \tau_ {2 k}), \quad \forall j, k\tag{6}
$$

Overall, the SMPS method utilizes a combinatorial expansion of reasoning paths to explore a diverse paradigm-based solution space, enhancing the robustness and depth of the reasoning results. 

## 4 Experimental Settings

## 4.1 Evaluation Setup

Datasets. We conduct extensive experiments to evaluate the model’s mathematical reasoning abilities, including arithmetic computation and theorem proving. The arithmetic computation ability is evaluated on the GSM8K (Cobbe et al., 2021), MATH (Hendrycks et al., 2021), AMC2023 (AI-MO, 2024b) and AIME2024 (AI-MO, 2024a) datasets. Theorem proving ability is evaluated on the miniF2F test set (Zheng et al., 2022), which features mathematical problems of Olympiad difficulty. Further details are in Appendix B.4. 

Metrics. Accuracy is the primary evaluation metric. For arithmetic tasks, we adopt widely-used CoT settings (Wei et al., 2022), rounding answers to the nearest integer. The SymPy library<sup>1</sup> is used for parsing and evaluation. To handle numerical representation variations, the model explicitly states the final answer (LI et al., 2024). For theorem proving, we follow the recent advances (Xin et al., 2024) adapting the miniF2F benchmark from Lean 3 to Lean 4. The pass@N metric evaluates proof correctness within N attempts. Our CoR-Math-7B uses SMPS with NLR and SR, where $N = N _ { \mathrm { N L R } } \times N _ { \mathrm { S R } }$ . Model settings are detailed in Appendix B.5. 

## 4.2 Implementation Details

We fine-tuned widely-used DeepSeekMath-Base 7B (Shao et al., 2024) and Llama-3.1 8B (Dubey et al., 2024) models, employing our PPT method on MPM dataset. Unless otherwise specified, CoR-Math-7B model is based on DeepSeekMath-Base 7B. The details are available in Appendix B.3. 

## 4.3 Baselines

We examine three categories of baseline models, with results reported using CoT prompting. 

General-purpose mathematical models. To evaluate CoR’s generalization, we includes Mustard (Huang et al., 2024), DeepSeekMath (Shao et al., 2024), InternLM-Math (Ying et al., 2024b), Llama-3.1 (Dubey et al., 2024), Mistral (Jiang et al., 2023), and Llemma (Azerbayev et al., 2024). Task-specific mathematical models. We consider several expert models on mathematical optimization for specific tasks, such as largescale mathematical data and inference search improvements. The arithmetic experts encompass Qwen2.5-Math (Yang et al., 2024), Wizard-Math (Luo et al., 2023), MetaMath (Yu et al., 2024), DART-Math (Tong et al., 2024), InternLM-Math (Ying et al., 2024b), , DeepSeekMath-Instruct / RL (Shao et al., 2024), Xwin-Math (Li et al., 2024), ToRA (Gou et al., 2024), and NuminaMath (LI et al., 2024). The theorem proving experts include LLM-Step (Welleck and Saha, 2023), GPT-f (Polu and Sutskever, 2020), Lean-STaR (Lin et al., 2024a), Hypertree Proof Search (Lample et al., 2022), DeepSeek-Prover (Xin et al., 2024), and InternLM2.5- StepProver (Wu et al., 2024). 

Foundation and proprietary models. We present open-source foundation models Llama-3.1 (Dubey et al., 2024), Mistral (Jiang et al., 2023), and Mixtral (Jiang et al., 2024), along with proprietary models o1-mini (OpenAI, 2024), GPT-4 (OpenAI, 2023), and GPT-4o (Hurst et al., 2024). 

## 5 Main Results

## 5.1 Comparisons with General-purpose Mathematical Models

To evaluate the comprehensive mathematical reasoning abilities of CoR-Math-7B, we compare it with widely-used models and SOTA mathematical models, which are based on single reasoning paradigms. As shown in Table 1, CoR-Math-7B outperforms others across five challenging benchmarks in a zero-shot setting, highlighting its strong cross-task versatility. The main findings are: (1) arithmetic computation: CoR-Math-7B achieves the best results, outperforming InterLM2-Math-Plus-7B by an absolute 13.7% on MATH and 2.9% on GSM8K. Compared to Llama-3.1-8B-Instruct, it achieves an increase in correct answers on AMC2023 (34 vs 16) and AIME2024 (12 vs 5). (2) theorem proving: CoR-Math-7B sets a new zero-shot benchmark on miniF2F, even surpassing GPT-4o model by a 41.0% absolute increase in 

<table><tr><td rowspan="2">Model</td><td rowspan="2">Model Types</td><td colspan="4">Arithmetic Computation</td><td colspan="2">Theorem Proving</td></tr><tr><td>MATH Pass@1</td><td>GSM8k Pass@1</td><td>AMC2023 Maj@64</td><td>AIME2024 Maj@64</td><td>miniF2F-test Sample Budget (N)</td><td>miniF2F-test Pass@N</td></tr><tr><td>o1-mini</td><td>Proprietary</td><td><eq>90.0^0</eq></td><td><eq>94.8^0</eq></td><td><eq>38/40^0</eq></td><td><eq>17/30^0</eq></td><td>1</td><td>13.2</td></tr><tr><td>GPT-4</td><td>Proprietary</td><td><eq>42.5^0</eq></td><td><eq>87.1^0</eq></td><td><eq>25/40^0</eq></td><td><eq>6/30^0</eq></td><td>128</td><td>24.6</td></tr><tr><td>GPT-4o</td><td>Proprietary</td><td><eq>76.6^0</eq></td><td><eq>90.5^0</eq></td><td><eq>24/40^0</eq></td><td><eq>3/30^0</eq></td><td>128</td><td>25.0</td></tr><tr><td>Llama-3.1-8B</td><td>Foundation</td><td><eq>4.2^0</eq></td><td><eq>6.2^0</eq></td><td><eq>1/40^0</eq></td><td><eq>0/30^0</eq></td><td>128</td><td>25.8</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>Foundation</td><td><eq>47.2^0</eq></td><td><eq>76.6^0</eq></td><td><eq>16/40^0</eq></td><td><eq>5/30^0</eq></td><td>128</td><td>23.4</td></tr><tr><td>Mistral-7B</td><td>Foundation</td><td>14.3</td><td>40.3</td><td><eq>5/40^0</eq></td><td><eq>0/30^0</eq></td><td><eq>1\times 32\times 100</eq></td><td>22.1</td></tr><tr><td>Mixtral-8x7B</td><td>Foundation</td><td>28.4</td><td>74.4</td><td><eq>8/40^0</eq></td><td><eq>0/30^0</eq></td><td><eq>1\times 32\times 100</eq></td><td>23.4</td></tr><tr><td>MUSTARD</td><td>GMM</td><td><eq>13.8^0</eq></td><td><eq>27.9^0</eq></td><td>-</td><td>-</td><td>1</td><td>7.8</td></tr><tr><td>Llemma-7B</td><td>GMM</td><td>18.6</td><td>41.0</td><td><eq>2/40^0</eq></td><td><eq>0/30^0</eq></td><td><eq>1\times 32\times 100</eq></td><td>26.2</td></tr><tr><td>DeepSeekMath-7B-Base</td><td>GMM</td><td><eq>11.8^0</eq></td><td><eq>22.2^0</eq></td><td><eq>3/40^0</eq></td><td><eq>0/30^0</eq></td><td><eq>1\times 32\times 100</eq></td><td>28.3</td></tr><tr><td>InternLM2-Math-7B-Base</td><td>GMM</td><td>21.5</td><td>49.2</td><td><eq>6/40^0</eq></td><td><eq>0/30^0</eq></td><td><eq>1\times 32\times 100</eq></td><td>30.3</td></tr><tr><td>InternLM2-Math-Plus-7B</td><td>GMM</td><td><eq>53.0^0</eq></td><td><eq>85.8^0</eq></td><td><eq>15/40^0</eq></td><td><eq>1/30^0</eq></td><td><eq>1\times 32\times 100</eq></td><td>43.3</td></tr><tr><td rowspan="3">CoR-Math-7B</td><td rowspan="3">GMM</td><td rowspan="3"><eq>66.7^0</eq></td><td rowspan="3"><eq>88.7^0</eq></td><td rowspan="3"><eq>34/40^0</eq></td><td rowspan="3"><eq>12/30^0</eq></td><td><eq>128\times 1</eq></td><td><eq>52.9^0</eq></td></tr><tr><td><eq>32\times 100</eq></td><td><eq>59.4^0</eq></td></tr><tr><td><eq>128\times 128</eq></td><td><eq>66.0^0</eq></td></tr></table>


Table 1: A overall comparison of CoR-Math-7B with three types of general mathematical reasoners (Proprietary, Foundational, and General-purpose Mathematical Models (GMM)) on three mathematical benchmarks. Results are shown for zero-shot (denoted by ) or few-shot settings by default. For the miniF2F benchmark, we report the best results from the relevant literature with a specified sample budget. Bolded scores are the best performance among all models except for the proprietary one.


<table><tr><td>Model</td><td>Sample Budget N</td><td>miniF2F</td></tr><tr><td>LLMStep</td><td>1×32×100</td><td>27.9</td></tr><tr><td>GPT-f</td><td>64×8×512</td><td>36.6</td></tr><tr><td>Hypertree Proof Search</td><td>64×5000</td><td>41.0</td></tr><tr><td>Lean-STaR</td><td>64×1×50</td><td>46.3</td></tr><tr><td>DeepSeek-Prover-V1.5-Base</td><td>6400</td><td>42.2</td></tr><tr><td>DeepSeek-Prover-V1.5-SFT + RMaxTS</td><td>4×6400</td><td>56.3</td></tr><tr><td>DeepSeek-Prover-V1.5-SFT + RMaxTS</td><td>32×6400</td><td>60.2</td></tr><tr><td>DeepSeek-Prover-V1.5-RL + RMaxTS</td><td>4×6400</td><td>59.6</td></tr><tr><td>DeepSeek-Prover-V1.5-RL + RMaxTS</td><td>32×6400</td><td>63.5</td></tr><tr><td>InternLM2.5-StepProver</td><td>64×32×100</td><td>54.5</td></tr><tr><td>InternLM2.5-StepProver-BF</td><td>1×32×600</td><td>47.3</td></tr><tr><td>InternLM2.5-StepProver-BF</td><td>256×32×600</td><td>59.4</td></tr><tr><td>InternLM2.5-StepProver-BF+CG</td><td>2×32×600</td><td>50.7</td></tr><tr><td>InternLM2.5-StepProver-BF+CG</td><td>256×32×600</td><td>65.9</td></tr><tr><td>CoR-Math-7B</td><td>128×128</td><td><eq>66.0^0</eq></td></tr></table>


Table 2: The zero-shot ( ) performance of CoR-Math-7B and the few-shot performance of theorem proving optimized models on the miniF2F benchmark. The highest scores are in bold.


## 5.2 Comparisons with Theorem Proving Experts

few-shot setting. (3) zero-shot vs few-shot: CoR-Math-7B’s zero-shot performance exceeds all fewshot results from other models, with an absolute 37.7% improvement over DeepSeekMath-7B-Base on miniF2F. These findings indicate the generalization capability and comprehensive mathematical reasoning ability of the CoR framework. This suggests NLR descriptions and SR verification rehearse precise mathematical reasoning for AR. Additionally, we provide qualitative analysis of error cases in Appendix D.2. In summary, CoR effectively handles diverse reasoning challenges, with synergies across paradigms boosting its overall performance. 

Table 2 shows CoR-Math-7B’s impressive zeroshot performance on miniF2F, achieving 66.0% accuracy without demonstrations, while competitors require few-shot examples. Its zero-shot performance is noteworthy when considering computational efficiency. For instance, under similar computational constraints, CoR-Math-7B surpasses DeepSeek-Prover-V1.5-RL + RMaxTS by 6.4% absolute points (66.0% vs 59.6% with 128 128 vs 4 6400) and InternLM2.5-StepProver-BF+CG by 15.3% absolute points (66.0% vs 50.7% with 128 128 vs $2 \times 3 2 \times 6 0 0 )$ . Even with larger sample sizes, CoR-Math-7B remains superior, surpassing few-shot InternLM2.5-StepProver-BF+CG by 0.1%, despite considering 300 times more solutions. These results highlight the efficiency of multi-paradigm reasoning and CoR’s ability to explore paradigm-based solution spaces. 

## 5.3 Comparisons with Arithmetic Experts

Since several mathematical reasoners are primarily trained on arithmetic computation rather than formal theorem proving, and often struggle with the latter, we classify them as experts in arithmetic computation to enable a fair comparison. Table 3 highlights CoR-Math-7B’s strong performance in arithmetic computation, surpassing expert models and achieving a 15% improvement over RL-based methods, with DeepSeekMath-RL-7B as the representative, on the MATH benchmark for arithmetic tasks. Unlike methods such as ToRA and NuminaMath, which rely on code without complete reasoning, CoR-Math-7B leverages multiparadigm reasoning to outperform them across all benchmarks. For example, on the MATH dataset, CoR-Math-7B achieves an 11.4% absolute improvement over the tool-integrated NuminaMath-7B-TIR, underscoring the effectiveness of complete code-based reasoning over fragmented interleaving of natural language and code. To further explore the relationship between resource utilization (training sample size) and performance, we fit a quadratic function to the current SOTA models and perform a Pareto frontier analysis (Branke et al., 2008), as shown in Fig. 1 (Details in Appendix B.6). CoR-Math-7B outperforms the optimal performance of single-paradigm methods with similar data volume, suggesting a new optimal curve for multi-paradigm reasoning that surpasses single-paradigm approaches. 

<table><tr><td rowspan="2">Model</td><td>MATH</td><td>GSM8k</td></tr><tr><td>ZS Pass@1</td><td>ZS Pass@1</td></tr><tr><td>WizardMath</td><td>10.7</td><td>54.9</td></tr><tr><td>MetaMath-7B</td><td>19.8</td><td>66.5</td></tr><tr><td>MetaMath-Llemma-7B</td><td>30.0</td><td>69.2</td></tr><tr><td>MetaMath-Mistral-7B</td><td>28.2</td><td>77.7</td></tr><tr><td>ToRA</td><td>40.1</td><td>68.8</td></tr><tr><td>ToRA-CODE</td><td>44.6</td><td>72.6</td></tr><tr><td>NuminaMath-7B-CoT</td><td><eq>54.4^†</eq></td><td><eq>66.6^†</eq></td></tr><tr><td>Xwin-Math-7B</td><td>40.6</td><td>82.6</td></tr><tr><td>DeepSeekMath-Instruct-7B</td><td>46.8</td><td>73.6</td></tr><tr><td>NuminaMath-7B-TIR</td><td><eq>55.3^†</eq></td><td><eq>73.6^†</eq></td></tr><tr><td>DeepSeekMath-RL-7B</td><td>51.7</td><td>88.2</td></tr><tr><td>DART-Math-7B</td><td>53.6</td><td>86.6</td></tr><tr><td>Qwen2.5-Math-7B-Instruct</td><td>83.6</td><td>95.2</td></tr><tr><td>CoR-Math-7B</td><td>66.7</td><td>88.7</td></tr></table>


Table 3: Zero-shot (ZS) performance on the MATH and GSM8K benchmarks for the arithmetic task. denotes our reported results with the open-sourced model weights. The best results are in bold, and the secondbest results are underlined.


## 6 Ablation Study

## 6.1 Impact of Stages in PPT Method

We evaluate the PPT method using two base models, Llama-3.1-8B and DeepseekMath-7B-Base, on the MATH and GSM8K benchmarks at different PPT stages. As shown in Fig. 4, the vanilla models exhibit limited performance. After stage ①, performance improves significantly, with Llama-3.1-Base gaining 47.9% on MATH and 61.0% on 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/db771bc9-d734-41ba-ac35-ccfc42738592/07954d088a14b79f08fcbfc5956553268db2e6f14398720d03a7cbaf135e72b0.jpg)



Figure 4: An evaluation of the effectiveness of the PPT strategy. We present the zero-shot Pass@1 results on the MATH and GSM8k benchmarks across three cumulative stages of the PPT strategy. The results highlight the PPT strategy’s cumulative effectiveness, showing increased performance with each progressive stage.


<table><tr><td>Benchmark</td><td>NLR→AR→SR→Summary</td><td>NLR→SR→AR→Summary</td></tr><tr><td>MATH</td><td>49.9</td><td>66.7</td></tr><tr><td>GSM8K</td><td>84.2</td><td>88.7</td></tr></table>


Table 4: Zero-shot Pass@1 results for varying paradigm orders on the MATH and GSM8k benchmarks. The best results are in bold.


GSM8K, emphasizing the need for mathematical understanding and model warming. Stage ②, utilizing NLR and AR data, yields limited gains compared to stage ① due to pre-training data already containing relevant content. However, this does not imply that stage ② is unimportant, as it plays a critical role in CoR by reactivating and refining computational skills essential for collaborating with other reasoning paradigms. Both stages ① and ② focus on reactivating models for mathematical tasks. In stage ③, training with three paradigms further enhances performance, suggesting that rare or unseen paradigms improve reasoning abilities. The consistent performance gains with added paradigms in Fig. 4 robustly demonstrate the superiority of our multi-paradigm PPT approach over single- or fewer-paradigm finetuning of the same base model, yielding a more powerful, comprehensive mathematical reasoner. 

## 6.2 Order of the Reasoning Paradigms

As shown in Table 4, we investigate the impact of varying the sequence of reasoning paradigms in CoR-Math-7B during zero-shot arithmetic tasks, modifying only the prompt structure. Given that NLR closely aligns with the pre-training of LLMs, it serves as the first paradigm. The results indicate that SR before AR resulted in the highest accuracy, with 66.7% on MATH, compared to 49.9% for AR before SR. A possible explanation is that SR plays a crucial role in decomposing the problem into manageable sub-steps, thereby providing a structured foundation for subsequent AR. This demonstrates that paradigms in CoR are interdependent, not isolated, and performance hinges on how they are linked during reasoning, beyond merely mastering individual paradigms. Furthermore, positioning AR preceding the Summary boxes may facilitate a more coherent integration of the computational outcome into the final answer. 

<table><tr><td>Model</td><td>GSM8k@1</td><td>MATH@1</td><td>AMC2023 (Maj@64)</td><td>AIME2024 (Maj@64)</td><td>miniF2F@128</td></tr><tr><td>DSM + NLR</td><td>33.9</td><td>27.8</td><td>14/40</td><td>1/30</td><td>-</td></tr><tr><td>DSM + AR</td><td>75.8</td><td>37.6</td><td>10/40</td><td>0/30</td><td>-</td></tr><tr><td>DSM + SR</td><td>-</td><td>-</td><td>-</td><td>-</td><td>44.3</td></tr><tr><td>DSM + CoR (ours)</td><td>88.7</td><td>66.7</td><td>34/40</td><td>12/30</td><td>52.9</td></tr></table>


Table 5: Zero-shot results of multi- and single-paradigm fine-tuning on DeepSeekMath-7B-Base (DSM) for mathematical Reasoning benchmarks. The best results are in bold.


<table><tr><td>Model</td><td>Size</td><td>MATH</td><td>GSM8k</td><td>miniF2F</td></tr><tr><td rowspan="2">Qwen2.5-Math (Base)</td><td>1.5B</td><td>34.0</td><td>39.3</td><td>0.0</td></tr><tr><td>7B</td><td>51.8</td><td>90.0</td><td>0.0</td></tr><tr><td rowspan="2">Qwen2.5-Math (CoR)</td><td>1.5B</td><td>57.6</td><td>84.5</td><td>51.6</td></tr><tr><td>7B</td><td>64.7</td><td>90.0</td><td>52.5</td></tr><tr><td rowspan="2">Llama-3.1 (Base)</td><td>8B</td><td>4.2</td><td>6.2</td><td>25.8</td></tr><tr><td>70B</td><td>16.8</td><td>20.5</td><td>22.5</td></tr><tr><td rowspan="2">Llama-3.1 (CoR)</td><td>8B</td><td>58.2</td><td>84.0</td><td>53.3</td></tr><tr><td>70B</td><td>70.7</td><td>90.0</td><td>56.2</td></tr></table>


Table 6: Zero-shot Pass@1 results on the mathematical benchmarks across different model scales. The best scores are in bold.


## 6.3 Impact of Model Scales

Table 6 evaluates base model performance under the CoR framework with varying parameters, such as Qwen2.5-Math models (1.5B / 7B) and Llama-3.1-8B (8B / 70B). The results show that CoR scales with model size. The 7B Qwen2.5-Math outperforms the 1.5B version by 7.1% on MATH and 5.5% on GSM8K. Similarly, Llama-3.1 70B gains 2.9% over its 8B version on minF2F. This supports that CoR exhibits parameter scalability. 

## 6.4 Multi- vs. Single-Paradigm Reasoning

To further validate the superior performance of multi-paradigm reasoning, we conducted an ablation experiment comparing it to single-paradigm fine-tuning We fine-tuned the DeepSeekMath-7B-Base model separately using single-paradigm paths (NLR, AR, SR) extracted from the MPM dataset. All experiments employed identical base models, problem sets, and training data sizes (10, 000 samples per paradigm). Experimental results, summarized in Table 5, clearly demonstrate that our CoR model significantly outperforms single-paradigm fine-tuned models across diverse tasks. Specifically, for GSM8k and MATH datasets, CoR achieves improvements of 12.9% and 29.1%, respectively, compared to the best-performing single-paradigm approaches. On AMC2023 and AIME2024, CoR solves 20 and 11 more problems, respectively, compared to single-paradigm approaches. Lastly, on minF2F, CoR achieves an accuracy improvement of 8.6%. Collectively, these improvements underline CoR’s effectiveness in integrating multiple reasoning paradigms, enabling robust and generalized problem-solving capabilities. 

## 7 Conclusions

This paper introduces CoR, a novel unified reasoning framework that enhances mathematical reasoning in LLMs by synergistically integrating natural language, algorithmic, and symbolic. CoR tackles the limitations of single-paradigm approaches with the PPT strategy and the MPM dataset, enabling LLMs to progressively master diverse reasoning paradigms for improved generalization and performance. Our CoR-Math-7B outperformed SOTA models on five challenging benchmarks, showcasing enhanced zero-shot generalization. Ablation studies validated the benefits of progressive training and paradigm sequence. Additionally, CoR offers a new perspective on testtime scaling through multi-paradigm reasoning to enhance both efficiency and performance. 

## Acknowledgments

This work was partly supported by the Shenzhen Science and Technology Program ( JSGG20220831110203007 ) and the research grant No. CT20240905126002 of the Doubao Large Model Fund. 

## Limitations

This paper proposes a general zero-shot reasoning method that supports different paradigms to solve multiple tasks. Evaluating zero-shot performance is challenging, as many methods rely on a single evaluation metric, like zero-shot pass@1. Additionally, most approaches focus on few-shot settings, as seen in benchmarks like miniF2F, limiting the comprehensiveness of our evaluation with aligned settings. To address this, we include additional experiments in Appendix C.2 to examine the impact of different evaluation strategies. On the other term, CoR can explore a multi-paradigm solution space, covering up to three paradigms, such as in large-scale arithmetic tasks. However, due to resource constraints and the fact that other methods can only predict within a single paradigm, we did not further consider using SMPS on benchmarks such as MATH to ensure fairness. 

## References



AI-MO. 2024a. Aime 2023. https://huggingface. co/datasets/AI-MO/aimo-validation-aime/. 





AI-MO. 2024b. Amc 2023. https://huggingface. co/datasets/AI-MO/aimo-validation-amc/. 





Zhangir Azerbayev, Hailey Schoelkopf, Keiran Paster, Marco Dos Santos, Stephen Marcus McAleer, Albert Q. Jiang, Jia Deng, Stella Biderman, and Sean Welleck. 2024. Llemma: An open language model for mathematics. In ICLR. OpenReview.net. 





Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Michal Podstawski, Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Hubert Niewiadomski, Piotr Nyczyk, and Torsten Hoefler. 2024. Graph of thoughts: Solving elaborate problems with large language models. In AAAI, pages 17682–17690. AAAI Press. 





Jürgen Branke, Kalyanmoy Deb, Kaisa Miettinen, and Roman Slowinski, editors. 2008. Multiobjective Optimization, Interactive and Evolutionary Approaches [outcome of Dagstuhl seminars], volume 5252 of Lecture Notes in Computer Science. Springer. 





Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W. Cohen. 2023. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. Trans. Mach. Learn. Res., 2023. 





Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. CoRR, abs/2110.14168. 





Tri Dao. 2024. Flashattention-2: Faster attention with better parallelism and work partitioning. In ICLR. OpenReview.net. 





Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurélien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Rozière, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Graeme Nail, Grégoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel M. Kloumann, Ishan Misra, Ivan Evtimov, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, and et al. 2024. The llama 3 herd of models. CoRR, abs/2407.21783. 





Edward A Feigenbaum, Julian Feldman, et al. 1963. Computers and thought, volume 37. New York McGraw-Hill. 





Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. 2023. PAL: program-aided language models. In ICML, volume 202 of Proceedings of Machine Learning Research, pages 10764–10799. PMLR. 





Zhibin Gou, Zhihong Shao, Yeyun Gong, Yelong Shen, Yujiu Yang, Minlie Huang, Nan Duan, and Weizhu Chen. 2024. Tora: A tool-integrated reasoning agent for mathematical problem solving. In ICLR. Open-Review.net. 





Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the MATH dataset. In NeurIPS Datasets and Benchmarks. 





Mohammad Javad Hosseini, Hannaneh Hajishirzi, Oren Etzioni, and Nate Kushman. 2014. Learning to solve arithmetic word problems with verb categorization. In EMNLP, pages 523–533. ACL. 





Yinya Huang, Xiaohan Lin, Zhengying Liu, Qingxing Cao, Huajian Xin, Haiming Wang, Zhenguo Li, Linqi Song, and Xiaodan Liang. 2024. MUSTARD: mastering uniform synthesis of theorem and proof data. In ICLR. OpenReview.net. 



Aaron Hurst, Adam Lerer, Adam P. Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, Aleksander Madry, Alex Baker-Whitcomb, Alex Beutel, Alex Borzunov, Alex Carney, Alex Chow, Alex Kirillov, Alex Nichol, Alex Paino, Alex Renzin, Alex Tachard Passos, Alexander Kirillov, Alexi Christakis, Alexis Conneau, Ali Kamali, Allan Jabri, Allison Moyer, Allison Tam, Amadou Crookes, Amin Tootoonchian, Ananya Kumar, Andrea Vallone, Andrej Karpathy, Andrew Braunstein, Andrew Cann, Andrew Codispoti, Andrew Galu, Andrew Kondrich, Andrew Tulloch, Andrey Mishchenko, Angela Baek, Angela Jiang, Antoine Pelisse, Antonia Woodford, Anuj Gosalia, Arka Dhar, Ashley Pantuliano, Avi Nayak, Avital Oliver, Barret Zoph, Behrooz Ghorbani, Ben Leimberger, Ben Rossen, Ben Sokolowsky, Ben Wang, Benjamin Zweig, Beth Hoover, Blake Samic, Bob McGrew, Bobby Spero, Bogo Giertler, Bowen Cheng, Brad Lightcap, Brandon Walkin, Brendan Quinn, Brian Guarraci, Brian Hsu, Bright Kellogg, Brydon Eastman, Camillo Lugaresi, Carroll L. Wainwright, Cary Bassin, Cary Hudson, Casey Chu, Chad Nelson, Chak Li, Chan Jun Shern, Channing Conger, Charlotte Barette, Chelsea Voss, Chen Ding, Cheng Lu, Chong Zhang, Chris Beaumont, Chris Hallacy, Chris Koch, Christian Gibson, Christina Kim, Christine Choi, Christine McLeavey, Christopher Hesse, Claudia Fischer, Clemens Winter, Coley Czarnecki, Colin Jarvis, Colin Wei, Constantin Koumouzelis, and Dane Sherburn. 2024. Gpt-4o system card. CoRR, abs/2410.21276. 



Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de Las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2023. Mistral 7b. CoRR, abs/2310.06825. 





Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de Las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Théophile Gervet, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2024. Mixtral of experts. CoRR, abs/2401.04088. 





Haein Kong, Yongsu Ahn, Sangyub Lee, and Yunho Maeng. 2024. Gender bias in llm-generated interview responses. CoRR, abs/2410.20739. 





Guillaume Lample, Timothée Lacroix, Marie-Anne Lachaux, Aurélien Rodriguez, Amaury Hayat, Thibaut Lavril, Gabriel Ebner, and Xavier Martinet. 2022. Hypertree proof search for neural theorem proving. In NeurIPS. 





Chen Li, Weiqi Wang, Jingcheng Hu, Yixuan Wei, Nanning Zheng, Han Hu, Zheng Zhang, and Houwen Peng. 2024. Common 7b language models already possess strong math capabilities. CoRR, abs/2403.04706. 





Jia LI, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Costa Huang, Kashif Rasul, Longhui Yu, Albert Jiang, Ziju Shen, Zihan Qin, Bin Dong, Li Zhou, Yann Fleureau, Guillaume Lample, and Stanislas Polu. 2024. Numinamath. https://github.com/project-numina/ aimo-progress-prize. 





Haohan Lin, Zhiqing Sun, Yiming Yang, and Sean Welleck. 2024a. Lean-star: Learning to interleave thinking and proving. CoRR, abs/2407.10040. 





Zicheng Lin, Tian Liang, Jiahao Xu, Xing Wang, Ruilin Luo, Chufan Shi, Siheng Li, Yujiu Yang, and Zhaopeng Tu. 2024b. Critical tokens matter: Tokenlevel contrastive estimation enhence llm’s reasoning capability. arXiv preprint arXiv:2411.19943. 





Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jianguang Lou, Chongyang Tao, Xiubo Geng, Qingwei Lin, Shifeng Chen, and Dongmei Zhang. 2023. Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct. CoRR, abs/2308.09583. 





Ruilin Luo, Zhuofan Zheng, Yifan Wang, Yiyao Yu, Xinzhe Ni, Zicheng Lin, Jin Zeng, and Yujiu Yang. 2025. Ursa: Understanding and verifying chainof-thought reasoning in multimodal mathematics. arXiv preprint arXiv:2501.04686. 





Frederic P. Miller, Agnes F. Vandome, and John McBrewster. 2009. Levenshtein Distance: Information theory, Computer science, String (computer science), String metric, Damerau?Levenshtein distance, Spell checker, Hamming distance. Alpha Press. 





OpenAI. 2023. GPT-4 technical report. CoRR, abs/2303.08774. 





OpenAI. 2024. Openai o1 system card. Preprint, arXiv:2412.16720. 





Stanislas Polu and Ilya Sutskever. 2020. Generative language modeling for automated theorem proving. CoRR, abs/2009.03393. 





Jiahao Qiu, Yifu Lu, Yifan Zeng, Jiacheng Guo, Jiayi Geng, Huazheng Wang, Kaixuan Huang, Yue Wu, and Mengdi Wang. 2024. Treebon: Enhancing inference-time alignment with speculative tree-search and best-of-n sampling. CoRR, abs/2410.16033. 





Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. 2020. Zero: memory optimizations toward training trillion parameter models. In SC, page 20. IEEE/ACM. 





Baptiste Rozière, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, Artyom Kozhevnikov, Ivan Evtimov, Joanna Bitton, Manish Bhatt, Cristian Canton-Ferrer, Aaron Grattafiori, Wenhan Xiong, Alexandre Défossez, Jade Copet, Faisal Azhar, Hugo Touvron, Louis Martin, Nicolas Usunier, Thomas Scialom, and Gabriel Synnaeve. 2023. Code llama: Open foundation models for code. CoRR, abs/2308.12950. 





Bilgehan Sel, Ahmad Al-Tawaha, Vanshaj Khattar, Ruoxi Jia, and Ming Jin. 2024. Algorithm of thoughts: Enhancing exploration of ideas in large language models. In ICML. OpenReview.net. 





Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. CoRR, abs/2402.03300. 





Yuxuan Tong, Xiwen Zhang, Rui Wang, Ruidong Wu, and Junxian He. 2024. Dart-math: Difficulty-aware rejection tuning for mathematical problem-solving. CoRR, abs/2407.13690. 





Shaobo Wang, Xiangqi Jin, Ziming Wang, Jize Wang, Jiajun Zhang, Kaixin Li, Zichen Wen, Zhong Li, Conghui He, Xuming Hu, and Linfeng Zhang. 2025. Data whisperer: Efficient data selection for task-specific llm fine-tuning via few-shot in-context learning. Annual Meeting of the Association for Computational Linguistics. 





Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2023. Self-consistency improves chain of thought reasoning in language models. In ICLR. OpenReview.net. 





Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. 2022. Chain-of-thought prompting elicits reasoning in large language models. In NeurIPS. 





Sean Welleck and Rahul Saha. 2023. LLMSTEP: LLM proofstep suggestions in lean. CoRR, abs/2310.18457. 





Zijian Wu, Suozhi Huang, Zhejian Zhou, Huaiyuan Ying, Jiayu Wang, Dahua Lin, and Kai Chen. 2024. Internlm2.5-stepprover: Advancing automated theorem proving via expert iteration on large-scale LEAN problems. CoRR, abs/2410.15700. 





Huajian Xin, Z. Z. Ren, Junxiao Song, Zhihong Shao, Wanjia Zhao, Haocheng Wang, Bo Liu, Liyue Zhang, Xuan Lu, Qiushi Du, Wenjun Gao, Qihao Zhu, Dejian Yang, Zhibin Gou, Z. F. Wu, Fuli 





Luo, and Chong Ruan. 2024. Deepseek-prover-v1.5: Harnessing proof assistant feedback for reinforcement learning and monte-carlo tree search. CoRR, abs/2408.08152. 





Yunfan Xiong, Ruoyu Zhang, Yanzeng Li, Tianhao Wu, and Lei Zou. 2024. Dyspec: Faster speculative decoding with dynamic token tree structure. CoRR, abs/2410.11744. 





An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, Keming Lu, Mingfeng Xue, Runji Lin, Tianyu Liu, Xingzhang Ren, and Zhenru Zhang. 2024. Qwen2.5-math technical report: Toward mathematical expert model via self-improvement. CoRR, abs/2409.12122. 





Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. 2023. Tree of thoughts: Deliberate problem solving with large language models. In NeurIPS. 





Huaiyuan Ying, Zijian Wu, Yihan Geng, Jiayu Wang, Dahua Lin, and Kai Chen. 2024a. Lean workbook: A large-scale lean problem set formalized from natural language math problems. CoRR, abs/2406.03847. 





Huaiyuan Ying, Shuo Zhang, Linyang Li, Zhejian Zhou, Yunfan Shao, Zhaoye Fei, Yichuan Ma, Jiawei Hong, Kuikun Liu, Ziyi Wang, Yudong Wang, Zijian Wu, Shuaibin Li, Fengzhe Zhou, Hongwei Liu, Songyang Zhang, Wenwei Zhang, Hang Yan, Xipeng Qiu, Jiayu Wang, Kai Chen, and Dahua Lin. 2024b. Internlm-math: Open math large language models toward verifiable reasoning. Preprint, arXiv:2402.06332. 





Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T. Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. 2024. Metamath: Bootstrap your own mathematical questions for large language models. In ICLR. OpenReview.net. 





Yiyao Yu, Junjie Wang, Yuxiang Zhang, Lin Zhang, Yujiu Yang, and Tetsuya Sakai. 2023. EALM: introducing multidimensional ethical alignment in conversational information retrieval. In SIGIR-AP, pages 32– 39. ACM. 





Yu-Xiang Zhang, Junjie Wang, Xinyu Zhu, Tetsuya Sakai, and Hayato Yamana. 2024. SSR: solving named entity recognition problems via a singlestream reasoner. ACM Trans. Inf. Syst., 42(5):138:1– 138:28. 





Jinman Zhao and Xueyan Zhang. 2024. Large language model is not a (multilingual) compositional relation reasoner. In First Conference on Language Modeling. 





Kunhao Zheng, Jesse Michael Han, and Stanislas Polu. 2022. minif2f: a cross-system benchmark for formal olympiad-level mathematics. In ICLR. OpenReview.net. 





Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc V. Le, and Ed H. Chi. 2023. Least-to-most prompting enables complex reasoning in large language models. In ICLR. OpenReview.net. 





Xinyu Zhu, Junjie Wang, Lin Zhang, Yuxiang Zhang, Yongfeng Huang, Ruyi Gan, Jiaxing Zhang, and Yujiu Yang. 2023. Solving math word problems via cooperative reasoning induced language models. In ACL (1), pages 4471–4485. Association for Computational Linguistics. 



## Appendix

## A Discussion on Reasoning Hierarchy: Paradigms, Paths, and Steps

This section attempts to explore the essence of reasoning, contrasts CoR with current methods, examines the factors contributing to its effectiveness, and provides a new perspective for future research. 

As shown in Fig. 5, this paper posits that reasoning texts generated by LLMs exhibit a reasoning hierarchical structure, which consists of three levels: reasoning paradigms, reasoning paths, and reasoning steps. 

• Reasoning steps represent the fundamental units, each comprising one or more tokens and encompassing an incomplete stage of the solution process. 

• Reasoning paths consist of several reasoning steps, forming a complete line of reasoning that typically includes a final answer and the solution process. 

• Reasoning paradigms comprise one or more reasoning paths. They often contain multiple potential final answers, thus necessitating a summarization method, such as a summary module, to extract the ultimate answer. Furthermore, a reasoning paradigm uses a single knowledge media, such as natural language. 

Furthermore, current studies can be categorized based on their focus. As shown in Fig. 5 (a), contemporary work concentrates within a single paradigm, optimizing along two dimensions: depth (the number of reasoning steps) and width (the number of reasoning paths). For instance, regarding reasoning depth, the CoT method (Wei et al., 2022) employs prompts to increase intermediate steps within one reasoning path to achieve higher performance. Concerning reasoning width, some approaches (Wang et al., 2023) involve altering sampling techniques to generate multiple distinct reasoning paths. Random sampling exemplifies this. Other studies (Zhu et al., 2023) employ scoring mechanisms for reasoning steps, guiding LLMs to generate several more desirable reasoning paths. Monte Carlo search illustrates this. Building upon these generated reasoning paths, existing work proposes various integration strategies to obtain the final answer, such as Best-of-N and Self-consistency (Wang et al., 2023). Specifically, as shown in Fig. 6 (a), recent advanced studies introduce the deep reasoning method, which focuses on generating serial concatenation of reasoning paths followed by summarization (like OpenAI o1 (OpenAI, 2024)). Since these methods are based on different optimization dimensions, we can combine them feasibly in applications. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/db771bc9-d734-41ba-ac35-ccfc42738592/0fb7fedffd511a9353d7af7447b593e07f88d6cb25b51fb8119391a782b672e3.jpg)



Figure 5: An overview of the reasoning hierarchical structure. (a) A single reasoning paradigm depicts multiple distinct reasoning paths. (b) An example of singleparadigm reasoning includes one reasoning path, which contains some reasoning steps. (c) Multi-paradigm reasoning includes several distinct reasoning paradigms.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/db771bc9-d734-41ba-ac35-ccfc42738592/00b5145b8617079061c31661401b07154a974169be58445bd5a5ec574dcb0cc2.jpg)



Figure 6: A comparison of reasoning methods in several advanced studies.


Specifically, some approaches prioritize a dominant paradigm for reasoning yet hope to integrate other paradigms for guidance. For instance, as shown in Fig. 6 (b), the reasoning process of ToRA (Gou et al., 2024) presents an interleaved reasoning method with different knowledge media. It involves initial natural language generation, which is followed by code generation. Subsequently, the process awaits the results of code execution before generating further natural language. InternLM2.5-StepProver (Wu et al., 2024) exhibits a similar pattern. It incorporates natural language annotations to support symbolic reasoning steps. However, these interleaved reasoning approaches do not constitute true multi-paradigm reasoning. The primary paradigm can independently achieve the final answer without the supplementary paradigms. Consequently, these methods imply a single-paradigm reasoning approach enhanced by other paradigms, rather than genuine multi-paradigm reasoning. 

Current methods often overlook the synergistic effects between paradigms and underestimate the importance of complete reasoning for achieving accurate results. To address this challenge, as shown in Fig. 5 (c) and Fig. 6 (c), we introduce Chain-of-Reasoning (CoR), a unified reasoning framework capable of multi-paradigm reasoning. This framework embodies the following potential advantages: 

Sequential reasoning dependency. As a type of multi-paradigm reasoning, CoR is not a mere accumulation of isolated steps; instead, it represents a coherent, interconnected process. The output from an earlier paradigm functions not only as input for a subsequent paradigm but also informs the foundational information for its reasoning. For instance, knowledge derived from a natural language paradigm guides subsequent algorithmic paradigms, thereby enhancing the efficiency and accuracy of code generation based on detailed natural language reasoning. This mechanism also provides substantial context from prior paradigms to later paradigms. It operates as a form of preliminary rehearsal where subsequent paradigms can follow correct reasoning steps or rectify incorrect ones. 

A novel direction for test-time scaling emerges. We observe that the achievable improvements of single-paradigm reasoning are increasingly constrained by various search methods applied to the solution space. For instance, DeepSeek-Prover-V1.5-RL + RMaxTS requires 32 6400 reasoning paths to achieve 63.5% few-shot accuracy on the miniF2F benchmark. While extending the number of candidate solution paths seems a natural way to enhance the hit rate for ground truth solutions, the substantial computational effort involved underscores the inherent limitations of single-paradigm reasoning. This observation suggests that scaling up reasoning within a single paradigm during test time is inadequate for addressing complex tasks, such as mathematical problem solving. From a broader perspective, our CoR framework reexamines this challenge by shifting the focus of scaling efforts from reasoning paths to reasoning paradigms, thereby proposing a novel avenue for advancement. 

Expanding the solution space. In multi-paradigm reasoning, the solution space is expanded by considering potential orthogonal relationships between reasoning paradigms. This allows generated solutions to be searched within different paradigms. Moreover, diverse solutions across paradigms can leverage chain relationships in reasoning to mutually inform each other. Therefore, our CoR framework enhances both intra-paradigm and inter-paradigm search capabilities within a large solution space. This diversity increases the likelihood of discovering optimal solutions. 

Journey-based reasoning learning. CoR enhances models’ learning trajectory, allowing them to perform deep reasoning both within specific paradigms and across multiple paradigms. This approach expands the “journey” by introducing the ability to navigate and collaborate across diverse reasoning paradigms. 

Compatibility with existing methods. Since most current approaches are fundamentally based on a single paradigm, CoR can seamlessly integrate these methods within its specified paradigms, enabling them to work synergistically and leverage their strengths. For example, CoT can be incorporated as a specific implementation of the NLR paradigm in CoR. This implies that CoR serves as a platform that facilitates continuous integration and collaboration. 

## B Experiment Details

In this study, we use open-source tools, models, and datasets in compliance with their respective open-source licenses and solely for academic research purposes. 

B.1 The detail of the Universal Text Template As shown in Fig. 7, we apply the designed universal text template on all training samples. 

## B.2 Example Prompts for Dataset Enhancement

To guide LLMs in generating and refining reasoning paradigms during dataset enhancement, we developed specific prompts $( p _ { s } )$ tailored to each seed dataset. These prompts provide structured instructions for the models to follow, ensuring the generation of high-quality and logically consistent reasoning paradigms. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/db771bc9-d734-41ba-ac35-ccfc42738592/2f32f4890ecf5905dfcc3cf90205ca5c38efc70782922859f13867ee5e4511e1.jpg)



Figure 7: The universal text template designed for multi-paradigm reasoning. The contents for each paradigm have been omitted.


<table><tr><td>Dataset</td><td># of Test set</td><td>Avg. Length</td></tr><tr><td>MATH</td><td>5,000</td><td>30.7</td></tr><tr><td>GSM8K</td><td>1,319</td><td>46.3</td></tr><tr><td>AIME2024</td><td>30</td><td>59.2</td></tr><tr><td>AMC2023</td><td>40</td><td>47.2</td></tr><tr><td>MiniF2f</td><td>244</td><td>30.5</td></tr></table>


Table 7: The statistics of the evaluation datasets.


Fig. 8 and Fig. 9 provide examples of such prompts designed for augmenting and refining formal proofs within the Lean 4 theorem prover for the Numina-TIR and Lean-Workbook datasets, respectively. The prompt specifies the input and output format and includes placeholders for the problem statement, informal proof, and corresponding formal proof in Lean 4. It also incorporates $N _ { \mathrm { s h o t } }$ few-shot examples to further guide the model. In our experiments, we follow the common settings (Jiang et al., 2023; Yang et al., 2024; Wang et al., 2025; Zhao and Zhang, 2024) and set $N _ { \mathrm { s h o t } } = 8$ . This structured approach and setting are applied to all datasets to ensure comprehensive coverage and logical consistency across the enhanced dataset. 

## B.3 Details of Training Settings

In all stages of the PPT method, a learning rate of $2 e \mathrm { ~ - ~ } 5$ and a warm-up ratio of 1% are implemented. To enhance computational efficiency, the training process is conducted using distributed optimization with DeepSpeed ZeRO Stage 3 (Rajbhandari et al., 2020) and is combined with Flash-Attention (Dao, 2024). Stage ➀ comprises 3 epochs, whereas the stage ➁ and ➂ are conducted over 4 epochs. A sequence length of 2, 048 tokens is used in Stages ➀ and ➁, which is increased to 4, 096 tokens in stage ➂ to support more complex reasoning tasks. Furthermore, we employ an annealing strategy at the end of stage ➂ with high-quality MPM samples, with the aim of further enhancing model accuracy in complex reasoning tasks. 

## B.4 Benchmarks

We report the statistics of the evaluation datasets in Table 7. 

## B.5 Details of Metrics

This study employs the pass@N metric on the miniF2F benchmark, with the evaluation grounded in a sample budget of N. To ensure a fair comparison of computational cost across different generation schemes, this paper defines the sample budget according to the following rules. 

• For single-pass sampling methods, we define the sample budget N as the total number of proofs generated, with large values of N factorized for ease of comparison to tree search methods. 

• For best-first-search methods, following the notation in Llemma (Azerbayev et al., 2024), we present $N = N _ { \mathrm { { a t t } } } \times N _ { \mathrm { { t a c t } } } \times N _ { \mathrm { { i t e r } } }$ where $N _ { \mathrm { a t t } }$ denotes the number of best-first-search attempts, $N _ { \mathrm { t a c t } }$ denotes the number of tactics generated for each expansion, and $N _ { \mathrm { i t e r } }$ denotes the number of expansion iterations. 

• For tree-based search methods, e.g., RMaxTS (Xin et al., 2024) and HTPS (Lample et al., 2022), we present $N = N _ { \mathrm { a t t } } \times N _ { \mathrm { e x } } .$ where $N _ { \mathrm { a t t } }$ denotes the number of tree search attempts, and $N _ { \mathrm { e x } }$ denotes the number of model generations invoked in tree expansions. 

• For our SMPS, we define the sample budget as $N = N _ { \mathrm { N L R } } \times N _ { \mathrm { S R } }$ , where $N _ { \mathrm { { N L R } } }$ denotes the number of initial semantic reasoning paths, and $N _ { \mathrm { S R } }$ denotes the number of symbolic reasoning paths extended for each other reasoning paradigm. 

## B.6 Experimental Details of MATH and GSM8K benchmarks

For arithmetic-related results, we analyze the number of mathematical samples utilized by the baseline model following the pre-training phase, as illustrated in Fig. 1. Specifically, as shown in Table 8, we present the sample quantity along with the model performance. 

## C Additional Analysis

## C.1 The Risk of Data Leakage

We measure the Levenshtein distance (Miller et al., 2009) between problems in the training dataset and those in the test dataset to mitigate the risk of data leakage. During the experiments, we set a similarity threshold of 0.7 and excluded cases from the training dataset that exceeded this threshold. As a result, approximately 1, 000 cases, accounting for less than 0.6% of our training dataset, were removed to prevent potential leakage. 

<table><tr><td rowspan="2">Model</td><td rowspan="2">SFT Data Size (k)</td><td>MATH</td><td>GSM8k</td><td rowspan="2">Average</td></tr><tr><td>ZS Pass@1</td><td>ZS Pass@1</td></tr><tr><td>WizardMath</td><td>868</td><td>10.7</td><td>54.9</td><td>32.8</td></tr><tr><td>MetaMath-7B</td><td>2,790</td><td>19.8</td><td>66.5</td><td>43.2</td></tr><tr><td>MetaMath-Llemma-7B</td><td>2,790</td><td>30.0</td><td>69.2</td><td>49.6</td></tr><tr><td>MetaMath-Mistral-7B</td><td>2,790</td><td>28.2</td><td>77.7</td><td>53.0</td></tr><tr><td>ToRA</td><td>85</td><td>40.1</td><td>68.8</td><td>54.5</td></tr><tr><td>ToRA-CODE</td><td>85</td><td>44.6</td><td>72.6</td><td>58.6</td></tr><tr><td>NuminaMath-7B-CoT</td><td>859</td><td><eq>54.4^†</eq></td><td><eq>66.6^†</eq></td><td>60.5</td></tr><tr><td>Xwin-Math-7B</td><td>1,440</td><td>40.6</td><td>82.6</td><td>61.6</td></tr><tr><td>DeepSeekMath-Instruct-7B</td><td>776</td><td>46.8</td><td>73.6</td><td>64.2</td></tr><tr><td>NuminaMath-7B-TIR</td><td>931</td><td><eq>55.3^†</eq></td><td><eq>73.6^†</eq></td><td>64.5</td></tr><tr><td>DeepSeekMath-RL-7B</td><td>920</td><td>51.7</td><td>88.2</td><td>70.0</td></tr><tr><td>DART-Math-7B</td><td>1,175</td><td>53.6</td><td>86.6</td><td>70.1</td></tr><tr><td>Qwen2.5-Math-7B-Instruct</td><td>3,026</td><td>83.6</td><td>95.2</td><td>89.4</td></tr><tr><td>CoR-Math-7B</td><td>1,098</td><td>66.7</td><td>88.7</td><td>77.7</td></tr></table>


Table 8: Zero-shot (ZS) performance on the MATH and GSM8K benchmarks for the arithmetic task, with data size in the Supervised Fine-Tuning (SFT) stage. denotes our reported results with the open-sourced model weights. The best results are in bold, and the second-best results are underlined


```txt
Example Prompts for TIR Dataset Enhancement
System Prompt:
You are an expert in Lean 4. Please respond to a math problem by translating the provided informal proof into Lean 4 code. Follow the format provided in the prompt. Please note that the informal proof and the formal proof need to be identical. Follow the format provided in the prompt.
User Input:
Now please translate the formal solution in Lean 4 following the instructions below. Please write the corresponding solution in Lean 4 (indicated by “Formal proof in Lean 4:”) given the “# Problem: ” and “# Informal proof: ”, filling in the “# Formal proof in Lean 4: ” section.
You must respond in the following format:
# Problem: ...
# Informal proof: ...
# Formal proof in Lean 4:

```lean4
(lean 4 code for proving)
...
---
Here are examples you may refer to:
____
N few-shot examples
____
# Problem: {problem}
# Informal proof: {informal_proof}
# Formal proof in Lean 4:
```lean4
...
... 
```


Figure 8: Example prompts for Numina-TIR dataset enhancement. The few-shot examples provided for the LLM have been omitted.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/db771bc9-d734-41ba-ac35-ccfc42738592/b66499b1bc09e43980f4de9bf5eab3a038e0877ee9d9029d54ff501dfd608ea0.jpg)



Figure 9: Example pompts for Lean-Workbook dataset enhancement. The few-shot examples provided for the LLM have been omitted.


<table><tr><td rowspan="2">Model</td><td colspan="2">MATH</td><td colspan="2">GSM8k</td></tr><tr><td>Pass@1</td><td>Maj@8</td><td>Pass@1</td><td>Maj@8</td></tr><tr><td>Llama-3.1-8B</td><td>58.18</td><td>59.46</td><td>84.00</td><td>87.26</td></tr><tr><td>DeepSeekMath-Base-7B</td><td>66.74</td><td>71.70</td><td>88.70</td><td>91.43</td></tr></table>


Table 9: Zero-shot results of different evaluation strategies across MATH and GSM8k benchmarks.


## C.2 Different Evaluation Strategies on Arithmetic Benchmarks

Table 9 explores the impact of various evaluation strategies on the CoR framework. The majority vote strategy benefits the CoR framework. Specifically, a majority vote strategy with 8 candidate samples on GSM8K can exceed GPT-4o’s Pass@1 performance (90.5%). 

## C.3 Ethical Considerations

In this work, we present a method that achieves robust zero-shot multi-paradigm reasoning capabilities. However, this comes with potential risks, which include two primary social and ethical concerns: 1) the risk of tool misuse, and 2) preexisting risks embedded in the backbone model parameters. The CoR framework supports various reasoning paradigms, some of which are deeply integrated with specific tools. For instance, the AR paradigm may involve unvetted Python code libraries. Therefore, we recommend performing predeployment audits on tools that may be utilized by the model. Furthermore, our method is compatible with existing LLMs, and several studies have highlighted the presence of biased behaviors in models like LLaMa (Kong et al., 2024). To mitigate ethical risks, we encourage the use of risk-free language models during deployment, which can reduce potential harm. Recent research suggests incorporating ethical alignment processes, such as filtering training data or generated content, to minimize unnecessary risks (Yu et al., 2023). Overall, we advocate for open discussions regarding the use of our framework, as a transparent and public environment can reduce the likelihood of misuse. 

## D Case Studies

## D.1 Cases of Instruction-free Reasoning

In the absence of explicit instructions, as illustrated in Fig. 10, the model generally adopts a paradigm sequence that aligns with the task types present in the training data, as described by MPM. The model transitions between paradigms autonomously. For example, during a computational task, the model initiates reasoning with the paradigm NLR, proceeds to SR for logical verification, and ultimately employs AR to achieve enhanced computational performance. 

## D.2 Qualitative Analysis of Error Cases

We now present a preliminary error pattern analysis for CoR-Math-7B across two tasks: arithmetic computation and theorem proving. 

In arithmetic computation, an analysis of 200 randomly sampled error cases reveals four primary types of errors: comprehension errors or incorrect premise assumptions (33 cases), logical fallacies (26 cases), computational errors (56 cases), and incomplete proofs or code syntax errors, which were most prevalent in the AR/SR paradigms (85 cases). Notably, cross-paradigm self-correction behaviors were observed. For example, AR rectified erroneous calculations initially produced by NLR, and SR supplemented incomplete logical steps that were missing in NLR outputs. In theorem proving, a separate analysis of 50 randomly sampled error cases revealed the same four error types, with the following distribution: 2 comprehension errors, 6 logical fallacies, 9 computational errors, and 33 incomplete proofs or syntax errors. Cross-paradigm self-correction behaviors were again consistently observed. As shown in Fig. 11 and Fig. 12, we provide an example of error cases and an example of self-correction behavior. 

We attribute these errors to the following potential causes. Despite MPMs systematic integration of multi-paradigm reasoning paths and PPTs paradigm-progressive training as discussed in Sec. 3.2 and Sec. 3.4 to mitigate distribution shifts, the overwhelming prevalence of NLR examples in the pretraining data —particularly the scarcity of SR-specific instances —has resulted in persistent competency gaps across paradigms. Additionally, CoR inherits the base LLMs fixed-context window, which introduces the risk of truncating long reasoning chains or omitting key steps, leading to incomplete outputs. Furthermore, the observed self-correction capabilities can be attributed to the complementarity of reasoning paradigms within CoR, where one reasoning paradigm can rectify errors made by another, thus validating the effectiveness of our multi-paradigm approach. Additionally, feedback from multi-step reasoning validation allows subsequent paradigms to verify earlier outputs, facilitating error detection and correction throughout the reasoning process. 

```txt
Examples of Instruction-free Reasoning Generated by CoR

Problem:
Cities A and B are 45 miles apart. Alicia lives in A and Beth lives in B. Alicia bikes towards B at 18 miles per hour. Leaving at the same time, Beth bikes toward A at 12 miles per hour. How many miles from City A will they be when they meet?

Solution:
Let's go through this step-by-step in natural language. To determine how many miles from City A Alicia and Beth will meet, we can use the concept of relative speed and the distance between their homes.
1. Calculate the relative speed between Alicia and Beth:
- Alicia's speed: 18 miles per hour
- Beth's speed: 12 miles per hour
- Relative speed = 18 + 12 miles per hour
2. Calculate the time it takes for them to meet:
- Distance between City A and City B: 45 miles
- Time to meet = Distance ÷ Relative speed
3. Calculate the distance each person covers before meeting:
- Distance covered by Alicia = Speed of Alicia × Time to meet
- Distance covered by Beth = Speed of Beth × Time to meet.

Next, let's write the corresponding formal proof in Lean 4 to prove this.
# Formal proof in Lean 4:

```lean4
def distance_AB := 45 -- miles
def speed_Alicia := 18 -- miles per hour
def speed_Beth := 12 -- miles per hour
-- Calculate the relative speed
def relative_speed := speed_Alicia + speed_Beth -- miles per hour
-- Calculate the time to meet
def time_to_meet := distance_AB/.toFloat / relative_speed/.toFloat -- in hours
-- Calculate the distance each person covers before meeting
def distance_covered_by_Alicia := speed_Alicia/.toFloat * time_to_meet
def distance_covered_by_Beth := speed_Beth/.toFloat * time_to_meet
-- Calculate how many miles from City A they will be when they meet
def distance_from_A_when_meet := distance_covered_by_Alicia
#eval distance_from_A_when_meet -- returns 27.0

```lean_output
27.000000

```
Let's use Python code to perform these calculations.
# Code in Python:

```python
import sympy as sp
# Given values
distance_AB = 45 # miles
speed_Alicia = 18 # miles per hour
speed_Beth = 12 # miles per hour
# Calculate the relative speed
relative_speed = speed_Alicia + speed_Beth # miles per hour
# Calculate the time to meet
time_to_meet = distance_AB / relative_speed # in hours
# Calculate the distance each person covers before meeting
distance_covered_by_Alicia = speed_Alicia * time_to_meet
distance_covered_by_Beth = speed_Beth * time_to_meet
# Calculate how many miles from City A they will be when they meet
distance_from_A_when_meet = distance_covered_by_Alicia
print(distance_from_A_when_meet)

```output
27.0

Alicia and Beth will meet boxed27 miles from City A. 
```


Figure 10: A case of the reasoning trajectory produced by CoR in an instruction-free setting.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/db771bc9-d734-41ba-ac35-ccfc42738592/bf6d148d2c923a8dc22bbbd293610d73962c9032199e8d40eedc709e9e096e9f.jpg)



Figure 11: A proof task failure case caused by incomplete proofs or code syntax errors.


Examples of Instruction-free Reasoning Generated by CoR

Problem:
Let $b \geq 2$ be an integer. Call a positive integer $n$ $b$ -beautiful if it has exactly two digits when expressed in base $b$ , and these two digits sum to $\sqrt{n}$ .

For example, 81 is 13-eautiful because $81 = \underline{63}_{13}$ and $6 + 3 = \sqrt{81}$ .

Find the least integer $b \geq 2$ for which there are more than ten $b$ -beautiful integers.

Solution:
Let's go through this step-by-step in natural language. To solve this problem, we need to find the smallest integer $b \geq 2$ such that there are more than ten $b$ -beautiful integers.

1. Understanding the Problem:
- We need to identify $b$ -beautiful integers, which have exactly two digits in base $b$ .
- These integers $n$ must satisfy the condition that the sum of their digits equals $\sqrt{n}$ .

...

6. Conclusion:
- The least integer $b \geq 2$ for which there are more than ten $b$ -beautiful integers is $b = 13$ .

Therefore, the smallest $b$ such that there are more than ten $b$ -beautiful integers is $b = 13$ .

Next, let's write the corresponding formal proof in Lean 4 to prove this.

# Formal proof in Lean 4:

```lean4
theorem lean_workbook_plus_1377 (b : ) (hb : 2 b)
(h : (n, (Nat.sqrt n = n Nat.digits b n = [n / b, n % b]))
    ñ (n, (Nat.sqrt n = n Nat.digits b n = [n / b, n % b])) : 13 b :=
    by
    contrapose! h
    interval_cases b <;> norm_num [Nat.digits_eq_of_eq_mul_left, Nat.digits_eq_of_eq_mul_right]
``````
```lean_output
27.000000
``````
Let's use Python code to perform these calculations.
# Code in Python:

```python
import sympy as sp
import math
def is_b_beautiful(n, b):
\"\"Check if a number n is b-beautiful.\\"\""
digits = [n // b, n % b]
return sum(digits) == math.sqrt(n)
def count_b_beautiful_numbers(b):
\"\"Count the number of b-beautiful numbers for a given b.\\"\""
count = 0
for n in range(1, b * b):
if is_b_beautiful(n, b):
count += 1
return count
def find_smallest_b_with_more_than_ten_beautiful_numbers():
\"\"Find the smallest b >= 2 such that there are more than ten b-beautiful numbers.\\"\""
b = 2
while True:
if count_b_beautiful_numbers(b) > 10:
return b
b += 1
# Find the smallest b
smallest_b = find_smallest_b_with_more_than_ten_beautiful_numbers()
print(smallest_b)
``````
```output
211
``````
The smallest integer $b \geq 2$ for which there are more than ten $b$ -beautiful integers is $b = 211$ . Thus, the answer is 211. 


Figure 12: A case of self-correction behaviors. Some lengthy content is omitted with "..." for brevity.
