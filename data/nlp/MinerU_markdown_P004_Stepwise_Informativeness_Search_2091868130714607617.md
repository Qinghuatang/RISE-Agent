# Stepwise Informativeness Search for Improving LLM Reasoning

Siyuan Wang<sup>1</sup>*, Enda Zhao<sup>2</sup>*, Xiang Ren<sup>1</sup> <sup>1</sup>University of Southern California, <sup>2</sup>Purdue University. sw_641@usc.edu; zhao1462@purdue.edu 

## Abstract

Advances in Large Language Models (LLMs) have improved multi-step reasoning by generating free-text rationales, but these models tend to lose focus over the middle of long contexts. This raises concerns that as reasoning progresses, LLMs may overlook information in earlier steps when decoding subsequent steps, leading to unreliable and redundant rationales. To address this, we propose guiding LLMs to generate more accurate and concise rationales by (1) proactively referencing information from underutilized prior steps, and (2) minimizing redundant information between new and existing steps. We introduce stepwise informativeness search, an inference-time tree search framework incorporating two selection heuristics: grounding-guided selection which prioritizes steps paying higher attention over underutilized steps; and novelty-guided selection which encourages steps with novel conclusions. We further utilize a self-grounding strategy that prompts LLMs to explicitly reference relevant prior steps as premises before deduction at each step, mitigating distraction from irrelevant content. Experiments on five reasoning datasets across five LLMs show the effectiveness and efficiency of our approach to improve reasoning with reduced errors and redundancy <sup>1</sup>. 

## 1 Introduction

Large Language Models (LLMs) (OpenAI, 2023; Team et al., 2023) have shown remarkable performance in reasoning tasks through Chain-of-Thought (CoT) (Wei et al., 2022) prompting, which elicits step-by-step rationales to derive answers. However, complex multi-step reasoning remains challenging, particularly for smaller-scale models (Dziri et al., 2024). Recent advances in treesearch algorithms (Wang et al., 2024b; Yao et al., 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/afc6774d-7097-4258-889a-9f6c3bfcdfad/8c10a52d64bc2e1fccaa0de861d4bf3fd0e118265ccb6a0dd450e2d1470881b6.jpg)



Figure 1: An example illustrating LLMs’ difficulty in referencing early-step information (e.g., underutilization of [Step-2,4,5,6]), and the inclusion of redundant steps (e.g., repeated conclusions in [Step-5, 7]). The rightward red arrow indicates the focus is on generating [Step-8] with [Step 1-7] have been generated.


2024; Zhang et al., 2024b) improve this by generating step-level candidates <sup>2</sup> and using scoring mechanisms to select the most promising ones iteratively, thereby improving overall generated rationales. However, they typically rely on domainspecific reward models or more powerful LLMs to assess candidate validity (Luo et al., 2024). 

Moreover, LLMs tend to focus on leading and recent contexts while losing attention in the middle (Hsieh et al., 2024). As reasoning progresses, this causes difficulty in referencing useful intermediate conclusions from earlier steps when decoding subsequent ones, leading to unreliable and redundant rationales. For example, in Fig. 1, [Step 2,4,5,6] provide useful information for deriving the final answer but are not effectively utilized. This results in redundant steps (e.g., [Step-7] and [Step-5] have repeated conclusions) and incorrect answer (e.g., [Step-8]). Consequently, LLMs risk getting trapped in repetitive reasoning loops (Chen et al., 2024) and generating unnecessarily lengthy rationales, increasing the likelihood of cumulative errors (Furuta et al., 2024). 

To address this, we propose to guide LLMs in generating more accurate and concise step-by-step rationales by (1) proactively referencing intermediate conclusions generated from underutilized steps, and (2) minimizing redundancy between new and existing steps. With higher-quality rationales generated, we can improve answer accuracy and reduce decoding costs. Underutilized steps are those whose intermediate conclusions have been less frequently referenced before the current step, suggesting untapped potential to offer useful information for subsequence reasoning. Meanwhile, reducing redundancy across steps can contribute novel information, enabling more efficient exploration of the reasoning space toward final answers. 

We introduce stepwise informativeness search, an inference-time tree search framework that prioritizes steps based on informativeness, either by leveraging underutilized steps or generating novel content. The framework follows a stepwise beam search paradigm (Xie et al., 2024), generating multiple candidate steps at each iteration. Based on standard cumulative step-level likelihood, it incorporates two heuristics to guide candidate selection. (1) Grounding-guided selection identifies underutilized steps by computing each step’s reference degree so far to estimate its information gain for subsequent reasoning. As LLMs naturally assign higher attention to grounding context (Zhang et al., 2023), we prioritize candidate steps with higher attention scores over underutilized steps. (2) Noveltyguided selection ranks candidates based on the novelty of their intermediate conclusions relative to prior steps. A trigram-based similarity measure filters out highly similar candidates for simplicity. 

We empirically validate that encouraging the grounding of underutilized steps improves reasoning, though LLMs inevitably generate unnecessary steps during reasoning. Specifically, for LLaMA-3- 8B self-generated rationales on CLUTRR (Sinha et al., 2019) problems requiring 8 steps, the model exhibits significantly higher accuracy 57.89% when all steps build upon at least one underutilized step, compared to only 22.39% when they do not. To further prevent grounding on underutilized yet irrelevant steps, we introduce a self-grounding strategy that elicits LLMs’ ability to identify relevant prior steps to provide premises before each deduction. 

This process enables connecting with distant underutilized steps by first specifying their step numbers, and reinforcing the generation of well-supported new steps through explicit grounding. We implement our informativeness search framework both with and without self-grounding strategy. 

Experiments on four multi-step reasoning and one commonsense reasoning datasets validate the effectiveness of the informativeness search framework and self-grounding strategy across five LLMs of varying families and scales. Overall, our framework can generate more effective solutions with improved accuracy and reduced tokens. Moreover, the two heuristics leverage the model’s internal outputs and attention to guide step search, making the approach domain-agnostic and efficient by eliminating the need for exhaustive interactions with external scorers or self-evaluation during decoding. 

## 2 Stepwise Beam Search for Reasoning

In this work, we formulate multi-step reasoning as a stepwise beam search process considering its generation parallelizability can accelerates search process (Xie et al., 2024). This contrasts with another common tree-search practice, Monte Carlo Tree Search (MCTS) methods (Feng et al., 2023; Zhang et al., 2024a), which involve extensive rollout simulations and are computationally expensive. 

Specifically, at each iteration, the model generates a set of reasoning steps in parallel, each delimited by a special end-of-line token $" / \mathrm { n } ^ { \prime \prime }$ . A beam of the top N steps are selected according to various criteria, where N is the beam size. Unlike step-level evaluation, stepwise beam search ranks candidates by their cumulative rewards (e.g., likelihood) across the sequence generated so far. 

Formally, the generation of a reasoning sequence $R = [ s _ { 1 } , s _ { 2 } , . . . , s _ { T } ]$ with $T$ steps is formulated as 

$$
P (R = s _ {1: T} | x) = \prod_ {t} P (s _ {t} | s _ {1: t - 1}, x),
$$

where $s _ { t }$ is the t-th step and $x$ is the input query. Stepwise generation and selection are performed with beam size N and sample size k as follows: starting with N sequences at step $t - 1$ , it generates k continuations from $P ( s _ { t } | s _ { 1 : t - 1 } , x )$ ) for each sequence $s _ { 1 : t - 1 }$ , forming a candidate set $C _ { t }$ containing Nk reasoning chains of length t. The top N sequences are then selected based on a scoring criteria ${ \phi ( C _ { t } , \gamma ( \cdot ) ) = \{ s ^ { 1 } , s ^ { 2 } , \ldots , s ^ { N } \} }$ . ϕ is the selection function $( { \bf e . g . } , t o p k ( \cdot ) )$ and $\gamma ( s _ { 1 : t } )$ evaluates the sequence so far $s _ { 1 : t }$ . Initially, given only an input x, we generate $N k$ candidates. 

A standard scoring criteria is the cumulative likelihood of a sequence, defined as: $\gamma _ { L } ( s _ { 1 : t } )$ = log $\prod _ { t } P ( s _ { t } | s _ { 1 : t - 1 } , x )$ . Alternative scoring functions $\gamma ( s _ { 1 : t } )$ are employed in self-evaluation (Xie et al., 2024) and deductive beam search (Zhu et al., 2024). The former prompts the backend LLM to provide a correctness score $\gamma _ { c } ( s _ { t } )$ to assess whether $s _ { t }$ is correct given $s _ { 1 : t - 1 }$ , which is then combined with likelihood: $\gamma _ { E } ( s _ { 1 : t } ) \ = $ log $\prod _ { t } P ( s _ { t } | s _ { 1 : t - 1 } , x ) \gamma _ { c } ( s _ { t } )$ . The latter trains an external deductive verifier $f$ to assess whether each step $s _ { t }$ is logically entailed by previous contexts, and replaces the sequence likelihood with a cumulative deductive score: $\gamma _ { D } ( s _ { 1 : t } ) \ =$ f(entails $s _ { t } , s _ { 1 : t - 1 } , x )$ 

While these methods improve performance, they require additional annotations or prompts to obtain domain-specific scoring models. They also incur interaction overhead by waiting for scorer response at each decoding step, yet failing to address aforementioned grounding and redundancy challenges. 

## 3 Informativeness Search Framework

Unlike iteration-based scoring functions described above, we introduce stepwise informativeness search framework with two scoring heuristics that utilize model’s intrinsic outputs and attention scores. This reduces reliance on off-the-shelf scorers and iterative interactions during decoding. It prioritizes steps based on informativeness, assessed by grounding-guided and novelty-guided heuristics that determine whether new decoded steps ground on underutilized steps and generate novel content. 

## 3.1 Grounding-Guided Selection

To ground each deduction upon underutilized steps to maximally leverage useful information, we design an algorithm to identify underutilized ones among all prior steps. The candidate sequences, denoted as $\bar { C } _ { t } = \{ s _ { 1 : t } ^ { 1 } , s _ { 1 : t } ^ { 2 } , . . . , s _ { 1 : t } ^ { N k } \}$ , are then evaluated and selected based on whether each current step $s _ { t } ^ { i }$ is well derived from corresponding underutilized steps. Further empirical analysis on the correlation between underutilized steps grounding and improved reasoning is provided in Appendix C. 

Identifying Underutilized Steps At each reasoning step, underutilized steps are those referenced less frequently up to that point, offering higher untapped potential for contributing information to subsequent reasoning. At the current step $s _ { t }$ , the immediately preceding step $s t { - 1 }$ is by default considered underutilized since it represents the most recent addition to the reasoning path. For additional underutilized steps, we perform a backward traversal from step $s _ { t - 2 } ~ \mathrm { t o } ~ s _ { 1 } ,$ , calculating the reference degree of each step to assess its information gain to subsequent reasoning. 

Specifically, for each prior step $s _ { j } \in$ $\{ s _ { t - 2 } , . . . , s _ { 2 } , s _ { 1 } \}$ , we first extract its intermediate conclusion $c _ { j }$ by segmenting it using special clause delimiters $( \mathrm { e . g . , \tilde { \Omega } { s o } ^ { \mathrm { 3 9 , \tilde { \Omega } } } , \tilde { \Omega } t h u s ^ { \mathrm { 3 9 } } }$ and commas). We then compare $c _ { j }$ with each subsequent step $s _ { m } \in \{ s _ { j + 1 } , . . . , s _ { t - 1 } \}$ before the current step using a trigram-based similarity measure. The information gain of $s _ { j }$ is computed as follows: 

$$
\operatorname{InfoGain} (s _ {j}) = 1 - \max _ {m \in j + 1, \dots , t - 1} \operatorname{Sim} _ {\mathrm{tri}} (c _ {j}, s _ {m})
$$

We classify a prior step as underutilized if its information gain exceeds a predefined threshold $\tau .$ . The set of underutilized steps at step t is: 

$$
\begin{array}{c} \mathcal {I} _ {t} = \left\{s _ {t - 1} \right\} \cup \left\{s _ {j} \mid \text { InfoGain } (s _ {j}) > \tau \right\}, \\ j \in \left\{1, \ldots , t - 2 \right\} \end{array}
$$

Grounding on Underutilized Steps After identifying the set of underutilized steps $\mathcal { T } _ { t } ^ { i }$ for each candidate sequence $s _ { 1 : t } ^ { i }$ in the candidate set $C _ { t } =$ $\{ s ^ { 1 } , s ^ { 2 } , \ldots , \bar { s ^ { N k } } \}$ (with subscripts omitted for simplicity), we prioritize candidates that more effectively ground their reasoning in $s _ { t } ^ { i }$ upon their respective underutilized steps. 

LLMs typically assign higher attention scores to their grounding context (Zhang et al., 2023). We leverage attention to assess how well each candidate focuses on its identified underutilized steps $\mathcal { T } _ { t } ^ { i }$ when constructing step $s _ { t } ^ { i } .$ . Given that there may be multiple underutilized steps $( | \mathcal { T } _ { t } ^ { i } | > 2 )$ , we do not enforce grounding on all of them. Instead, we apply a soft attention mechanism that prioritizes candidates assigning higher weights to a subset of these steps. Specifically, we compute the grounding score of $s _ { t } ^ { i }$ over $\mathcal { T } _ { t } ^ { i }$ as $\gamma _ { a } ( s _ { t } ^ { i } )$ by applying mean pooling across all tokens in $s _ { t } ^ { i }$ and only highly attended tokens within $\mathcal { T } _ { t } ^ { i }$ . Detailed calculations are provided in Appendix D. This approach maintains robustness to irrelevant or noisy steps while capturing valuable signals from underutilized steps. 

We then integrate this attention-based measure into the original cumulative likelihood scoring function to obtain an grounding-enhanced score: 

$$
\gamma_ {G} (s _ {1: t}) = \gamma_ {L} (s _ {1: t}) + \alpha \cdot \gamma_ {a} (s _ {t})
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/afc6774d-7097-4258-889a-9f6c3bfcdfad/6b01d13620132a010078b4c2488abc1395d5eed694e0690063b5c3b547884dd9.jpg)



Figure 2: Upper: Overview of our informativeness search framework illustrated with beam size of 1. Green blocks indicate selected steps, red-crossed blocks denote discarded steps, and gray blocks contain incorrect deductions. The orange block ([Step-7]) highlights a low-quality step that only can be filter by our method. Bottom: Evaluation comparison of [Step-7]. Previous methods accept [Step-7] as logically correct at both step and sequence levels, whereas our framework filters it out based on its low novelty and poor grounding on underutilized steps.


where $\begin{array} { r } { \gamma _ { L } ( s _ { 1 : t } ) = \log \prod _ { t } P ( s _ { t } | s _ { 1 : t - 1 } , x ) } \end{array}$ and α is a weighted hyperparameter. Then N candidates are selected from ${ C _ { t } = \{ s ^ { 1 } , s ^ { 2 } , \ldots , s ^ { N k } \} }$ with the highest $\gamma _ { G } ( s _ { 1 : t } )$ . We validate this attentionbased operation in Sec. 5.3 by analyzing the consistency between highly attended content and actual grounded information. 

## 3.2 Novelty-Guided Selection

To reduce redundancy across multiple intermediate steps, we assess the conclusion novelty of each newly generated step $s _ { t } ^ { i }$ in a candidate sequence $s _ { 1 : t } ^ { i } .$ , and select candidates with higher novelty. We extract intermediate conclusions from $s _ { t } ^ { i }$ and all its prior steps $\{ s _ { 1 } ^ { i } , \ldots , s _ { t - 1 } ^ { i } \}$ by segmenting the corresponding sentences using special clause delimiters $( \mathrm { e . g . , \hbar ^ { 6 } s o ^ { 3 } } ,$ , “thus” and commas), forming a set of conclusions $\{ c _ { 1 } ^ { i } , \ldots , c _ { t - 1 } ^ { i } , c _ { t } ^ { i } \}$ . We then calculate the trigram-based similarity between the newly generated conclusion $c _ { t } ^ { i }$ and all preceding conclusions $\{ c _ { 1 } ^ { i } , \ldots , c _ { t - 1 } ^ { i } \}$ . The novelty score of $s _ { t } ^ { i }$ is then obtained as follows: 

$$
N (s _ {t} ^ {i}) = 1 - \max _ {j \in 1, \dots , t - 1} \mathrm{Sim} _ {\mathrm{tri}} (c _ {t} ^ {i}, c _ {j} ^ {i})
$$

where $\mathrm { S i m } _ { \mathrm { t r i } } ( \cdot , \cdot )$ measures trigram-based similarity with justification discussed in Appendix E. We also explore embedding-based similarity measure in Sec. 5.4. To incorporate novelty into candidate selection, we calibrate the groundingenhanced scoring function with novelty score. At step t, candidates with low-novelty conclusions (i.e., $N ( s _ { t } ) \leq \theta )$ are filtered out, retaining only diverse and meaningful candidates. The adjusted scoring function is defined as below, where θ is a predefined threshold. 

$$
\gamma_ {N} (s _ {1: t}) = \left\{ \begin{array}{l l} \gamma_ {G} (s _ {1: t}), & \text { if } N (s _ {t}) > \theta , \\ - 1 0 0, & \text { otherwise }. \end{array} \right.
$$

## 3.3 Self-Grounding Strategy

To handle irrelevant steps that may arise during reasoning and prevent grounding-guided selection from focusing on irrelevant prior steps, especially when contexts contain distracting information, we introduce a self-grounding strategy. This approach leverages LLMs’ inherent ability to anchor reasoning in relevant prior information, either from prior steps or the input query, that serve as necessary premises for each new deduction. The strategy explicitly prompts LLMs to reason step by step, structuring each step in the format: “[Step-i] From <source>, <deduction>.” where “<source>” refers to either relevant prior steps or the input query that provide premises for deducing new conclusions in “<deduction>”. For example, “[Step-1] From Query, we know ...”, “[Step-2] From Step-1 and Query, we know ...” and “[Step-3] From Step-1 and Step-2, because ...”. This explicit stepgrounding process ensures that each new step directly builds upon established information, maintaining logical coherence while minimizing irrelevant or unsupported conclusions. Moreover, explicitly referencing step numbers facilitates connections with distant underutilized steps. Further details on the prompts and few-shot demonstrations are provided in Appendix B. 

<table><tr><td>Models</td><td>Methods</td><td>FOLIO</td><td>ProofWriter</td><td>MMLU-Pro</td><td>GPQA-Diamond</td><td>Avg.</td></tr><tr><td rowspan="10">Llama3.2-3B-Instruct</td><td>Few-shot CoT</td><td>38.73%</td><td>40.00%</td><td>28.57%</td><td>21.72%</td><td>32.25%</td></tr><tr><td>Self-Grounding CoT</td><td>45.59%</td><td>43.33%</td><td>28.57%</td><td>22.73%</td><td>35.06%</td></tr><tr><td>Best-of-N</td><td>45.59%</td><td>37.00%</td><td>30.00%</td><td>22.73%</td><td>33.83%</td></tr><tr><td>Self-Consistency</td><td>46.57%</td><td>47.67%</td><td>29.64%</td><td>22.73%</td><td>36.65%</td></tr><tr><td>Tree-of-Thought</td><td>44.12%</td><td>44.17%</td><td>26.43%</td><td>22.73%</td><td>34.36%</td></tr><tr><td>Self-Eval Beam Search</td><td>45.10%</td><td>47.00%</td><td>30.71%</td><td>19.19%</td><td>35.50%</td></tr><tr><td>Deductive Beam Search</td><td>48.04%</td><td>38.17%</td><td>25.71%</td><td>24.75%</td><td>34.17%</td></tr><tr><td>MCTS + Math-PRM</td><td>/</td><td>/</td><td>26.07%</td><td>22.22%</td><td>/</td></tr><tr><td>Informativeness Search</td><td>46.57%</td><td>50.33%</td><td>33.57%</td><td>27.27%</td><td>39.44%</td></tr><tr><td>Informativeness Search (w/ SG)</td><td>51.96%</td><td>53.67%</td><td>33.93%</td><td>24.24%</td><td>40.95%</td></tr><tr><td rowspan="10">Llama3-8B-Instruct</td><td>Few-shot CoT</td><td>54.90%</td><td>55.33%</td><td>37.50%</td><td>29.29%</td><td>44.25%</td></tr><tr><td>Self-Grounding CoT</td><td>55.39%</td><td>57.00%</td><td>38.57%</td><td>30.30%</td><td>45.32%</td></tr><tr><td>Best-of-N</td><td>56.86%</td><td>50.00%</td><td>39.29%</td><td>30.30%</td><td>44.11%</td></tr><tr><td>Self-Consistency</td><td>57.84%</td><td>60.17%</td><td>39.29%</td><td>31.31%</td><td>47.15%</td></tr><tr><td>Tree-of-Thought</td><td>55.88%</td><td>53.33%</td><td>39.29%</td><td>27.78%</td><td>44.07%</td></tr><tr><td>Self-Eval Beam Search</td><td>59.31%</td><td>56.17%</td><td>35.00%</td><td>29.80%</td><td>45.07%</td></tr><tr><td>Deductive Beam Search</td><td>54.90%</td><td>48.83%</td><td>37.50%</td><td>27.78%</td><td>42.25%</td></tr><tr><td>MCTS + Math-PRM</td><td>/</td><td>/</td><td>27.14%</td><td>28.28%</td><td>/</td></tr><tr><td>Informativeness Search</td><td>58.33%</td><td>61.33%</td><td>40.00%</td><td>33.33%</td><td>48.25%</td></tr><tr><td>Informativeness Search (w/ SG)</td><td>59.80%</td><td>62.00%</td><td>40.71%</td><td>35.35%</td><td>49.46%</td></tr></table>


Table 1: Experimental results (accuracy %) of different methods on Llama3.2-3B-Instruct and LLaMA3-8B-Instruct. SG denotes the Self-Grounding strategy. Shaded rows present results from our proposed method.


## 4 Experiments

## 4.1 Setup

Baselines We evaluate against both sequencelevel CoT methods and step-level search methods. Sequence-level methods include: (1) Few-shot CoT (Wei et al., 2022) performs step-by-step reasoning. (2) Self-Grounding CoT is our proposed self-grounding strategy without search. (3) Best-of-N (Lightman et al., 2023) samples Nk rationales and selects the best via LLM self-evaluation as we lack general reward models for diverse tasks. (4) Self-Consistency (Wang et al., 2022) samples Nk rationales and uses majority voting for the final answer. Step-level methods include: (5) Tree-ofthought (Yao et al., 2024) performs breadth-first tree search with self-evaluation at each step. (6) Self-Eval Beam Search (Xie et al., 2024) and (7) Deductive Beam Search (Zhu et al., 2024) both use stepwise beam search, with the former relying on self-evaluation and the latter on deductive scoring trained on synthesized datasets. (8) MCTS (Zhang et al., 2024a) where we use the minimum score across all steps from Qwen2.5-Math-PRM-7B (Zhang et al., 2025) to evaluate simulated solutions. As this is a mathematical PRM, we report MCTS results only on MMLU-Pro and GPQA-Diamond. We evaluate our informativeness search with and without the self-grounding (SG) strategy. 

Implementation Details We evaluate our framework on four multi-step reasoning datasets: FO-LIO (Han et al., 2022), ProofWriter (Tafjord et al., 2020), MMLU-Pro (Wang et al., 2024c) and GPQA (Rein et al., 2023). We mainly evaluate our method and baselines on Llama3.2-3B-Instruct and Llama3-8B-Instruct, using a two-shot prompting strategy with a 1024-token generation limit. We set N = 3 and k = 2 for all stepwise beam search methods. The parameter α is set to 2 and the threshold τ to 0.7. θ is set to 0.5 for FOLIO and ProofWriter, 0.4 for MMLU-Pro and GPQA-Diamond. Further details and search configurations are provided in Appendix A. 

## 4.2 Main Results

Table 1 presents the overall performance comparison across four benchmark datasets. Our method consistently outperforms all baseline methods across both deductive and diverse reasoning datasets when implemented with either Llama3.2- 3B-Instruct or Llama3-8B-Instruct. This demonstrates the general superiority of our informativeness search framework and self-grounding strategy. Notably, our method yields more substantial improvements on Llama3.2-3B-Instruct, suggesting its particular effectiveness in enhancing reasoning for lower-performing models. Additionally, self-grounding further enhances informativeness search, except when using Llama3.2-3B-Instruct on GPQA-Diamond. We attribute this to Llama3.2- 3B-Instruct’s inability to perform self-grounding effectively for the challenging GPQA-Diamond task. Step-level methods like tree-of-thought, deductive beam search and MCTS show moderate performance due to their reliance on specialized reward model or verifiers, limiting their generalizability. In contrast, informativeness search is broadly applicable without requiring task-specific customization. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/afc6774d-7097-4258-889a-9f6c3bfcdfad/8471492b143d89fa075e47419eac8366cdb9069ef8321fbbed17048ae2dd0d61.jpg)



Figure 3: Accuracy and average token count (Avg. # Tokens) of final predicted rationales using different methods on Llama3.2-3B-Instruct.


## 4.3 Efficiency Analysis

Average Rationale Length We analyze the average token count of final predicted rationales using different methods on Llama3.2-3B-Instruct to examine the relationship between rationale length and accuracy. As shown in Figure 3, our method generates shorter rationales with fewer tokens than fewshot CoT and stepwise beam search while achieving higher accuracy, both with and without the self-grounding strategy. Notably, our approach exhibits greater token reduction in deductive reasoning, correlating with more significant performance improvements. We attribute this to our informativeness search framework can effectively reduce redundancy by combining grounding-guided and novelty-guided selection. This minimizes cumulative errors and prevents circular reasoning loops, ultimately leading to better performance. 

Total Inference Cost We further analyze the total inference cost, including candidate step generation and evaluation throughout the search process for all methods involving stepwise beam search. As shown in Table 2, we compare token consumption, inference time and memory usage across all methods, evaluated consistently on the same NVIDIA RTX A6000 GPU with batch size 1. Our method effectively reduces token usage and inference time compared to the baseline and other beam search methods while maintaining comparable memory usage, exhibiting superior efficiency. The high cost of Self-Eval and deductive beam search stems from additional interactions to obtain evaluation feedback after each step. Moreover, deductive beam search requires additional computational resources for training a domain-specific deductive verifier. 

<table><tr><td>Search Method</td><td>Token Num.</td><td>Inference Time</td><td>Memory</td></tr><tr><td>Self-Eval</td><td>75.0×K</td><td>589 min</td><td>7348 MiB</td></tr><tr><td>Deductive</td><td>46.3×K</td><td>346 min</td><td>9789 MiB</td></tr><tr><td>Stepwise</td><td>2.40×K</td><td>123 min</td><td>10918 MiB</td></tr><tr><td>Stepwise w/ SG</td><td>2.48×K</td><td>135 min</td><td>11052 MiB</td></tr><tr><td>Informativeness</td><td>1.89×K</td><td>103 min</td><td>10895 MiB</td></tr><tr><td>Informativeness w/ SG</td><td>2.17×K</td><td>112 min</td><td>11256 MiB</td></tr></table>


Table 2: Efficiency comparison of different stepwise beam search methods on FOLIO. Baseline stepwise beam search uses only cumulative likelihood scoring.


## 4.4 Results on Additional LLMs

To further validate the broad effectiveness of our method, we implement it on Phi-4 (Abdin et al., 2024), Qwen2.5-7B-Instruct (Qwen et al., 2025) (14B and 7B-parameter models from different families), and DeepSeek-R1-Distill-Llama-8B (Guo et al., 2025), a slow-thinking Llama3-8B variant distilled from DeepSeek-R1. We evaluate performance on FOLIO, ProofWriter, and MMLU-Pro, comparing against few-shot CoT, self-grounding, and self-consistency baselines using corresponding backbones. A one-shot prompting strategy is used with N = 3 and k = 1, and we extend the generation limit to 2048 tokens to accommodate long CoT from R1-Distill-Llama-8B. As shown in Table 3, our framework consistently improves performance on more powerful LLMs, though selfgrounding fails on R1-Distill-Llama-8B, as it learns to generate free-form CoT and struggles to follow a structured response format. Despite this, our informativeness search still yields significant improvements, notably reducing redundant tokens in final rationales (Table 4). This aligns with DeepSeek-R1’s over-thinking problems as pointed by (Chen et al., 2024; Cuadron et al., 2025). These results, along with Table 1 demonstrate our method’s robustness across models. 

<table><tr><td>Method</td><td>FOLIO</td><td>ProofWriter</td><td>MMLU-Pro</td></tr><tr><td colspan="4">Phi-4</td></tr><tr><td>Few-shot CoT</td><td>73.67%</td><td>72.55%</td><td>71.79%</td></tr><tr><td>Self-Grounding CoT</td><td>73.50%</td><td>72.06%</td><td>72.14%</td></tr><tr><td>Self-Consistency</td><td>71.17%</td><td>72.55%</td><td>72.50%</td></tr><tr><td>Informativeness Search w/ SG</td><td>76.67%</td><td>77.94%</td><td>72.86%</td></tr><tr><td colspan="4">Qwen2.5-7B-Instruct</td></tr><tr><td>Few-shot CoT</td><td>65.20%</td><td>54.17%</td><td>53.21%</td></tr><tr><td>Self-Grounding CoT</td><td>67.16%</td><td>54.83%</td><td>53.57%</td></tr><tr><td>Self-Consistency</td><td>69.12%</td><td>56.33%</td><td>54.29%</td></tr><tr><td>Informativeness Search w/ SG</td><td>70.59%</td><td>58.00%</td><td>55.36%</td></tr><tr><td colspan="4">DeepSeek-R1-Distill-Llama-8B</td></tr><tr><td>Few-shot CoT</td><td>61.76%</td><td>48.67%</td><td>38.57%</td></tr><tr><td>Self-Grounding CoT</td><td>53.92%</td><td>38.17%</td><td>35.36%</td></tr><tr><td>Self-Consistency</td><td>62.25%</td><td>63.50%</td><td>46.07%</td></tr><tr><td>Informativeness Search</td><td>70.10%</td><td>66.50%</td><td>47.50%</td></tr></table>


Table 3: Additional results on Phi-4, Qwen2.5-7B-Instruct and R1-Distill-Llama-8B.


<table><tr><td>Method</td><td>FOLIO</td><td>ProofWriter</td><td>MMLU-Pro</td></tr><tr><td>Few-shot CoT</td><td>1105</td><td>1861</td><td>1636</td></tr><tr><td>Informativeness Search</td><td>588</td><td>1023</td><td>1001</td></tr></table>


Table 4: Average token count of the final predicted reasoning paths from R1-Distill-Llama-8B.


## 4.5 Applicability to General Reasoning Task

To validate the generalization ability of our method on general reasoning tasks beyond multi-step reasoning datasets, we further evaluate it on a randomly sampled subset (100 instances) from the ARC Challenge dataset, a benchmark for commonsense reasoning. We report the results for Llama3.2-3B-Instruct, DeepSeek-R1-Distill-Llama-8B and Phi-4, as shown in Table 5. The consistent performance gains across all models demonstrate the applicability of our method beyond the original four multi-step reasoning datasets. 

<table><tr><td>Methods</td><td>Llama3.2-3B</td><td>R1-Llama-8B</td><td>Phi-4</td></tr><tr><td>Few-shot CoT</td><td>71%</td><td>68%</td><td>94%</td></tr><tr><td>Self-Grounding CoT</td><td>73%</td><td>74%</td><td>94%</td></tr><tr><td>Self-Consistency</td><td>70%</td><td>75%</td><td>95%</td></tr><tr><td>Stepwise Beam Search</td><td>75%</td><td>76%</td><td>94%</td></tr><tr><td>Informativeness Search w/ SG</td><td>78%</td><td>83%</td><td>97%</td></tr></table>


Table 5: Performance on ARC Challenge Dataset.


## 5 Further Analysis

## 5.1 Ablation Study

To investigate the contribution of our proposed novelty-guided and grounding-guided selection, we conduct an ablation study using LLama3.2- 3B-Instruct on FOLIO and MMLU-Pro datasets. Starting with stepwise beam search as our baseline, we separately incorporate each component individually, and compare them with our final Informative Search). Table 6 reveals that each selection heuristic contributes substantially to performance improvements, with their integration producing our best-performing search framework overall. Notably, novelty-based selection proves especially effective on FOLIO, addressing the challenge of redundant step generation in deductive reasoning tasks, while grounding-guided selection shows particular strength on MMLU-Pro. These findings validate the standalone effectiveness of each heuristic and their complementary nature when integrated into our complete framework. 

<table><tr><td>Methods</td><td>FOLIO</td><td>MMLU-Pro</td></tr><tr><td>Stepwise Beam Search</td><td>41.18%</td><td>30.36%</td></tr><tr><td>w/ only Novelty-Guided Heuristic</td><td>45.10%</td><td>32.14%</td></tr><tr><td>w/ only Grounding-Guided Heuristic</td><td>42.65%</td><td>32.50%</td></tr><tr><td>Informativeness Search</td><td>46.57%</td><td>33.57%</td></tr></table>


Table 6: Ablation study using LLaMA3.2-3B-Instruct.


## 5.2 Redundant Step Analysis

In complex multi-step reasoning tasks, LLMs tend to generate repeated intermediate conclusions, either from same or different premises, which can trap reasoning in circular loops. For detailed investigation, we measure the average number of repeated conclusions across steps per rationale generated by our method compared to few-shot CoT and self-grounding CoT baselines using LLama3.2-3B-Instruct. Specifically, we split rationales into steps using end-of-line token “/n” and extract intermediate conclusions based on special clause delimiters as operated in Sec. 3.2. A step is considered redundant if its conclusion shares over 70% tri-word overlap with any previous conclusions in the same rationale. As shown in Figure 4, LLMs exhibit a pronounced tendency to produce redundant steps, particularly in deductive reasoning tasks. This occurs because deductive contexts often contain verbally similar information, causing LLMs to lose track of logical progression and become stuck in circular reasoning. In contrast, our self-grounding strategy and informativeness search substantially reduce redundant steps, enabling more effective and efficient multi-step reasoning. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/afc6774d-7097-4258-889a-9f6c3bfcdfad/733fadbdf643d19a10669782e29ddbcac0071f49ef606dfe98be8ac7481bb6cd.jpg)



Figure 4: Average count of redundant steps whose conclusions have over 70% tri-word overlap with any previous conclusions in the same rationale.


## 5.3 Validity of Attention-Based Selection

To validate our attention-based implementation in grounding-guided selection, we examine whether LLMs naturally assign higher attention to grounded steps than other steps. Using the CLUTRR dataset, which provides well-annotated reasoning paths, we conduct a teacher-forcing analysis where all previous ground-truth steps are fed into the model to prompt the next step. We then compute the average attention score over both grounded and nongrounded steps. This analysis is performed both with and without self-grounding, using Llama3.2- 3B-Instruct and Llama3-8B-Instruct. As shown in Fig. 5, LLMs exhibit significantly higher attention over grounded steps. This demonstrates the consistency of LLMs’ attention patterns and their grounding behavior, and confirms the validity of our attention-based implementation. 

## 5.4 Effectiveness using Embedding-based Similarity

We further explore the effectiveness of our method when utilizing more robust embedding-based similarity instead of simple trigram-based similarity to measure novelty and information gain. Specifically, we employ a lightweight sentence-transformers model, all-MiniLM-L6-v2 (with only 22.7M parameters), which maps each sentence into a 384- dimensional embedding for cosine similarity computation. We compare this embedding-based method to our original trigram-based measure on FOLIO and MMLU-Pro using Llama3.2-3B-Instruct and Llama3-8B-Instruct. Results in Table 7 suggest that the embedding-based method provides comparable or slightly improved performance on MMLU-Pro, which contains more free-form text. The slight drop on FOLIO may stem from its structured and templated format, where trigram matching more precisely captures exact phrase overlap. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/afc6774d-7097-4258-889a-9f6c3bfcdfad/b434b10e2afe8934492d03abda6252e5426257c7333fb5ede5959130c6879a8d.jpg)



Figure 5: Average attention on grounded and other steps.


<table><tr><td>Method</td><td>FOLIO</td><td>MMLU-Pro</td></tr><tr><td colspan="3">Llama3.2-3B-Instruct</td></tr><tr><td>Few-shot CoT</td><td>38.73%</td><td>28.57%</td></tr><tr><td>Informativeness Search w/SG (Trigram)</td><td>51.96%</td><td>33.93%</td></tr><tr><td>Informativeness Search w/SG (Embedding)</td><td>51.47%</td><td>35.00%</td></tr><tr><td colspan="3">Llama3-8B-Instruct</td></tr><tr><td>Few-shot CoT</td><td>54.90%</td><td>37.50%</td></tr><tr><td>Informativeness Search w/SG (Trigram)</td><td>59.80%</td><td>40.71%</td></tr><tr><td>Informativeness Search w/SG (Embedding)</td><td>59.31%</td><td>40.36%</td></tr></table>


Table 7: Additional results using embedding-based similarity.


## 6 Related Work

LLMs (OpenAI, 2023; Abdin et al., 2024; Guo et al., 2025) and Chain-of-Thought (CoT) prompting (Wei et al., 2022; Zhou et al., 2022) have demonstrated remarkable performance in reasoning tasks by generating step-by-step rationales. However, for complex multi-step reasoning problems, LLMs often underutilize earlier critic information as rationale getting longer due to tendency to lose focus on middle-context content (See et al., 2017; Peysakhovich and Lerer, 2023; Junqing et al., 2023; Hsieh et al., 2024). They also frequently generate repeated sub-conclusions, leading to redundant reasoning and error accumulation (Dziri et al., 2024; Furuta et al., 2024). These difficulties are pronounced in smaller LLMs with limited reasoning capacity (Fu et al., 2023). An intuitive method is to simply prompt for concise outputs. However, LLMs often struggle to maintain output quality under length constraints, leaving grounding and redundancy issues unresolved (Nayab et al., 2024; Han et al., 2024). 

This inspires generating multiple rationales and selecting the most likely solution via majority voting (Wang et al., 2022) or best-of-N (Wang et al., 2024b). However, they are computationally costly due to exponentially growing search space. To reduce search space, tree search techniques use scoring mechanisms to prioritize promising candidates at each step, such as stepwise beam search (Xie et al., 2024), Tree-of-Thought (Yao et al., 2024), and Monte Carlo Tree Search (Jiang et al., 2024; Feng et al., 2023; Zhang et al., 2024a). While effective, they face practical limitations, relying on extensive rollouts (Wang et al., 2024b,a) and annotations (Lightman et al., 2023) for training specialized reward models. Besides, they introduce latency due to interactions with external or self-evaluators during autoregressive decoding (Xie et al., 2024; Yao et al., 2024), and ignore the grounding and redundancy issues in this work. 

## 7 Conclusion

In this work, we address the challenge of LLMs losing focus on intermediate steps during multistep reasoning, which can lead to unreliable and redundant rationales. To mitigate this, we propose an inference-time tree search framework incorporating grounding-guided and novelty-guided heuristics, enhancing rationale generation by proactively grounding underutilized steps and minimizing redundancy between reasoning steps. Our selfgrounding strategy further prompts LLMs to explicitly reference relevant prior steps before each deduction. Experiments show that our method improves reasoning accuracy and efficiency with fewer errors and reduced redundancy. 

## Limitations

Our work has several limitations to address in future research. First, due to computational constraints, our main experiments operate within a limited search space with beam size 3 and sample size 2, and use LLM backbones of at most 14B parameters. Future work can explore larger search spaces and more powerful LLMs to further unlock the potential of our framework. Second, while our method currently relies solely on stepwise beam search with standard cumulative likelihood, incorporating our selection heuristics with other scoring mechanism, such as self-evaluation and process reward models, as well as other tree-search algorithms like MCTS could be potential future work. 

## Acknowledgments

This research is supported in part by the Office of the Director of National Intelligence (ODNI), Intelligence Advanced Research Projects Activity (IARPA), via the HIATUS Program contract #2022- 22072200006, the Defense Advanced Research Projects Agency with award HR00112220046, and NSF IIS 2048211. We would like to thank all the collaborators in USC INK research lab for their constructive feedback on the work. 

## References



Marah Abdin, Sam Ade Jacobs, Ammar Ahmad Awan, Jyoti Aneja, Ahmed Awadallah, Hany Awadalla, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Harkirat Behl, et al. 2024. Phi-3 technical report: A highly capable language model locally on your phone. arXiv preprint arXiv:2404.14219. 





Xingyu Chen, Jiahao Xu, Tian Liang, Zhiwei He, Jianhui Pang, Dian Yu, Linfeng Song, Qiuzhi Liu, Mengfei Zhou, Zhuosheng Zhang, et al. 2024. Do not think that much for 2+ 3=? on the overthinking of o1-like llms. arXiv preprint arXiv:2412.21187. 





Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457. 





Alejandro Cuadron, Dacheng Li, Wenjie Ma, Xingyao Wang, Yichuan Wang, Siyuan Zhuang, Shu Liu, Luis Gaspar Schroeder, Tian Xia, Huanzhi Mao, Nicholas Thumiger, Aditya Desai, Ion Stoica, Ana Klimovic, Graham Neubig, and Joseph E. Gonzalez. 2025. The danger of overthinking: Examining the reasoning-action dilemma in agentic tasks. Preprint, arXiv:2502.08235. 





Nouha Dziri, Ximing Lu, Melanie Sclar, Xiang Lorraine Li, Liwei Jiang, Bill Yuchen Lin, Sean Welleck, Peter West, Chandra Bhagavatula, Ronan Le Bras, et al. 2024. Faith and fate: Limits of transformers on compositionality. Advances in Neural Information Processing Systems, 36. 





Xidong Feng, Ziyu Wan, Muning Wen, Stephen Marcus McAleer, Ying Wen, Weinan Zhang, and Jun Wang. 2023. Alphazero-like tree-search can guide large language model decoding and training. arXiv preprint arXiv:2309.17179. 





Yao Fu, Hao Peng, Litu Ou, Ashish Sabharwal, and Tushar Khot. 2023. Specializing smaller language models towards multi-step reasoning. In International Conference on Machine Learning, pages 10421–10430. PMLR. 





Hiroki Furuta, Yutaka Matsuo, Aleksandra Faust, and Izzeddin Gur. 2024. Exposing limitations of language model agents in sequential-task compositions on the web. In ICLR 2024 Workshop on Large Language Model (LLM) Agents. 





Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948. 





Simeng Han, Hailey Schoelkopf, Yilun Zhao, Zhenting Qi, Martin Riddell, Wenfei Zhou, James Coady, David Peng, Yujie Qiao, Luke Benson, et al. 2022. Folio: Natural language reasoning with first-order logic. arXiv preprint arXiv:2209.00840. 





Tingxu Han, Chunrong Fang, Shiyu Zhao, Shiqing Ma, Zhenyu Chen, and Zhenting Wang. 2024. Token-budget-aware llm reasoning. arXiv preprint arXiv:2412.18547. 





Cheng-Yu Hsieh, Yung-Sung Chuang, Chun-Liang Li, Zifeng Wang, Long T Le, Abhishek Kumar, James Glass, Alexander Ratner, Chen-Yu Lee, Ranjay Krishna, et al. 2024. Found in the middle: Calibrating positional attention bias improves long context utilization. arXiv preprint arXiv:2406.16008. 





Jinhao Jiang, Zhipeng Chen, Yingqian Min, Jie Chen, Xiaoxue Cheng, Jiapeng Wang, Yiru Tang, Haoxiang Sun, Jia Deng, Wayne Xin Zhao, et al. 2024. Technical report: Enhancing llm reasoning with rewardguided tree search. arXiv preprint arXiv:2411.11694. 





He Junqing, Pan Kunhao, Dong Xiaoqun, Song Zhuoyang, Liu Yibo, Liang Yuxin, Wang Hao, Sun Qianguo, Zhang Songxin, Xie Zejian, et al. 2023. Never lost in the middle: Improving large language models via attention strengthening question answering. arXiv preprint arXiv:2311.09198. 





Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. 2023. Let’s verify step by step. arXiv preprint arXiv:2305.20050. 





Liangchen Luo, Yinxiao Liu, Rosanne Liu, Samrat Phatale, Harsh Lara, Yunxuan Li, Lei Shu, Yun Zhu, Lei Meng, Jiao Sun, et al. 2024. Improve mathematical reasoning in language models by automated process supervision. arXiv preprint arXiv:2406.06592. 





Sania Nayab, Giulio Rossolini, Giorgio Buttazzo, Nicolamaria Manes, and Fabrizio Giacomelli. 2024. Concise thoughts: Impact of output length on llm reasoning and cost. arXiv preprint arXiv:2407.19825. 





OpenAI. 2023. Gpt-4 technical report. Preprint, arXiv:2303.08774. 





Alexander Peysakhovich and Adam Lerer. 2023. Attention sorting combats recency bias in long context language models. arXiv preprint arXiv:2310.01427. 





Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. 2025. Qwen2.5 technical report. Preprint, arXiv:2412.15115. 





David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. 2023. Gpqa: A graduate-level google-proof q&a benchmark. arXiv preprint arXiv:2311.12022. 





Abigail See, Peter J Liu, and Christopher D Manning. 2017. Get to the point: Summarization with pointer-generator networks. arXiv preprint arXiv:1704.04368. 





Koustuv Sinha, Shagun Sodhani, Jin Dong, Joelle Pineau, and William L Hamilton. 2019. Clutrr: A diagnostic benchmark for inductive reasoning from text. arXiv preprint arXiv:1908.06177. 





Oyvind Tafjord, Bhavana Dalvi Mishra, and Peter Clark. 2020. Proofwriter: Generating implications, proofs, and abductive statements over natural language. arXiv preprint arXiv:2012.13048. 





Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. 2023. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805. 





Chaojie Wang, Yanchen Deng, Zhiyi Lyu, Liang Zeng, Jujie He, Shuicheng Yan, and Bo An. 2024a. Q*: Improving multi-step reasoning for llms with deliberative planning. arXiv preprint arXiv:2406.14283. 





Peiyi Wang, Lei Li, Zhihong Shao, Runxin Xu, Damai Dai, Yifei Li, Deli Chen, Yu Wu, and Zhifang Sui. 2024b. Math-shepherd: Verify and reinforce llms step-by-step without human annotations. In Proceedings ofthe 62nd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), pages 9426–9439. 





Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2022. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171. 





Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. 2024c. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. arXiv preprint arXiv:2406.01574. 





Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. 2022. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837. 





Yuxi Xie, Kenji Kawaguchi, Yiran Zhao, James Xu Zhao, Min-Yen Kan, Junxian He, and Michael Xie. 2024. Self-evaluation guided beam search for reasoning. Advances in Neural Information Processing Systems, 36. 





Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. 2024. Tree of thoughts: Deliberate problem solving with large language models. Advances in Neural Information Processing Systems, 36. 





Dan Zhang, Sining Zhoubian, Ziniu Hu, Yisong Yue, Yuxiao Dong, and Jie Tang. 2024a. Rest-mcts*: Llm self-training via process reward guided tree search. arXiv preprint arXiv:2406.03816. 





Di Zhang, Xiaoshui Huang, Dongzhan Zhou, Yuqiang Li, and Wanli Ouyang. 2024b. Accessing gpt-4 level mathematical olympiad solutions via monte carlo tree self-refine with llama-3 8b. arXiv preprint arXiv:2406.07394. 





Qingru Zhang, Chandan Singh, Liyuan Liu, Xiaodong Liu, Bin Yu, Jianfeng Gao, and Tuo Zhao. 2023. Tell your model where to attend: Post-hoc attention steering for llms. arXiv preprint arXiv:2311.02262. 





Zhenru Zhang, Chujie Zheng, Yangzhen Wu, Beichen Zhang, Runji Lin, Bowen Yu, Dayiheng Liu, Jingren Zhou, and Junyang Lin. 2025. The lessons of developing process reward models in mathematical reasoning. arXiv preprint arXiv:2501.07301. 





Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc Le, et al. 2022. Least-to-most prompting enables complex reasoning in large language models. arXiv preprint arXiv:2205.10625. 





Tinghui Zhu, Kai Zhang, Jian Xie, and Yu Su. 2024. Deductive beam search: Decoding deducible rationale for chain-of-thought reasoning. arXiv preprint arXiv:2401.17686. 



## A Implementation Details

## A.1 Baseline Details

For Best-of-N and Self-Consistency, we adopt a sampling configuration with temperature $T = 1 . 0$ and top-40 token truncation. For tree-of-thought (ToT) and self-eval beam search (Self-Eval BS), we prompt LLMs to conduct self-evaluation. For deductive beam search that provide a general verifier checkpoint and two data subsets for training a commonsense and a mathematical verifier, we select the best-performing verifier for each dataset. Specifically, we use the general or commonsense verifier for FOLIO, ProofWriter, and MMLU-Pro, and the general or mathematical verifier for GPQA. For MCTS which operates in a iterative four-stage manner: selection, expansion, simulation and backprogation, we use the minimum score across all steps from Qwen2.5-Math-PRM-7B (Zhang et al., 2025) to evaluate simulated rollout. 

Comparison among ToT, MCTS, and Selfevaluation Beam Search ToT and Selfevaluation beam search both perform stepwise tree search using a scorer function at each step. The key difference lies in granularity: Self-eval scores and accumulates at the individual step level, while ToT evaluates each thought generated so far as a unit. In contrast, MCTS follows a fundamentally different approach. It performs simulation-based search, where each node is evaluated through multiple rollouts, and scores are provided by a reward model. While this enables deeper exploration, it also introduces significantly higher computational cost. 

## A.2 Dataset Details

We evaluate our framework on four multi-step reasoning datasets: FOLIO (Han et al., 2022), ProofWriter (Tafjord et al., 2020), MMLU-Pro (Wang et al., 2024c) and GPQA (Rein et al., 2023), and and one commonsense dataset ARC Challenge (Clark et al., 2018). FOLIO and ProofWriter focus on deductive reasoning, requiring 1-8 and 1-6 reasoning steps respectively, with test sets of 204 and 600 cases. MMLU-Pro covers 14 domains, including math, physics, chemistry, engineering, law, and psychology, from which we uniformly sample 280 cases. GPQA specializes in biology, physics, and chemistry, and we use its Diamond subset containing 198 expert-answered but non-expert-failed questions. 

## A.3 Implementation Details

We apply grounding-guided and novelty-guided selection primarily after the third step, as the initial steps typically ground on the given context and contain minimal redundant steps. Additionally, our prompt requires summarizing the final answer after the reasoning process, separated by the special token “[END]”. We restrict grounding-guided and novelty-guided selection to the reasoning phase only, ensuring that the final answer output, which often overlaps with the last reasoning step, remains unaffected. 

## A.4 Varying Search Configurations

For step-level candidate generation in stepwise beam search, we explore both temperature sampling and tokenwise beam search. As shown in Table 8, our method with grounding and noveltyguided selection consistently outperforms stepwise beam search baseline (with cumulative likelihood scoring), regardless of whether self-grounding is applied. Additionally, tokenwise beam search for candidate generation yields slightly better performance than temperature sampling. 

<table><tr><td>Methods</td><td>FOLIO</td><td>MMLU-Pro</td></tr><tr><td colspan="3">Beam Search</td></tr><tr><td>Stepwise Beam Search</td><td>41.18%</td><td>30.36%</td></tr><tr><td>Informativeness Search</td><td>46.57%</td><td>33.57%</td></tr><tr><td>Stepwise Beam Search (w/ SG)</td><td>50.49%</td><td>32.86%</td></tr><tr><td>Informativeness Search (w/ SG)</td><td>51.96%</td><td>33.93%</td></tr><tr><td colspan="3">Temperature Sampling</td></tr><tr><td>Stepwise Beam Search</td><td>41.67%</td><td>29.64%</td></tr><tr><td>Informativeness Search</td><td>44.12%</td><td>31.43%</td></tr><tr><td>Stepwise Beam Search (w/ SG)</td><td>47.55%</td><td>29.64%</td></tr><tr><td>Informativeness Search (w/ SG)</td><td>48.53%</td><td>32.50%</td></tr></table>


Table 8: Different candidate step generation methods.


We further evaluate the impact of varying beam sizes in our informativeness search, using both tokenwise beam search and temperature sampling for candidate step generation. Specifically, we set the sample size to 2 and vary the beam size from 1 to 4. As shown in Fig. 6, both alternatives consistently outperform the few-shot CoT baseline. Additionally, our informativeness search continues to improve as beam size increases. Notably, when the search space is constrained (i.e., with a smaller beam size), tokenwise beam search performs better. Based on these findings, we adopt tokenwise beam search for all stepwise beam search methods in our reported results (Table 1 4) considering its better performance and accelerated computational speed. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/afc6774d-7097-4258-889a-9f6c3bfcdfad/94801ec221812aa03a50eaffa6abb69a37cd447be27a329315f745f16d886a1d.jpg)



Figure 6: The impact of beam size on our utility-based search for the FOLIO dataset on Llama3.2-3B-Instruct.


## A.5 Comparison to Tokenwise Beam Search

We further compare our informativeness search (beam size N = 3, sample size k = 2) with naive tokenwise beam search for whole rationale generation using beam size 3 and 6. Table 9 demonstrate the effectiveness of our method. 

<table><tr><td>Method</td><td>FOLIO</td><td>ProofWriter</td><td>MMLU-Pro</td><td>GPQA-D</td></tr><tr><td>Few-shot CoT</td><td>38.73%</td><td>40.00%</td><td>28.57%</td><td>21.72%</td></tr><tr><td>Tokenwise BS (3)</td><td>43.63%</td><td>45.00%</td><td>28.93%</td><td>21.72%</td></tr><tr><td>Tokenwise BS (6)</td><td>46.08%</td><td>42.17%</td><td>31.07%</td><td>19.19%</td></tr><tr><td>Informativeness Search</td><td>46.57%</td><td>50.33%</td><td>33.93%</td><td>27.27%</td></tr></table>


Table 9: Comparison with tokenwise beam search using Llama3.2-3B-Instruct for whole rationale generation. Numbers in parentheses denote the beam size.


## B Framework Prompts

Table 10, 11, 12 and 13 present the prompts used in our informativeness search framework without selfgrounding strategy for the FOLIO, ProofWriter, MMLU-Pro and GPQA-Diamond datasets. For illustration, Table 14 provides the prompt used in our informativeness search framework with selfgrounding strategy on GPQA-Diamond <sup>3</sup>. 

## C Correlation between of Grounding Challenge and Reasoning Performance

We provide a detailed illustration of the grounding challenge that LLMs face when referencing prior reasoning steps. Specifically, we analyze all instances involving 8-9 reasoning steps from CLUTRR (Sinha et al., 2019), a dataset with well annotated rationales. We evaluate the performance of Llama3-8B-Instruct across instances grouped by the maximum distances between referencing and referenced steps. As shown in Fig. 7, perfor mance degrades as the distance to the referenced prior steps grows. This demonstrates the inherent difficulty of grounding distant prior step, with longer distances (steps accumulating) progressively degrade reasoning performance. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/afc6774d-7097-4258-889a-9f6c3bfcdfad/e84d82e854ae6f6b0dd9619b981bbe6bb042a273edcf50000f51d4553916e7c1.jpg)



Figure 7: Accuracy versus maximum distance between referencing and referenced steps on CLUTRR.


## D Attention Calculation

To compute the grounding score $\gamma _ { a } ( s _ { t } ^ { i } )$ that measures how effectively step $s _ { t } ^ { i }$ attends to the underutilized steps $\mathcal { T } _ { t } ^ { i }$ , we use a selective attention mechanism that emphasizes the most relevant connections while remaining robust to noisy information. Our approach proceeds in three stages. First, we collect attention weights from all L layers and H heads of the model. For each generated step, we extract the attention scores from the N tokens in the current step $s _ { t } ^ { i }$ to the M tokens in the underutilized prior steps $\mathcal { T } _ { t } ^ { i }$ , yielding $L \times H \times M \times N$ attention values. Next, we take the mean over the N tokens in the current step $s _ { t } ^ { i }$ , reducing it to $L \times H \times M$ scores to measure how much the step $s _ { t } ^ { i }$ as a whole attends to each token in underutilized steps. Finally, we select the top-K scores (we experiment and conclude that $K = 5 0 \sim 1 0 0$ yield similar results) and compute their mean to obtain the final attention score $\gamma _ { a } ( s _ { t } ^ { i } )$ . This selective averaging maintains robustness to irrelevant or noisy steps while capturing most valuable signals from underutilized steps. 

## E Justification of Using Trigram-based Similarity

We simply adopt trigram-based similarity to measure novelty and information gain for each step for the following reasons. (1) Similarity computation includes both semantics-based similarity and N-gram-based similarity. However, the former often requires additional model inference, introducing substantial computational overhead. To reduce costs, we opted for $N .$ -gram-based similarity, which can be computed efficiently without model calls. (2) Among N-gram-based metrics, uni-gram similarity performs poorly, as it only captures word overlap and lacks sentence-level representation. In contrast, higher-order N-grams (e.g., 4-gram) are overly strict and fail to recognize semantically similar sentences with slightly different wording. Through empirical comparisons on bi-gram and trigram similarity, we found that trigram-based similarity provides the best trade-off between robustness and flexibility, making it the preferred choice in our implementation. Alternatively, lightweight embedding-based measures can further enhance semantic fidelity and merit further exploration. 

## F Further Discussion on Handling Intermediate Noise or Repetition

LLMs inevitably generate irrelevant or incorrect steps during reasoning, making not all underutilized steps useful. However, we argue that our method encouraging the grounding of underutilized steps still improve reasoning. As discussed in Sec. 1, empirical evidence shows that when LLaMA-3-8B is prompted to self-generate multistep rationales on CLUTRR instances requiring eight ground-truth steps, accuracy is significantly higher (57.89%) in cases where each step grounds on at least one underutilized prior step, compared to 22.39% when they do not. Besides, our proposed self-grounding strategy further mitigate the impact of irrelevant steps by prompting the model to identify and state relevant prior steps before each deduction, leveraging its internal reasoning to surface useful content. Finally, we adopt a soft attention mechanism prioritizes grounding on a selective subset of underutilized steps, maintaining robustness against noise while capturing valuable signals. 

Additionally, our framework can extend to longthinking models that generate intermediate conclusions similar to their final answers. Unlike shortthinking models that terminate reasoning upon reaching likely answer, long-thinking models usually explore multiple alternative solutions or perform verification. However, valuable alternative solutions and verifications often appear in varied expressions. For instance, the conclusion “the answer to $2 + 3$ is $5 ^ { \circ }$ may emerge through diverse intermediate forms such as “two apples plus three apples equals five apples” or “2 is II and 3 is III in Roman numerals, adding them gives $\mathbf { V } ^ { \ast }$ . Similarly, useful verifications include phrases like “which is the same as 2 plus $3 ^ { \circ }$ or “this aligns with 2 plus 3 equals $5 ^ { \circ }$ . Our trigram-based similarity metric recognizes these varied expressions as semantically novel, encouraging diverse reasoning paths that increase information richness and model confidence in its answer. The strong performance of DeepSeek-R1-Distill-Llama-8B in Table 3 validates the generalization of our approach to long-thinking models. Moreover, we can apply in-block novelty-guided selection, filtering within reasoning blocks segmented by special transition tokens (e.g., “Wait,” or “Alternatively”) to maintain diversity while mitigating the risk of filtering meaningful content. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/afc6774d-7097-4258-889a-9f6c3bfcdfad/0fdd6fa409dc4ed7a90a7279d056961f230b9b8f002b34288f34380cc167f9ac.jpg)



Table 10: The prompt without self-grounding on FOLIO.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/afc6774d-7097-4258-889a-9f6c3bfcdfad/63511eff793168f00201ec0686a1a1743dc24d4842e367a2a2ccf7b5288260f1.jpg)



Table 11: The prompt without self-grounding on ProofWriter.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/afc6774d-7097-4258-889a-9f6c3bfcdfad/51484a060ce6d787b02d69351cc225ce11bea7840bb29f7ab2d73895c2cbda7a.jpg)



Table 12: The prompt without self-grounding on MMLU-Pro.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/afc6774d-7097-4258-889a-9f6c3bfcdfad/a106bcd47b1bcd67511e3e96130b04183374fff4190d5692f72ce8d2dbd65ce7.jpg)



Table 13: The prompt without self-grounding on GPQA-Diamond.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/afc6774d-7097-4258-889a-9f6c3bfcdfad/6c5677b680249489081ef1cfb553f8a2761e8faac0d91e31144ee13ab9e59fc4.jpg)



Table 14: The prompt with self-grounding on GPQA-Diamond.
