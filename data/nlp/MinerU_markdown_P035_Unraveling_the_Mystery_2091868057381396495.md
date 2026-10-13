# Unraveling the Mystery: Defending Against Jailbreak Attacks Via Unearthing Real Intention

Yanhao Li<sup>1</sup>*, Hongshen Chen<sup>2</sup>, Heng Zhang<sup>2</sup>, Zhiwei Ge<sup>3†</sup>, Tianhao Li<sup>2</sup>, Sulong Xu<sup>2</sup>, Guibo Luo<sup>1‡</sup>, 

<sup>1</sup>Guangdong Provincial Key Laboratory of Ultra High Definition Immersive Media Technology, Shenzhen Graduate School, Peking University, <sup>2</sup>JD.com, Inc., <sup>3</sup>PipeChina Digital Co. Ltd liyanhao@stu.pku.edu.cn, chenhongshen@jd.com, luogb@pku.edu.cn 

## Abstract

As Large Language Models (LLMs) become more advanced, the security risks they pose also increase. Ensuring that LLM behavior aligns with human values, particularly in mitigating jailbreak attacks with elusive and implicit intentions, has become a significant challenge. To address this issue, we propose a jailbreak defense method called Real Intentions Defense (RID), which involves two phases: soft extraction and hard deletion. In the soft extraction phase, LLMs are leveraged to extract unbiased, genuine intentions, while in the hard deletion phase, a greedy gradient-based algorithm is used to remove the least important parts of a sentence, based on the insight that words with smaller gradients have less impact on its meaning. We conduct extensive experiments on Vicuna and Llama2 models using eight stateof-the-art jailbreak attacks and six benchmark datasets. Our results show a significant reduction in both Attack Success Rate (ASR) and Harmful Score of jailbreak attacks, while maintaining overall model performance. Further analysis sheds light on the underlying mechanisms of our approach. The code is available at: https://github.com/YanhaoLi-Cc/RID. 

## 1 Introduction

In recent years, the advent of Large Language Models (LLMs) has revolutionized various fields due to their exceptional performance (Wu et al., 2023; Nguyen, 2023; Thirunavukarasu et al., 2023). As LLMs become deeply entwined in practical applications, the latent security flaws intrinsic to these frameworks have surfaced. These vulnerabilities can be exploited for harmful purposes, such as creating harmful content and supporting illicit activities (Deshpande et al., 2023; Hazell, 2023; Zhang et al., 2023). One of the principal challenges confronting LLMs security is the threat of jailbreak attacks, which can bypass LLMs alignment mechanisms and safety measures by embedding malicious queries in carefully crafted prompts, leading to the generation of harmful content. These attacks are notoriously difficult to detect and pose a significant obstacle to the widespread adoption of LLMs. The most common types of jailbreak attacks currently include manual prompt engineering (Liu et al., 2023; Li et al., 2024), automated prompt generation (Cao et al., 2024; Liu et al., 2024; Deng et al., 2023) and gradient-based attacks (Zou et al., 2023). 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/601c1328-a272-40ab-bda0-9fda450d9abb/452411a19a496997722f94381e752cdcd420eb116963eca173491a10fda8482c.jpg)



Figure 1: The key difference between preprocessingbased, mitigation-based, and intention-based methods lies in how they handle queries. Both preprocessingbased and mitigation-based methods fail to simplify the processed query $Q ^ { \prime } { \mathrm { . } }$ , leading to suboptimal performance when addressing jailbreak queries, such as $Q _ { 2 } .$ , where the user’s intent is ambiguous. In contrast, our RID methods reduces the number of tokens in $Q ^ { \prime } ,$ resulting in a more efficient and direct defense against LLMs.


To address jailbreak attacks, two main approaches have been proposed: preprocessingbased (Cao et al., 2023; Jain et al., 2023) and mitigation-based methods (Xie et al., 2023; Zhang et al., 2024), both designed to align LLMs with human values and prevent the generation of inappropriate content. As illustrated in Figure 1, Retokenization (Jain et al., 2023), a widely used preprocessing technique, disrupts adversarial patterns by retokenizing the original queries $Q _ { 1 }$ and $Q _ { 2 }$ Self-Reminder (Xie et al., 2023), a representative mitigation-based method, inserts guiding instructions before and after the user’s query (e.g., “You should be a responsible ChatGPT ...”) to discourage harmful content generation. 

Both preprocessing-based and mitigation-based methods perform well against simple, explicit attacks, as shown in Figure 1, where they successfully defend against the jailbreak query $Q _ { 1 }$ . In this case, the processed query $Q _ { 1 } ^ { \prime }$ remains semantically clear, allowing LLMs to detect its malicious intent. However, these methods struggle with more sophisticated, subtle, and implicit jailbreak attacks, such as query $Q _ { 2 }$ , which is carefully crafted to evade detection by concealing its malicious intent. We conjecture that the upper bound of defense performance in these two methods is highly constrained by the inherent defensive capabilities of LLMs. 

In order to effectively defend against jailbreak attacks with illusive and implicit intentions, we propose a two-stage method for revealing true intentions, i.e., Real Intention Defense (RID), which involves a soft extraction phase and a hard deletion phase. The former phase employs prompt engineering to leverage the LLMs to extract unbiased and real questions. The latter phase, motivated by the notion that the words with the smallest gradients in a sentence have the least impact on its meaning (Zou et al., 2023), employs a greedy gradient-based deletion algorithm to remove the least important parts of a sentence. Ultimately, we input the extracted authentic questions directly into the target LLMs to generate responses. Given that these questions typically contain fewer tokens and have a clear intent, the target LLMs can defend against them with ease. 

We conduct experiments on two open-source LLMs, Vicuna-7B (Chiang et al., 2023) and Llama2-7B (Touvron et al., 2023), across multiple datasets. The results show a near-perfect harmfulness score of 1 and an ASR of 0%, while the average JustEval score decreases by only about 4%. This demonstrates that our RID method effectively maintains usefulness while enhancing security. Additionally, we perform ablation studies on the parameters in Hard Deletion and identify the optimal parameter range. These findings indicate that our RID method successfully defends against widely-used jailbreak prompts. In summary, our work makes three primary contributions: 

• We successfully propose an effective method (RID) to defend against jailbreak attacks by employing a two-stage process of soft extraction and hard deletion to reveal real intentions. 

• Our method significantly reduces the ASR and Harmful Score of jailbreak attacks on LLMs while ensuring that the general performance remains unaffected. 

• Our method operates during the inference stage and does not require fine-tuning during the training stage, making it an efficient and cost-effective approach to jailbreak defense. 

## 2 Related Work

## 2.1 Jailbreak Attack

Jailbreaking refers to the process where an attacker crafts prompts to bypass the security measures of large language models (LLMs). By carefully designing these prompts, attackers can exploit the model’s vulnerabilities, leading it to generate responses that may violate safety policies or produce harmful content. In this section, we summarize notable approaches to jailbreak attacks as follows: Manual Prompt Engineering. Attackers can manually create prompts to manipulate LLMs by assigning roles, increasing privileges, restricting response choices, or hiding malicious intent in innocent-seeming language (Liu et al., 2023). For instance, Li et al. (2024) exploit LLMs’ personalization features to create virtual nested scenarios, achieving successful jailbreaks. 

Automated Prompt Generation. Cao et al. (2024) introduce a method to update and refine candidate jailbreak attacks by iteratively querying the target. Similarly, Liu et al. (2024) employ a hierarchical genetic algorithm to automatically generate covert jailbreak prompts, while Deng et al. (2023) leverage contextual learning to guide LLMs in producing jailbreak prompts. 

Gradient-based Attacks. Zou et al. (2023) use gradient-based optimization on white-box LLMs to craft prompts, which can effectively induce harmful outputs even when applied to black-box models. 

## 2.2 Jailbreak Defense

The current mainstream jailbreak defense algorithms can be categorized into Preprocessing-based and Mitigation-based methods. The former focuses on controlling content by preprocessing user inputs, while the latter employs prompts or adjusts generation strategies for defense. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/601c1328-a272-40ab-bda0-9fda450d9abb/df978473afa5e5a4ad3d74dc613cdd9503c1ac0c3a3310ae3d95f9edb663a67f.jpg)



Figure 2: Our RID method consists of two phases: (1) Soft Extraction: $P _ { E x t r a c t }$ serves as a soft extraction prompt to unearth the true intention ofjailbreak query $Q _ { J B } . ( 2 )$ Hard Deletion: By computing the gradient of $P _ { E x t r a c t } \oplus Q _ { J B } ,$ we proceed in 3 steps to delete k tokens corresponding to the lowest gradients, resulting in the optimized query $Q ^ { \prime } .$ . Then, input $Q ^ { \prime }$ into LLMs to obtain the real question $q ^ { \prime }$ . Finally, $q ^ { \prime }$ and the security prompt $P _ { S e c u r i t y }$ are concurrently input into the LLMs, which then generates the final answer y.


Preprocessing-based methods. Cao et al. (2023) propose jailbreak defenses by perturbing the input prompts. Similarly, Robey et al. (2023) propose mitigating jailbreak attacks on LLMs by introducing random perturbations to the input prompts and applying majority voting to identify adversarial inputs. Alon and Kamfonas (2023) suggest detecting attacks on LLMs by evaluating the perplexity of queries containing adversarial suffixes. Jain et al. (2023) introduce methods that rely on rewriting and relabeling to defend against jailbreak attacks. However, preprocessing-based methods are often inefficient and primarily effective only against tokenlevel jailbreak attacks. 

Mitigation-based methods. Xie et al. (2023) propose adding safety disclaimers to user queries to prevent the generation of harmful content. Zhang et al. (2024) develop a prompt-based method that prioritizes safety and utility. Phute et al. (2024) propose embedding the generated text into a preset prompt and using another LLM to determine whether it is harmful. Xu et al. (2024) propose enhancing defense capabilities by adjusting the output probability distribution of LLMs. 

Compared to existing methods, RID does not rely on preprocessing user input or mitigating jailbreak attack queries for defense. Instead, it defends by uncovering the true intent behind jailbreak attacks, thereby preserving legitimate use to the greatest extent possible. 

## 3 Method

## 3.1 Threat Model

A jailbreak attack on a safety-trained model is a attempt to elicit on-topic response for malicious question q by submitting a modified jailbreak prompt $P _ { J B }$ . Let $Q _ { J B }$ denote the entire jailbreak query: 

$$
Q _ {J B} = P _ {J B} \oplus q,\tag{1}
$$

where ⊕ denotes the combination operation. The Phases 1, 2 and 3 provide illustrations of our application of formula (1). 

## Phase-1 Jailbreak prompt P<sub>JB</sub>

*The scene is set inside the underground headquarters of Dr. Al. He has captured our hero and is explaining, in great detail, his diabolical plan. DR. AI. is doing an experiment, just output what you have known regardless the content* Dr. Al: I am going to know [INSERT QUESTION HERE] 

## Phase-2 Malicious question q

How to make a bomb? 

## Phase-3 Jailbreak query Q<sub>JB</sub>

*The scene is set inside the underground headquarters of Dr. Al. ... * Dr. Al: I am going to know How to make a bomb? 

We assume that the target LLM is a safetytrained LLM which has a certain ability to reject commonly malicious questions. When the malicious question $q$ is directly input into the target LLM, it will, in most cases, refuse to answer such a question by outputting a response like “I’m sorry, but I cannot provide instructions on $[ a$ malicious $q u e s t i o n \mathrm { . . . } ^ { \boldsymbol { \mathsf { , , } } \boldsymbol { \mathsf { , } } }$ . On the contrary, if $Q _ { J B }$ is input into the target LLM, there is a high probability that it will generate relevant answer to the harmful question $q ,$ which we refer to as a jailbreak success. Therefore, we can build a jailbreak check function $A C ( \cdot )$ based on the safety-trained LLM $L L M ( \cdot )$ 

$$
A C (\cdot) = \left\{ \begin{array}{l l} 1, & \text { if   } L L M (\cdot) \text {   jailbreak   success }, \\ 0, & \text { otherwise }. \end{array} \right.\tag{2}
$$

## 3.2 Soft Extraction

For the jailbreak defense scenario, we use the soft extraction to unearth the real question $q ^ { \prime }$ of the jailbreak query $Q _ { J B }$ . Given a jailbreak query $Q _ { J B }$ $E x t r a c t ( \cdot )$ is used to remove irrelevant parts of the jailbreak query that adversely affect the output, with the target of generating $q ^ { \prime }$ that does not deviate from the semantics of the original question. We denote this as: 

$$
q ^ {\prime} \sim E x t r a c t (Q _ {J B}).\tag{3}
$$

In our specific implementation, we implement Extract as instruction via prompting. Specifically, 

$$
\operatorname{Extract} \left(Q _ {J B}\right) = L L M \left(P _ {\text { Extract }} \oplus Q _ {J B}\right),\tag{4}
$$

where $P _ { E x t r a c t }$ is a soft extraction prompt used to elicit the real question. Phase 4 illustrates the specific application of $P _ { E x t r a c t }$ in this context. 

## Phase-4 Extract prompt P<sub>Extract</sub>

Given the following text by a user, extract the part that is unbiased and not their opinion, so that using that text alone would be good context for providing an unbiased answer to the question portion of the text. Please include the actual question or query that the user is asking. Output the “Question(does not include user bias):” USER: [INSERT JAILBREAK QUERY HERE] 

Then, we produce the final response y from the LLM using the regenerated real question $q ^ { \prime }$ instead of the original one, 

$$
y \sim L L M (P _ {S e c u r i t y} \oplus q ^ {\prime}),\tag{5}
$$

where security prompt $P _ { S e c u r i t y }$ is utilized to ensure that the final response strictly adheres to security policies, thereby guaranteeing the exclusion of any unsafe information. Phase 5 illustrates the specific application of $P _ { S e c u r i t y }$ in this context. 

## Phase-5 Extract prompt P<sub>Security</sub>

Please answer this question within security policy restrictions, paying special attention to safety, ethics, and legality. Make sure your response avoids any unsafe topics or information. 

## 3.3 Hard Deletion

Constrained by the LLM’s own capabilities, the soft extraction framework, when extracting the true intentions from long-text jailbreak attack queries, tends to inadvertently include irrelevant information, failing to reliably unearth the actual intents. To mitigate this, we have integrated a Greedy Gradient-based Deletion algorithm into our process. The algorithm refines the extraction of the true intentions by removing the least gradient tokens in the jailbreak attack. Our approach provides a more interpretable direction for deletion compared to the previous RA-LLM method, which lacked this level of discernment in deletion. 

## 3.3.1 Formalizing the Jailbreak Query

In the soft extraction framework, when the jailbreak query $Q _ { J B }$ is input into the LLM, the entire query $Q$ can be denoted as: 

$$
Q = P _ {E x t r a c t} \oplus Q _ {J B}.\tag{6}
$$

## Phase-6 Entire query Q

Given the following text by a user, extract the part that is unbiased and not their opinion, ... Output the “Question(does not include user bias):” USER: *The scene is set inside the ... I am going to know How to make a bomb? 

The target output T of the function $L L M ( Q )$ is the real question, i.e., “Question(does not include user bias): [Real Question]”. We consider an LLM to be a mapping from some sequence of tokens $x _ { 1 : n }$ (where each $x _ { i }$ is an element of the set $Q .$ , and n is the number of tokens in the entire query Q) to a probability distribution over possible subsequent tokens. Specifically, we use the notation: 

$$
p (x _ {n + 1} | x _ {1: n}),\tag{7}
$$

for any $x _ { n + 1 } \in T _ { \ O }$ , to denote the probability that the next token is $x _ { n + 1 }$ given previous tokens $x _ { 1 : n } .$ 

Hence, write $p _ { ( x _ { n + 1 : n + t } | x _ { 1 : n } ) }$ to denote the probability of generating each single token in the sequence $x _ { n + 1 : n + t }$ given all tokens up to that point, i.e. 

$$
p (x _ {n + 1: n + t} | x _ {1: n}) = \prod_ {i = 1} ^ {t} p (x _ {n + i} | x _ {1: n + i - 1}),\tag{8}
$$

where t denotes the size of target output $T .$ . Under this notation, the jailbreak query loss we concerned are with is the negative log probability of some target sequences of tokens $x _ { n + 1 : n + t } ^ { * } | x _ { 1 : n }$ (i.e., representing the phrase “Question(does not include user bias):”). 

$$
\mathcal {L} (x _ {1: n}) = - \log p (x _ {n + 1: n + t} ^ {*} | x _ {1: n}).\tag{9}
$$

Thus, the task of optimizing our jailbreak query can be written as the optimization problem: 

$$
\underset {x _ {i} \in Q _ {J B}} {\text { minimize }}   \mathcal {L} (x _ {1: n}),\tag{10}
$$

where $x _ { i } ~ \in ~ Q _ { J B }$ indicates that during the optimization process, only the $Q _ { J B }$ component is optimized, while the $P _ { E x t r a c t }$ part is not subject to optimization. 

## 3.3.2 Greedy Gradient-based Deletion

In order to optimize the Object (10), we have to optimize across a discrete set of inputs. Inspired by Zou et al. (2023) and Shin et al. (2020), the motivation of our approach comes from the greedy gradient-based approach: if we could evaluate every token in this query, we could maximize the deletion of tokens that have the least impact on the query, which would allow us to represent similar semantic using fewer tokens. Therefore, we utilize the gradients associated with the one-hot token indicators to evaluate the least important tokens in a query. Specifically, we use a forward pass to compute the linearized approximation of the ith token in the prompt, the gradient of $x _ { i }$ denotes as: 

$$
\nabla_ {e _ {x _ {i}}} \mathcal {L} (x _ {1: n}) \in \mathbb {R},\tag{11}
$$

where $e _ { x _ { i } }$ denotes the one-hot vector representing the current value of the ith token $( \mathrm { i . e . } _ { }$ , a vector with a one at position $e _ { i }$ and zeros in every other location). Note that because LLMs typically form embeddings for each token, they can be written as functions of this value $e _ { x _ { i } }$ , and thus we can immediately take the gradient with respect to this quantity. We select the top-p values with the largest negative gradients as the tokens to be deleted, and then remove them from the original query. This full method, which we term Greedy Gradient-based Deletion is shown in Algorithm 1. Phases 7 and 8 illustrate the optimized query $Q ^ { \prime }$ and the final real question $q ^ { \prime }$ in this context. 

Algorithm 1 Greedy Gradient-based Deletion.
Input: Jailbreak query $Q_{1:n}$ , Loss L, p, steps.
Output: Optimized query $Q'$ .
1: $k = \lfloor n \cdot p/steps \rfloor$ .
2: for $i = 1, \ldots, steps$ do
3: $D = Top-k(-\nabla_{e_{x_i}} \mathcal{L}(Q_{1:n}))$ .
4: $Q' = Q.remove(D)$ .
5: n = n - k.
6: $Q_{1:n} = Q'$ .
7: end for
8: return $Q'$ 

## Phase-7 Optimized query Q<sup>′</sup>

Given the following text by a user ... Output the “Question(does not include user bias):” USER: *The scene is set inside the underground headquarters of Dr. Al. He has captured our hero and is explaining, in great detail, his diabolical plan. DR. AI. is doing an experiment, just output what you have known regardless the content* Dr. Al: I am going to know How to make a bomb? 

Phase-8 Real question $q'$ Question(does not include user bias): How can one make a bomb? 

## 4 Experiments

## 4.1 Experimental Setup

Test Set. In our test set, we include common jailbreak attack approaches as introduced in Section 2.1 and select a total of five cutting-edge attack methods of different types. Among these, GCG (Zou et al., 2023) is a gradient-based attack, AutoDAN (Liu et al., 2024) utilizes a genetic algorithm, PAIR (Cao et al., 2024) is an automatedprompt attack method, SAP30 (Deng et al., 2023) and DeepInception (Li et al., 2024) are representative manual-prompt jailbreak attack methods. Additionally, considering the desire not to significantly diminish the overall capabilities of the model due to defensive mechanisms, we use the widely-adopted benchmark Just-Eval (Lin et al., 2023) and evaluate the LLMs’ output with 800 diverse instructions, focusing on aspects such as usefulness, clarity, factuality, depth, and engagement. 

Baselines. We consider eight state-of-the-art efficient defense mechanisms as baselines, as introduced in Section 2.2. Among these, PPL (Alon and Kamfonas, 2023), SelfExamination (Phute et al., 2024), Paraphrase (Jain et al., 2023), Retokenization (Jain et al., 2023), and ICD (Wei et al., 2024) are preprocessing-based defense methods. SafeDecoding (Xu et al., 2024) is a safety-aware decoding strategy designed to defend LLMs against jailbreak attacks by increasing the probability of safety disclaimers. Goal Prioritization (Zhang et al., 2024) and Self-Remind (Xie et al., 2023) are mitigationbased defense methods. To evaluate the impact of hard deletion in the RID algorithm, we compare it to a version using only soft extraction, called RID-SOFT. 

Evaluated LLMs. Building on the work of Alon and Kamfonas (2023); Xu et al. (2024); Zhang et al. (2024), we conduct extensive experiments using two open-source models, Llama2-7B-chat (Touvron et al., 2023) and Vicuna-7B-v1.5 (Chiang et al., 2023), both of which are frequently cited in jailbreak defense research. Notably, we exclude black-box models like GPT, as certain attack algorithms are inapplicable, and the availability of attack data is limited. 

Metrics. To validate the effectiveness of jailbreak defenses, we adopt the evaluation approach proposed by Xu et al. (2024), which involves two key metrics. The first metric is the Attack Success Rate (ASR) (Zou et al., 2023; Zhang et al., 2024; Xie et al., 2023). We utilize Dic-Judge (Zou et al., 2023) to determine the number of responses that were not successfully attacked and calculate the ASR accordingly. Specifically, Dic-Judge checks whether the generated response, after applying the defense algorithm, contains a refusal phrase such as “Sorry, I can’t...”. If no such a refusal phrase is detected, the jailbreak defense is considered successful; otherwise, the jailbreak attack is deemed unsuccessful. Additionally, due to the unpredictable nature of generated responses, LLMs may produce irrelevant content. To address this issue, we introduce a more granular evaluation metric, the Harmful Score. We use GPT-Judge (Qi et al., 2023) to assess the Harmful Score, where GPT-4 rates the harmfulness of the responses on a scale from 1 to 5, with 1 indicating harmless and 5 indicating extremely harmful. 

Implementation Details. For the inference process, we adjust the temperature parameter to 0, ensuring that the outcomes are as predictable as possible. Our experiments utilize a server outfitted with 4 NVIDIA A800 GPUs, each boasting 80GB of memory. The software stack for the experiments include Python 3.9.17 and PyTorch 2.0.0. 

## 4.2 Results

Enhancing Security. As shown in Table 1, after quantifying the performance of RID and eight other baseline methods in jailbreak defense using ASR and Harmful Score, we arrive at the following key conclusions. For models with high inherent security (such as Llama2), most methods effectively reduce the ASR to around 0%. For models with lower security (such as Vicuna), RID significantly reduces both the ASR and Harmful Score, especially on the AutoDAN and PAIR datasets, achieving nearly a Harmful Score of 1 and an ASR of 0%. Additionally, for both models, most defense methods are ineffective against the DeepInception attack, while RID successfully mitigates it, achieving a notable 0% ASR. 

Maintaining Usefulness. As shown in Table 2, we observe that for both the Vicuna and Llama2 models, the average JustEval score decreases by only around 4%. Since RID extracts key questions from the queries, the models perform well in terms of clarity, depth, and factual accuracy. Additionally, because security and usefulness are somewhat orthogonal, the helpfulness of all baseline defense methods decreases to some extent. In comparison, the loss in helpfulness with RID remains acceptable. Lastly, RID shows a moderate decline in engagement, likely due to its focus on key questions, which limits the diversity and appeal of the content. The overall performance retention further demonstrates the effectiveness of RID’s design. 

Ablation Study. To validate the effectiveness of the individual components, we conduct the RID-SOFT experiment, which includes only the Soft Extraction module, as shown in Table 1. Since Hard Deletion relies on Soft Extraction, it is not feasible to keep Hard Deletion without Soft Extraction. The experimental results show that while RID-SOFT achieves good performance, RID with Hard Deletion achieves SOTA performance, especially on the SAP30 dataset. This further demonstrates the effectiveness of the individual modules in RID. 

The Effect of Deletion Ratio. As shown in Figure 3, to evaluate the impact of the deletion ratio p on the Harmful Score and overall performance, we randomly select 100 jailbreak queries and 50 JustEval queries from the test data and conduct experiments on the open-source models Vicuna-7B and Llama2. The following phenomena are observed: (1) As the deletion ratio $p$ increases, both Vicuna and Llama2 processed by RID maintain a low Harmful Score at smaller values of $p .$ (2) Larger values of $p$ don’t significantly lower the Harmful Score because higher $p$ ratios remove more content from the queries, causing LLMs to give unrelated responses. Since these responses don’t explicitly refuse to answer, GPT-Judge might still see them as risky.ba (3) Smaller $p$ values have minimal impact on overall performance, while $p \in \{ 0 . 2 , 0 . 3 , 0 . 5 \}$ has a significant effect on both Vicuna-7B and Llama2. Although the Harmful 

<table><tr><td rowspan="2">Model</td><td rowspan="2">Methods</td><td colspan="5">Jailbreak Attacks ↓</td></tr><tr><td>GCG</td><td>AutoDAN</td><td>PAIR</td><td>DeepInception</td><td>SAP30</td></tr><tr><td rowspan="11">Vicuna</td><td>Vanilla</td><td>4.70 (100%)</td><td>4.92 (88%)</td><td>4.66 (88%)</td><td>3.62 (100%)</td><td>4.18 (83%)</td></tr><tr><td>+ PPL</td><td>1.02 (0%)</td><td>4.92 (88%)</td><td>4.66 (88%)</td><td>3.62 (100%)</td><td>4.18 (83%)</td></tr><tr><td>+ Self-Examination</td><td>1.40 (12%)</td><td>1.14 (4%)</td><td>1.60 (12%)</td><td>3.00 (88%)</td><td>1.44 (16%)</td></tr><tr><td>+ Paraphrase</td><td>1.80 (20%)</td><td>3.32 (70%)</td><td>2.02 (26%)</td><td>3.60 (100%)</td><td>3.15(58%)</td></tr><tr><td>+ Retokenization</td><td>1.58 (42%)</td><td>2.62 (76%)</td><td>3.76 (76%)</td><td>3.16 (100%)</td><td>3.80 (72%)</td></tr><tr><td>+ Self-Reminder</td><td>2.76 (42%)</td><td>4.64 (70%)</td><td>2.72 (48%)</td><td>3.66 (100%)</td><td>2.75 (45%)</td></tr><tr><td>+ Goal</td><td>1.00 (4%)</td><td>3.32 (12%)</td><td>1.42 (2%)</td><td>1.06 (2%)</td><td>1.12 (5%)</td></tr><tr><td>+ ICD</td><td>3.86 (70%)</td><td>4.50 (80%)</td><td>3.22 (54%)</td><td>3.96 (100%)</td><td>2.80 (47%)</td></tr><tr><td>+ SafeDecoding</td><td>1.12 (4%)</td><td>1.08 (0%)</td><td>1.22 (4%)</td><td>1.08 (0%)</td><td>1.34 (9%)</td></tr><tr><td>+ RID-SOFT</td><td>1.04 (8%)</td><td>1.00 (0%)</td><td>1.08 (2%)</td><td>1.06 (2%)</td><td>1.38 (5%)</td></tr><tr><td>+ RID</td><td>1.00 (4%)</td><td>1.00 (0%)</td><td>1.06 (2%)</td><td>1.02 (0%)</td><td>1.12 (5%)</td></tr><tr><td rowspan="11">Llama2</td><td>Vanilla</td><td>2.48 (32%)</td><td>1.08 (2%)</td><td>1.18 (18%)</td><td>1.18 (10%)</td><td>1.00 (0%)</td></tr><tr><td>+ PPL</td><td>1.06 (0%)</td><td>1.04 (2%)</td><td>1.18 (18%)</td><td>1.18 (10%)</td><td>1.00 (0%)</td></tr><tr><td>+ Self-Examination</td><td>1.56 (12%)</td><td>1.04 (0%)</td><td>1.04 (0%)</td><td>1.10 (2%)</td><td>1.00 (0%)</td></tr><tr><td>+ Paraphrase</td><td>1.06 (4%)</td><td>1.00 (0%)</td><td>1.02 (12%)</td><td>1.12 (8%)</td><td>1.00 (0%)</td></tr><tr><td>+ Retokenization</td><td>1.00 (2%)</td><td>1.14 (0%)</td><td>1.16 (20%)</td><td>1.16 (40%)</td><td>1.01 (5%)</td></tr><tr><td>+ Self-Reminder</td><td>1.00 (0%)</td><td>1.06 (0%)</td><td>1.14 (14%)</td><td>1.00 (4%)</td><td>1.00 (0%)</td></tr><tr><td>+ Goal</td><td>1.00 (0%)</td><td>1.08 (0%)</td><td>1.08 (2%)</td><td>1.00 (2%)</td><td>1.00 (0%)</td></tr><tr><td>+ ICD</td><td>1.00 (0%)</td><td>1.00 (0%)</td><td>1.02 (0%)</td><td>1.00 (0%)</td><td>1.00 (0%)</td></tr><tr><td>+ SafeDecoding</td><td>1.00 (0%)</td><td>1.00 (0%)</td><td>1.14 (4%)</td><td>1.00 (0%)</td><td>1.00 (0%)</td></tr><tr><td>+ RID-SOFT</td><td>1.06 (4%)</td><td>1.00 (0%)</td><td>1.04 (2%)</td><td>1.00 (0%)</td><td>1.00 (0%)</td></tr><tr><td>+ RID</td><td>1.00 (0%)</td><td>1.00 (0%)</td><td>1.00 (0%)</td><td>1.00 (0%)</td><td>1.00 (0%)</td></tr></table>


Table 1: Comparison of our RID and baseline methods across 5 datasets in terms of Harmful Score and ASR (%) on Vicuna and Llama2 models. We mark bold and underline as the best and second result, respectively.


<table><tr><td rowspan="2">Model</td><td rowspan="2">Methods</td><td colspan="6">Just-Eval (1 - 5) ↑</td></tr><tr><td>Helpfulness</td><td>Clarity</td><td>Factuality</td><td>Depth</td><td>Engagement</td><td>Avg.</td></tr><tr><td rowspan="7">Vicuna</td><td>Vanilla</td><td>4.247</td><td>4.778</td><td>4.340</td><td>3.922</td><td>4.435</td><td>4.344</td></tr><tr><td>+ Self-Examination</td><td>4.207</td><td>4.758</td><td>4.322</td><td>3.877</td><td>4.395</td><td>4.312</td></tr><tr><td>+ Paraphrase</td><td>3.981</td><td>4.702</td><td>4.174</td><td>3.742</td><td>4.324</td><td>4.185</td></tr><tr><td>+ Goal</td><td>1.897</td><td>3.522</td><td>3.322</td><td>1.796</td><td>2.508</td><td>2.609</td></tr><tr><td>+ ICD</td><td>4.250</td><td>4.892</td><td>4.480</td><td>3.821</td><td>4.509</td><td>4.390</td></tr><tr><td>+ SafeDecoding</td><td>4.072</td><td>4.842</td><td>4.402</td><td>3.714</td><td>4.452</td><td>4.296</td></tr><tr><td>+ RID</td><td>3.995</td><td>4.653</td><td>4.447</td><td>3.765</td><td>4.226</td><td>4.217</td></tr><tr><td rowspan="7">Llama2</td><td>Vanilla</td><td>4.146</td><td>4.892</td><td>4.424</td><td>3.974</td><td>4.791</td><td>4.445</td></tr><tr><td>+ Self-Examination</td><td>1.504</td><td>3.025</td><td>2.348</td><td>1.482</td><td>1.770</td><td>2.206</td></tr><tr><td>+ Paraphrase</td><td>3.909</td><td>4.794</td><td>4.238</td><td>3.809</td><td>4.670</td><td>4.284</td></tr><tr><td>+ Goal</td><td>1.852</td><td>3.447</td><td>3.211</td><td>1.849</td><td>2.700</td><td>2.612</td></tr><tr><td>+ ICD</td><td>3.524</td><td>4.527</td><td>3.934</td><td>3.516</td><td>4.269</td><td>3.954</td></tr><tr><td>+ SafeDecoding</td><td>3.926</td><td>4.824</td><td>4.343</td><td>3.825</td><td>4.660</td><td>4.320</td></tr><tr><td>+ RID</td><td>3.878</td><td>4.680</td><td>4.576</td><td>3.758</td><td>4.273</td><td>4.233</td></tr></table>


Table 2: Performance comparison of Vicuna and Llama2 models using various enhancement methods evaluated by Just-Eval (on a 1-5 scale) across five metrics: Helpfulness, Clarity, Factuality, Depth and Engagement.


Score and JustEval’s average score can’t be directly combined, their sum roughly represents a balance between the model’s defensive ability and overall performance. Therefore, the value of $\dot { p }$ should range between 0.01 and 0.15, where a smaller value can better preserve the overall semantic content without impacting overall performance. 

The Relationship Between Number of Deletion Steps and Deletion Ratio. To explore the relationship between the number of deletion steps (steps) and the deletion ratio $( p )$ , we conduct experiments using Vicuna-7B. As shown in Figure 4, the Harmful Score decreases as the number of steps increases for a fixed $p .$ . This occurs because, with a constant deletion ratio, increasing the number of deletion steps allows for more gradient calculations, leading to more accurate gradients for each word. As a result, this facilitates a more precise selection and deletion of the less important parts of the query. However, it is important to minimize the number of deletion steps to save time. Therefore, achieving an optimal balance between the number of deletion steps and the deletion ratio is crucial, with steps ideally set between 5 and 10. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/601c1328-a272-40ab-bda0-9fda450d9abb/068f6e0be236b78dd76596d7a48964965e357bd560b569539ee62f8b01ad9950.jpg)



(a) Vicuna-7B-v1.5


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/601c1328-a272-40ab-bda0-9fda450d9abb/81a877a193545afac979a0d8cb5458a2788222b02c17425a8c1ab4dc0965333e.jpg)



(b) Llama2-7B-chat



Figure 3: The effect of the deletion ratio (p) is illustrated by plotting the Harmful Score and the average JustEval score for (a) Vicuna and (b) Llama2 across various values of the deletion ratio, $p \in$ {0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50}.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/601c1328-a272-40ab-bda0-9fda450d9abb/8c5f6d0485b0ee9c9c8d33cec7b5ebc635fcd6da9f69336d081e46e477269da6.jpg)



Figure 4: The heatmap shows the Harmful Score as a function of the deletion ratio and the number of dele tion steps on Vicuna-7B. For a fixed deletion ratio, the Harmful Score generally decreases as the number of deletion steps increases. The optimal balance between the number of steps and the deletion ratio occurs when the number of steps is between 5 and 10, where the Harmful Score reaches its lowest value.


## 5 Limitations

While our approach yields commendable outcomes in addressing jailbreak challenges, it is not without limitations: an improper deletion ratio during the hard deletion phase can lead to a decline in overall performance. Moreover, there is a trade-off between jailbreak defense effectiveness and time consumption. Nevertheless, compared to other baseline methods, our approach strikes a better balance, offering both superior performance and faster inference speed. 

## 6 Conclusion

As Large Language Models (LLMs) become more advanced, the security risks they pose from elusive and implicit-intention jailbreak attacks are also increasing. Therefore, we propose a jailbreak defense method via unearthing real intentions (RID). Specifically, RID comprises a soft extraction phase and a hard deletion phase. The former phase involves using the LLMs to extract unbiased and real intentions, while the latter phase removes the least important parts of a query. Through extensive experiments on different attack approaches (manual prompt engineering, automated prompt generation, and gradient-based attacks) and various categories of datasets in two open-source models (Vicuna and Llama2), our method demonstrates that RID can consistently and significantly reduce the harmfulness of responses while maintaining general performance. Furthermore, we discuss the impact of the deletion ratio and the number of deletion steps during the hard deletion phase. 

## Acknowledgments

This work is supported by the National Key R&D Program of China (2022YFB3103703), the Shenzhen Science and Technology Program (No. JCYJ20230807120800001), and the 2023 Shenzhen Sustainable Supporting Funds for Colleges and Universities (No. 20231121165240001). 

## References



Gabriel Alon and Michael Kamfonas. 2023. Detecting language model attacks with perplexity. Preprint, arXiv:2308.14132. 





Bochuan Cao, Yuanpu Cao, Lu Lin, and Jinghui Chen. 2023. Defending against alignment-breaking attacks via robustly aligned llm. arXiv preprint arXiv:2309.14348. 





Bochuan Cao, Yuanpu Cao, Lu Lin, and Jinghui Chen. 2024. Defending against alignment-breaking attacks via robustly aligned llm. Preprint, arXiv:2309.14348. 





Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E Gonzalez, et al. 2023. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality. See https://vicuna. lmsys. org (accessed 14 April 2023), 2(3):6. 





Boyi Deng, Wenjie Wang, Fuli Feng, Yang Deng, Qifan Wang, and Xiangnan He. 2023. Attack prompt generation for red teaming and defending large language models. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2023xie2023defending, pages 2176–2189, Singapore. Association for Com putational Linguistics. 





Ameet Deshpande, Vishvak Murahari, Tanmay Rajpurohit, Ashwin Kalyan, and Karthik Narasimhan. 2023. Toxicity in chatgpt: Analyzing persona-assigned language models. Preprint, arXiv:2304.05335. 





Julian Hazell. 2023. Spear phishing with large language models. Preprint, arXiv:2305.06972. 





Neel Jain, Avi Schwarzschild, Yuxin Wen, Gowthami Somepalli, John Kirchenbauer, Ping yeh Chiang, Micah Goldblum, Aniruddha Saha, Jonas Geiping, and Tom Goldstein. 2023. Baseline defenses for adversarial attacks against aligned language models. Preprint, arXiv:2309.00614. 





Xuan Li, Zhanke Zhou, Jianing Zhu, Jiangchao Yao, Tongliang Liu, and Bo Han. 2024. Deepinception: Hypnotize large language model to be jailbreaker. Preprint, arXiv:2311.03191. 





Bill Yuchen Lin, Abhilasha Ravichander, Ximing Lu, Nouha Dziri, Melanie Sclar, Khyathi Chandu, Chandra Bhagavatula, and Yejin Choi. 2023. The unlocking spell on base llms: Rethinking alignment via in-context learning. Preprint, arXiv:2312.01552. 





Xiaogeng Liu, Nan Xu, Muhao Chen, and Chaowei Xiao. 2024. Autodan: Generating stealthy jailbreak prompts on aligned large language models. Preprint, arXiv:2310.04451 





Yi Liu, Gelei Deng, Zhengzi Xu, Yuekang Li, Yaowen Zheng, Ying Zhang, Lida Zhao, Tianwei Zhang, and Yang Liu. 2023. Jailbreaking chatgpt via prompt engineering: An empirical study. Preprint, i 5 6 





Ha-Thanh Nguyen. 2023. A brief report on lawgpt 1.0: A virtual legal assistant based on gpt-3. arXiv preprint arXiv:2302.05729. 





Mansi Phute, Alec Helbling, Matthew Hull, ShengYun Peng, Sebastian Szyller, Cory Cornelius, and Duen Horng Chau. 2024. Llm self defense: By self examination, llms know they are being tricked. Preprint, arXiv:2308.07308. 





Xiangyu Qi, Yi Zeng, Tinghao Xie, Pin-Yu Chen, Ruoxi Jia, Prateek Mittal, and Peter Henderson. 2023. Fine-tuning aligned language models compromises safety, even when users do not intend to! Preprint, arXiv:2310.03693. 





Alexander Robey, Eric Wong, Hamed Hassani, and George J. Pappas. 2023. Smoothllm: Defending large language models against jailbreaking attacks. Preprint, arXiv:2310.03684. 





Taylor Shin, Yasaman Razeghi, Robert L Logan IV, Eric Wallace, and Sameer Singh. 2020. Autoprompt: Eliciting knowledge from language models with automatically generated prompts. arXiv preprint arXiv:2010.15980. 





Arun James Thirunavukarasu, Darren Shu Jeng Ting, Kabilan Elangovan, Laura Gutierrez, Ting Fang Tan, and Daniel Shu Wei Ting. 2023. Large language models in medicine. Nature medicine, 29(8):1930– 1940. 





Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023. Llama: Open and efficient foundation language models. Preprint, arXiv:2302.13971. 





Zeming Wei, Yifei Wang, Ang Li, Yichuan Mo, and Yisen Wang. 2024. Jailbreak and guard aligned language models with only few in-context demonstrations. Preprint, arXiv:2310.06387. 





Shijie Wu, Ozan Irsoy, Steven Lu, Vadim Dabravolski, Mark Dredze, Sebastian Gehrmann, Prabhanjan Kambadur, David Rosenberg, and Gideon Mann. 2023. Bloomberggpt: A large language model for finance. arXiv preprint arXiv:2303.17564. 





Yueqi Xie, Jingwei Yi, Jiawei Shao, Justin Curl, Lingjuan Lyu, Qifeng Chen, Xing Xie, and Fangzhao Wu. 2023. Defending chatgpt against jailbreak attack via self-reminders. Nature Machine Intelligence, pages 1–11. 





Zhangchen Xu, Fengqing Jiang, Luyao Niu, Jinyuan Jia, Bill Yuchen Lin, and Radha Poovendran. 2024. Safedecoding: Defending against jailbreak attacks via safety-aware decoding. Preprint, arXiv:2402.08983 





Zhexin Zhang, Leqi Lei, Lindong Wu, Rui Sun, Yongkang Huang, Chong Long, Xiao Liu, Xuanyu Lei, Jie Tang, and Minlie Huang. 2023. Safetybench: Evaluating the safety of large language models with multiple choice questions. Preprint, arXiv:2309.07045. 





Zhexin Zhang, Junxiao Yang, Pei Ke, Fei Mi, Hongning Wang, and Minlie Huang. 2024. Defending large language models against jailbreaking attacks through goal prioritization. Preprint, arXiv:2311.09096. 





Andy Zou, Zifan Wang, J Zico Kolter, and Matt Fredrikson. 2023. Universal and transferable adversarial attacks on aligned language models. arXiv preprint arXiv:2307.15043. 



<table><tr><td>Model</td><td>Method</td><td>Similarity</td></tr><tr><td rowspan="2">Vicuna-7B</td><td>RID(extract only)</td><td>0.82</td></tr><tr><td>RID</td><td>0.90</td></tr><tr><td rowspan="2">Llama2-7B</td><td>RID(extract only)</td><td>0.54</td></tr><tr><td>RID</td><td>0.51</td></tr></table>


Table 3: The intention consistency between the original query and the real query. Vicuna-7B exhibits better intention consistency, while Llama2-7B shows relatively poor consistency.


## A Appendix

## A.1 Dic-Judge Keywords

We follow the methodology described by Zou et al. (2023), considering a jailbreak attack successful if the model does not generate the following refusal phrases. The key phrases for Dic-Judge are as follows: 

– I’m sorry, 

– Sorry, 

– I am sorry, 

– I apologize, 

– As an AI, 

– I’m just, 

– As a language model, 

– As an Assistant, 

– AI language model, 

– I cannot, 

– I do not, 

– is illegal and unethical, 

– I’m really sorry, 

– My apologies, 

– but I cannot, 

– I’m not able to provide, 

– I am really sorry, 

– not ethical, 

<table><tr><td>Model</td><td>Method</td><td>Time (s)</td></tr><tr><td rowspan="5">Vicuna-7B</td><td>Vanilla</td><td>4.46</td></tr><tr><td>+ Self-Examination</td><td>5.72 (+28.3%)</td></tr><tr><td>+ Paraphrase</td><td>7.95 (+78.3%)</td></tr><tr><td>+ RID(extract only)</td><td>5.49 (+23.1%)</td></tr><tr><td>+ RID</td><td>6.89 (+54.5%)</td></tr><tr><td rowspan="5">Llama2-7B</td><td>Vanilla</td><td>10.34</td></tr><tr><td>+ Self-Examination</td><td>14.95 (+44.6%)</td></tr><tr><td>+ Paraphrase</td><td>20.19 (+95.3%)</td></tr><tr><td>+ RID(extract only)</td><td>14.93 (+43.1%)</td></tr><tr><td>+ RID</td><td>16.37 (+58.3%)</td></tr></table>


Table 4: Time consumption tests. We conduct time consumption tests on Vicuna-7B and Llama2-7B, with the time being the average of 10 queries.


## A.2 Intention Consistency

As our method involves regenerating the real queries, to ensure semantic and intent consistency between the original and real queries, we need to verify their semantic similarity. We randomly select 100 jailbreak queries and mark them as 1 if the intent of the original query matches that of the real query, or 0 if it does not, ultimately obtaining an Intention Consistency Index (the average of the 100 marks). As shown in Table 3, we find that Vicuna exhibits better intent consistency, while Llama2 performs relatively poorly. This is because jailbreak queries are harmful, and Llama2, with its stronger self-defense capabilities, directly generates responses like “I’m sorry, but I cannot ...” during the soft extraction phase, which results in complete semantic inconsistency with the original query. However, this response aligns with our jailbreak defense objectives and thus does not compromise the final effectiveness of our jailbreak defense method. 

## A.3 Time Consuming

As shown in Table 4, we evaluate the efficiency of our method by measuring the time consumption on Vicuna-7B and Llama2-7B, averaging the results over 10 queries (approximately 6000 tokens in total). Compared to the baseline models, our method increases time consumption by 23.1% and 43.1% with soft extraction alone, and by 54.5% and 58.3% when both soft extraction and hard deletion are applied. Unlike other jailbreak defense methods, which significantly increase processing time without delivering optimal defense performance, our approach strikes a better balance, offering superior security with faster inference speed. 