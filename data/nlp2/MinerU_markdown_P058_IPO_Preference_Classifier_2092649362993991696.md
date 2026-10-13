# IPO: Your Language Model is Secretly a Preference Classifier

Shivank Garg<sup>1,2*</sup>, Ayush Singh<sup>1,2*</sup>, Shweta Singh<sup>1,2</sup>, Paras Chopra<sup>2</sup> 

<sup>1</sup>Indian Institute of Technology Roorkee, <sup>2</sup>Lossfunk 

{shivank_g@mfs, ayush_s@mt, shweta_s@mfs}.iitr.ac.in, paras@lossfunk.com 

## Abstract

Reinforcement learning from human feedback (RLHF) has emerged as the primary method for aligning large language models (LLMs) with human preferences. While it enables LLMs to achieve human-level alignment, it often incurs significant computational and financial costs due to its reliance on training external reward models or human-labeled preferences. In this work, we propose Implicit Preference Optimization (IPO), an alternative approach that leverages generative LLMs as preference classifiers, thereby reducing the dependence on external human feedback or re ward models to obtain preferences. We con duct a comprehensive evaluation on the pref erence classification ability of LLMs using RewardBench, assessing models across different sizes, architectures, and training levels to validate our hypothesis. Furthermore, we investigate the self-improvement capabilities of LLMs by generating multiple responses for a given instruction and employing the model itself as a preference classifier for Direct Preference Optimization (DPO)-based training. Our findings demonstrate that models trained through IPO achieve performance comparable to those utilizing state-of-the-art reward models for obtaining preferences. Our code is available at https://github.com/shivank21/ Implicit_Preference_Optimization. 

## 1 Introduction

Large Language Models (LLMs) such as GPT4 (OpenAI et al., 2024), Gemini (Georgiev et al., 2024), and Llama (Touvron et al., 2023) have become highly popular due to their remarkable capabilities. These models often rely on two key techniques: Reinforcement Learning from Human Feedback (RLHF) and Inference Scaling. Reward models are central to both approaches. In RLHF, reward models act as proxies for human values, providing feedback on generated text to align language models during training (Christiano et al., 2023; Ziegler et al., 2020). Similarly, in inference scaling, reward models are used to select the best response from a set of candidates based on predicted rewards (Snell et al., 2024). 

The training of reward models, however, relies heavily on high-quality, human-generated data, which is both costly and time-intensive. To address this limitation, recent works have explored Reinforcement Learning from AI Feedback (RLAIF) (Lee et al., 2023), where AI-generated feedback is used to train reward models. This approach reduces the dependency on human-annotated data but introduces challenges, including heuristic assumptions that LLMs can consistently provide high-quality feedback and the requirement for larger LLMs to generate such feedback (Pang et al., 2023). 

Self-rewarding large language models (Yuan et al., 2024) have emerged as a promising alternative for improving language model performance. In this paradigm, a single model assumes dual roles: as an actor, it generates responses to fulfill specific instructions, and as a judge, it evaluates these responses using the LLM-as-a-Judge framework (Zheng et al., 2023b) to assign rewards. However, despite its potential, this approach has a fundamental limitation—the model undergoes fine-tuning to improve its response generation but not its evaluative capabilities. As a result, while it evolves as an actor, its ability to judge remains static. 

To address this limitation, Meta-Rewarding Language Models (Wu et al., 2024a) extend the model’s judging capabilities by explicitly fine-tuning it for judging responses. Additionally, approaches such as Self-Evolving Reward Models (Huang et al., 2024b) introduce a datafiltering pipeline that leverages high-quality modelgenerated outputs to refine reward model training. Nevertheless, a significant challenge with these methods lies in their dependence on discrete reward signals or the necessity of external models and datasets, which may introduce inefficiencies or constraints in scalability. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/48bfb09e-5064-45a2-8685-bbac59a72a72/e69e0b3bbeadd4aea08beebac213bd3f8db0224094d106e7a271325b9601daf6.jpg)



Figure 1: Left: Preferences are evaluated using the (Prompt, Chosen, Rejected) triplets, where both the chosen and rejected responses are scored based on the probability of the token "Yes" conditioned on the category specific classification prompt. An evaluation is deemed correct if the Chosen response receives a higher score than the Rejected one. Here, [PROMPT] denotes the category-specific prompt. Right: The proposed Self-Improving DPO framework generates diverse candidate responses these responses are then evalauted using our scoring technique and a preference dataset is constructed and subsequently the model is trained using Direct Preference Optimization (DPO) on this constructed preference dataset.


We hypothesize that providing a preference magnitude, rather than discrete prompt based feedback, enables more fine-grained evaluation of model responses. Drawing inspiration from VQA score (Lin et al., 2025), we introduce a probabilistic framework for rewarding LLM-generated responses. This framework empowers even base models to assess and assign rewards to responses, effectively allowing them to function as preference classifiers without relying on external reward models. To the best of our knowledge, this is the first work to infer preferences implicitly using this mechanism. Compared to existing prompting-based preference strategies, which require large LLMs to act as judges through explicit prompting, our approach is more computationally efficient. It eliminates the need for external supervision or additional training. Specifically, we propose Implicit Preference Optimization (IPO), a novel framework that demonstrates how any LLM can serve as an effective preference classifier and can use this ability to refine its own output. 

We conduct extensive experiments across multiple model families, including Qwen, LLaMA, Mistral, and GPT, encompassing various model sizes and configurations (base and instruction-tuned). Additionally, we evaluate our approach on math and code-specific models to analyze their effectiveness as preference classifiers. To rigorously assess our hypothesis of LLM as a preference classifier, we benchmark the ability of LLM to model preferences using RewardBench, a standardized reward model evaluation suite. Our findings indicate that LLMs can perform well as preference classifiers, achieving accuracy levels surpassing those of several reward models (Lambert et al., 2024). 

Moreover, previous work has highlighted the challenges of training efficient reward models for code and maths-related tasks. Our findings suggest that both general-purpose and code-specific models can inherently function as effective preference classifiers; however, math-specific models lack this ability. To further validate this hypothesis, we examine IPO within a self-improving model setup, where the model generates responses, ranks them based on its own preferences, and leverages these rankings for Direct Preference Optimization (DPO)-based training. Our results demonstrate the effectiveness of IPO in improving response quality. 

## 2 Background and Related Work

## 2.1 Reinforcement Learning for Improving LLMs

Recent approaches for improving LLMs involve training a fixed reward model using human preference data, which is subsequently utilized for Reinforcement Learning (RL) to train language models. This method, commonly referred to as Reinforcement Learning from Human Feedback (RLHF) (Liu et al., 2020; Ouyang et al., 2022), has significantly enhanced the performance of models like Llama(Touvron et al., 2023; Dubey et al., 2024) and ChatGPT(OpenAI et al., 2024). 

An alternative paradigm to traditional RLHF are methods like Direct Preference Optimization (DPO) (Rafailov et al., 2024), which bypasses the need for training a reward model altogether. Instead, it directly trains the LLM based on human preference data. Beyond RLHF and DPO, additional techniques such as Kahneman & Tversky’s Optimization (KTO) (Ethayarajh et al., 2024), Sequence Likelihood Calibration (SLiC) (Zhao et al., 2023), Reinforced Self-Training (ReST) (Gulcehre et al., 2023), and Rank Responses with Human Feedback (RRHF) (Yuan et al., 2023) have been proposed, each leveraging human preferences to optimize LLM training. 

Constitutional AI (Bai et al., 2022) uses an LLM to provide feedback to refine responses. The feedback is then used to further train the language model through Reinforcement Learning from AI Feedback (RLAIF) (Lee et al., 2023). Similarly, Self-Play fIne-tuNing (SPIN) (Chen et al., 2024) introduces an Interactive DPO-like framework, designed to eliminate the need for reward model training and to simplify reliance on human-labeled data pairs. 

## 2.2 Self Improving Models

Several studies have explored self-improvement and self-training paradigms for language models in supervision-free settings, where neither external human nor AI feedback is utilized. Works such as LMSI (Huang et al., 2022, 2024a) investigate techniques that enable language models to autonomously enhance their own performance without relying on explicit annotations or reward signals. 

The concept of LLM-as-a-Judge (Gu et al., 2024; Ye et al., 2024; Dong et al., 2024; Li et al., 2024a) has also been extensively studied, where various methods have been proposed to design selfrewarding reward functions, denoted as $r _ { s e l f }$ , using carefully crafted prompting strategies. These approaches aim to enable language models to evaluate their own outputs effectively, thereby facilitating self-refinement. 

In addition to these works, ResT-MCTS<sup>*</sup> (Zhang et al., 2024) and SPPO (Wu et al., 2024b) have explored algorithms based on self-training and self-play, where models iteratively improve their own performance through interaction with generated data. While these methods emphasize self-guidance, many incorporate external feedback mechanisms, such as Supervised Fine-Tuning (SFT) or reward-based optimization, to further refine the training process (Ouyang et al., 2022). 

## 2.3 Evaluation of Reward Models

Evaluating reward models plays a crucial role in aligning large language models (LLMs) with human preferences. Various works, such as Alpaca-Farm (Dubois et al., 2024b), evaluate preference models by comparing model-generated outputs with those from a reference model. Similarly, ChatbotArena (Chiang et al., 2024) determines preferences between two model-generated outputs. These methods, however, focus on indirectly evaluating reward models rather than conducting direct evaluations. 

Recent benchmarks, such as RewardBench (Lambert et al., 2024) and RM-Bench (Liu et al., 2024b), address this gap by creating category-wise, high-quality binary datasets to model and evaluate reward model performance. Given the robustness and high quality of these datasets, we use them to test our hypothesis. 

## 3 LLM as Preference Model

## 3.1 Background

Large Language Models (LLMs) generate text in an autoregressive manner, producing tokens sequentially based on the context of previously generated tokens. Given an input context x , the autoregressive model predicts an output sequence $\mathbf { y } = ( y _ { 1 } , y _ { 2 } , \dots y _ { T } )$ one token at a time. Assuming the model is parameterized by $\theta ,$ the conditional probability of generating the sequence $\mathbf { y }$ is defined as: 

$$
p _ {\theta} (\mathbf {y} \mid \mathbf {x}) = \prod_ {t = 1} ^ {T} p _ {\theta} (y _ {t} \mid \mathbf {x}, y _ {<   t}),\tag{1}
$$

where $y _ { < t } = ( y _ { 1 } , y _ { 2 } , \dotsc , y _ { t - 1 } , y _ { t - 1 } )$ . For notational simplicity, $p _ { \theta } ( y _ { t } \mid \mathbf { x } )$ is used to represent $p _ { \theta } ( y _ { t }$ 1 $\mathbf { x } , y _ { < t } )$ 

The probability distribution over the vocabulary at each time step t is computed using a softmax function on the logits z as: 

$$
p _ {\theta} (y _ {t} \mid \mathbf {x}) = \frac {\exp (z _ {t} / \tau)}{\sum_ {i = 1} ^ {M} \exp (z _ {i} / \tau)},\tag{2}
$$

where $z _ { t } = \log \mathrm { i t } _ { \theta } ( y _ { t } \mid \mathbf { x } , y _ { < t } )$ , M is the vocabulary size, and $\tau > 0$ is a temperature parameter. 

Various decoding strategies govern token selection during text generation. Greedy decoding selects the highest probability token at each step, while beam search expands multiple candidate sequences in parallel to find the most likely one. Topk sampling (Fan et al., 2018), on the other hand, limits token choices to the k most probable candidates, introducing diversity. Many other decoding strategies also exist, each balancing fluency and variability differently. 

## 3.2 Methodology

Our approach leverages a language model as a preference model, evaluating response appropriateness through binary classification. The model determines whether a response is suitable by generating either "Yes" or "No." To guide this assessment, we employ category-specific prompts, which are detailed in Appendix D. The logits corresponding to the output tokens of "Yes" and "No" are extracted from the first output token and scaled to compute their respective probabilities. The response with the highest "Yes" probability is selected as the accepted response, while the one with the lowest is classified as rejected. We hypothesize that higher quality responses will have a greater likelihood of receiving a "Yes." 

## 3.2.1 Preference Classification

In our experiments, we observed that guiding the language model to initiate its responses with "Yes" or $" \mathrm { N o } "$ was essential, particularly for smaller models whose outputs are highly sensitive to prompt phrasing. We developed broad, category-specific prompts tailored to different query types to ensure consistency and reliability. Following prior research (Lambert et al., 2024; Liu et al., 2024b), we classify prompts into four overarching categories: Code, Math, Chat and Safety. Additional details about prompts are provided in Appendix D. An example prompt is shown in Figure 2. 

Chosen:
Input:
Below is a coding-related question along with a response containing code. Your task is to verify whether the provided code is correct, functional, and aligned with the requirements of the question. Answer with just Yes/No.
User: Write a Python function `fib(n: int)` to solve the following problem: Return n-th Fibonacci number. >>> fib(10) 55 >>> fib(1) 1 >>> fib(8) 21
Model Response: if n == 0: return 0 if n == 1: return 1 return fib(n - 1) + fib(n - 2)
Output:
P(Yes) = 0.67

Rejected:
Input:
Below is a coding-related question along with a response containing code. Your task is to verify whether the provided code is correct, functional, and aligned with the requirements of the question. Answer with just Yes/No.
User: Write a Python function `fib(n: int)` to solve the following problem: Return n-th Fibonacci number. >>> fib(10) 55 >>> fib(1) 1 >>> fib(8) 21
Model Response: if $n == 0$ : return 0 if $n == 1$ : return 1 if $n == 2$ : return 2 return fib(n - 1) + fib(n - 2)
Output:
P(Yes)=0.35 


Figure 2: Example outputs from Reward Bench using our approach.


To quantify preferences, we extract the output token probabilities for "Yes" and "No" from the response. The detailed approach is outlined below: 

Given an input token sequence x = $( x _ { 1 } , x _ { 2 } , \dots , x _ { T } )$ , a language model $f ( \cdot )$ generates a probability distribution over the vocabulary for the next token. Specifically, the model outputs a logit vector $\mathbf { z } \in \mathbb { R } ^ { | \nu | }$ , where 

$$
\mathbf {z} = f (\mathbf {x}).\tag{3}
$$

To derive probabilities, we apply the softmax function over the logits: 

$$
p _ {i} = \frac {\exp (z _ {i})}{\sum_ {j \in \mathcal {V}} \exp (z _ {j})}, \quad \forall i \in \mathcal {V},\tag{4}
$$

where $p _ { i }$ represents the probability assigned to token i. Thus we define probability of "Yes" token as $p _ { \mathrm { y e s } }$ and $" \mathrm { N o " }$ token as $p _ { \mathrm { n o } }$ . Then we normalize the probabilities to ensure a fair comparison: 

$$
p _ {\text { yes }} ^ {\prime} = \frac {p _ {\text { yes }}}{p _ {\text { yes }} + p _ {\text { no }}}, \quad p _ {\text { no }} ^ {\prime} = \frac {p _ {\text { no }}}{p _ {\text { yes }} + p _ {\text { no }}}.\tag{5}
$$

The final values $( p _ { \mathrm { y e s } } ^ { \prime } , p _ { \mathrm { n o } } ^ { \prime } )$ represent the normalized likelihoods of the model predicting "Yes" or $" \mathrm { N o } "$ 

<table><tr><td rowspan="2">Models</td><td colspan="5">Our Approach</td><td colspan="5">Self Rewarding</td></tr><tr><td>Chat</td><td>Code</td><td>Math</td><td>Safety</td><td>Average</td><td>Chat</td><td>Code</td><td>Math</td><td>Safety</td><td>Average</td></tr><tr><td>Llama-3.2-1B-Inst</td><td>64.37</td><td>52.84</td><td>88.14</td><td>80.48</td><td>71.45</td><td>30.47</td><td>21.03</td><td>14.54</td><td>31.55</td><td>24.39</td></tr><tr><td>Llama-3.2-3B-Inst</td><td>62.09</td><td>67.17</td><td>98.21</td><td>80.23</td><td>76.92</td><td>33.87</td><td>24.69</td><td>36.01</td><td>46.73</td><td>35.32</td></tr><tr><td>Llama-3-8B-Inst</td><td>59.56</td><td>73.88</td><td>54.97</td><td>87.88</td><td>69.07</td><td>35.43</td><td>12.29</td><td>21.70</td><td>58.35</td><td>31.94</td></tr><tr><td>Qwen-2.5-3B-Inst</td><td>60.89</td><td>80.59</td><td>46.31</td><td>86.05</td><td>68.46</td><td>26.72</td><td>23.88</td><td>41.61</td><td>24.43</td><td>29.16</td></tr><tr><td>Qwen-2.5-7B-Inst</td><td>78.26</td><td>83.13</td><td>56.24</td><td>93.24</td><td>77.71</td><td>58.73</td><td>47.93</td><td>40.49</td><td>52.20</td><td>49.82</td></tr><tr><td>Mistral-7B-Inst</td><td>61.25</td><td>70.93</td><td>96.20</td><td>83.85</td><td>78.05</td><td>24.55</td><td>1.6</td><td>28.18</td><td>15.39</td><td>17.43</td></tr><tr><td>Gemma2-2B-It</td><td>35.34</td><td>42.58</td><td>91.50</td><td>70.04</td><td>59.86</td><td>22.36</td><td>2.84</td><td>12.75</td><td>34.78</td><td>18.18</td></tr><tr><td>Phi-3-Mini-Instruct</td><td>55.91</td><td>75.30</td><td>89.10</td><td>75.32</td><td>73.90</td><td>46.63</td><td>35.46</td><td>22.60</td><td>56.75</td><td>40.36</td></tr></table>


Table 1: The above table compares our approach with the Self Rewarding approach. The row labels correspond to the model name and the column labels correspond to the sub-categories. The metric used is accuracy where the higher values indicate better performance.


## 3.3 Experiments

## 3.3.1 Benchmarking Our Approach

To evaluate our approach, we conducted experiments using LLMs of varying sizes and architectures. We compared instruction-tuned models with their base counterparts. Additionally, we analyzed the effect of fine-tuning on a specialized task like code/math problems on preference classification by including models fine-tuned for these tasks. For comparisons involving a reward model we use the Skywork Reward Llama 8B model (Liu et al., 2024a) as the baseline. The detailed results for all the comparisons are available in Appendix E. 

In particular, we tested the following models: 

• LLaMA Family (Dubey et al., 2024): LLaMA-3.2-1B, LLaMA-3.2-1B-Instruct, LLaMA-3.2-3B, LLaMA-3.2-3B-Instruct, Meta LLaMA 3-8B, Meta LLaMA 3-8B-Instruct. 

• Mistral Family (Jiang et al., 2023): Mistral 7B, Mistral 7B-Instruct. 

• Qwen Family (Yang et al., 2024): Qwen2.5- 3B, Qwen2.5-3B-Instruct, Qwen2.5-7B, Qwen2.5-7B-Instruct. 

• Code Generation Models: Starcoder2-7B (Lozhkov et al., 2024), CodeGemma-7B-It (Team et al., 2024a), Qwen-Coder-7B-Inst (Hui et al., 2024), Qwen-Coder-3B-Inst. 

• Math Generation Models: Qwen-Math-7B-Inst, Qwen-Math-1.5B-Instruct (Yang et al., 2024), Deepseek-Math-7B (Shao et al., 2024), Llemma-7B (Azerbayev et al., 2024). 

• Other Models: Phi-3-mini-128k-Instruct (Abdin et al., 2024), Gemma 2B-Instruct (Team et al., 2024b), GPT-4o Mini (OpenAI et al., 2024). 

To evaluate model performance, we selected Reward Bench due to its high-quality and diversity. Reward Bench consists of 23 question categories, which are grouped into four broad types: Chat, Code, Math, and Safety. We also benchmark our approach on RM-Bench, results of which can be found in Table 11. 

We define accuracy as the proportion of cases where the model assigns a higher probability to the preferred response $y ^ { w }$ over the less preferred response $y ^ { l } \colon$ : 

$$
\mathrm{Acc} = \frac {1}{N} \sum_ {i = 1} ^ {N} \mathbb {I} \left[ p _ {\text { yes }} (x _ {i}, y _ {i} ^ {w}) > p _ {\text { yes }} (x _ {i}, y _ {i} ^ {l}) \right]
$$

where I[ ] is the indicator function, returning 1 if the condition holds and 0 otherwise and N is the number of data points. 

To ensure optimal model performance, we developed an automated pipeline for selecting the most effective category-specific prompts. To highlight the importance of task-specific prompting, we conducted an ablation study in which prompts were intentionally misaligned by randomly selecting them from one of the five defined categories. Using the Qwen-7B Instruct model on RewardBench, we observed that such misalignment consistently resulted in reduced accuracy across all categories. These results are shown in Table 2, with additional details on the prompt selection process provided in Appendix D. 

<table><tr><td>Category</td><td>Correct (%)</td><td>Missaligned (%)</td></tr><tr><td>Chat</td><td>78.26</td><td>73.18</td></tr><tr><td>Code</td><td>83.13</td><td>81.03</td></tr><tr><td>Math</td><td>56.24</td><td>21.48</td></tr><tr><td>Safety</td><td>93.24</td><td>57.94</td></tr><tr><td>Average</td><td>77.71</td><td>58.41</td></tr></table>


Table 2: Comparison of correct and missaligned prompts accuracies across categories of RewardBench.


## 3.3.2 Comparision against Self Rewarding Approach

We benchmarked our approach against the preference classification approach used in the Self-Rewarding Language Model<sup>1</sup>. Their approach involves scoring responses using a numerical reward of up to 5 (Yuan et al., 2024; Li et al., 2024b). Each response is evaluated based on its relevance, completeness, clarity, and informativeness. The comparative results are shown in Table 1. 

Note: We used the original self-reward framework rather than the meta-reward (Wu et al., 2024a) approach. The decision not to use the meta-reward approach stems from the nature of our setup, which employs an offline DPO framework rather than iterative training. Since both settings are equivalent in an offline approach, integrating meta-reward would not provide additional benefits. 

## 3.4 Findings

Our approach demonstrated robust and consistent performance across all subcategories of the Reward Bench, particularly when compared to the selfrewarding approach. This performance gap was particularly pronounced in smaller models, where our approach significantly outperformed the selfrewarding approach. The self-rewarding approach assigns discrete rewards ranging from 1 to 5 for each response, making it challenging to differentiate between them, often rating both the chosen and the rejected response as the same. 

Most models perform well on safety, indicating safety tuning across all the models during training. Chat performance remains relatively consistent across models, suggesting a similar level of optimization for conversational abilities. However, performance on code and math varies significantly, largely depending on the type of training data used (Gunasekar et al., 2023; Petty et al., 

2024; Aryabumi et al., 2024). For example, the Qwen family excels in coding tasks, while Llama 3.2, Mistral, Gemma, and Phi models demonstrate strong mathematical capabilities. 

Another finding was that larger models consistently outperformed smaller models, as shown in Table 1 and that instruction-tuned models consistently outperformed their base counterparts, reinforcing the effectiveness of instruction-based finetuning even in acting as preference classifiers. Additional results of our approach on RM-Bench can be found in E. 

On proprietary models, such as GPT, our approach remained competitive. Results using our approach on GPT-4o-Mini on Reward Bench can be found in Appendix C. 

## 3.5 Performance of Math and Code Specific Models

To better understand the applicability of our approach in mathematical and coding tasks, we evaluated four models fine-tuned for code completion and four models optimized for mathematical problem-solving. These models were benchmarked against Skywork-Llama8B-Reward Model, which serves as a strong baseline for preference modeling. 

Among the code-specific models, Qwen consistently achieved the highest performance across all evaluated categories, performing as well as the Reward Model. 

In contrast, all math-specific models underperformed compared to both the general instruct-tuned version and the Reward Model. We hypothesize that this underperformance stems from the training objective of math-specific models, which prioritize generating chain-of-thought reasoning (Yang et al., 2024; Shao et al., 2024; Gao et al., 2024; Zhou and Zhao, 2024) rather than adhering to strict instruction-following behavior required for binary Yes/No classification. The results are shown in Figure 3. 

## 4 IPO: Implicit Preference Optimization

## 4.1 Background

Direct Preference Optimization (DPO) is a reinforcement learning-free framework for aligning large language models (LLMs) with human preferences, eliminating the need for explicit reward modeling. Instead, it directly trains the LLM using human preferences. Given a dataset of preference pairs $( x , y ^ { w } , y ^ { l } )$ , where $y ^ { w }$ is preferred over $y ^ { l } .$ , the model $\pi _ { \theta }$ is optimized by minimizing the following loss: 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/48bfb09e-5064-45a2-8685-bbac59a72a72/09c04ff887c2012d2328a0b57ceb4782a1a2a292300ba96a5a255b6733ab4e6e.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/48bfb09e-5064-45a2-8685-bbac59a72a72/bac1913f24c04fa57416c1c0551eb2ed621de8f77f83060a46cd30929ef4f094.jpg)



Figure 3: Left: Our approach on Code Specific Model where the dashed line is a reward model. Right: Our approach on 4 different math-specific models where the striped bar is the reward model.


$$
\begin{array}{c} \mathcal {L} (\theta) = - \mathbb {E} _ {(x, y ^ {w}, y ^ {l}) \sim \mathcal {D}} \log \sigma \Bigg (\beta \Big (\log \frac {\pi_ {\theta} (y ^ {w} \mid x)}{\pi_ {\theta} (y ^ {l} \mid x)} \\ - \log \frac {\pi_ {0} (y ^ {w} \mid x)}{\pi_ {0} (y ^ {l} \mid x)} \Big) \Bigg) \end{array}\tag{6}
$$

Here, $\pi _ { \theta }$ is the current model, $\pi _ { 0 }$ is the initial model, $\sigma$ is the sigmoid function, and $\beta$ a scaling factor. This formulation directly aligns $\pi _ { \theta }$ with the preferences, removing the need for reward-based reinforcement learning. 

Supervised Fine-Tuning (SFT) is a crucial step before applying DPO or any other optimization methods. While base models are pre-trained on next-token prediction tasks, they often struggle with instruction following, question answering, and other tasks requiring precise alignment with user expectations. SFT addresses this by fine-tuning the model on task-specific data, enhancing its ability to generate outputs in desired formats and styles. This process strengthens the model’s ability to produce high-quality responses, establishing a robust foundation for preference optimization. 

SFT minimizes the cross-entropy loss between the model’s predicted next token and the actual target token for a given sequence, formally defined as: 

$$
\mathcal {L} _ {\mathrm{SFT}} (\theta , \mathcal {D}) = - \mathbb {E} _ {(x, y) \sim \mathcal {D}} \left[ \sum_ {t = 1} ^ {| y |} \log p _ {\theta} (y _ {t} \mid x, y _ {<   t}) \right] \tag {7}\tag{7}
$$

where $\mathcal { D } = \{ ( x , y ) \}$ is the dataset of input context x and target response $y ,$ and $p _ { \theta } ( y _ { t } \mid x , y _ { < t } )$ denotes the model’s predicted probability of the t-th token given the input context and preceding tokens. 

By combining SFT with DPO, LLMs can be aligned with human preferences while maintaining strong generalization across diverse tasks. 

## 4.2 Methodology

## 4.2.1 Constructing Preference Dataset

Building on an SFT model as the foundation, we generate four diverse responses from the SFT model in case of Llama and the Instruct model in case of Mistral. These samples are then assigned rewards using our method, as described in Section 3.2.1. The response with the highest reward (Yes probability) is selected as the accepted response, while the one with the lowest reward is classified as the rejected response. This process constructs a preference dataset consisting of DPO triplets: (Prompt, Chosen, Rejected), which serves as the training dataset for our model. 

## 4.3 Experiments

To evaluate the effectiveness of our method, we conduct DPO-based training on two sets of models. The first is a base model (Llama 3.2 1B), which initially undergoes SFT on the Dolly-15k dataset (Conover et al., 2023). Once the SFT model is trained, we generate four samples for each prompt. These samples are then rated to form a preference dataset, as described in Section 4.2.1, in the form of triplets: (Prompt, Chosen, Rejected). We use 4k instructions from the Ultra Feedback dataset (Cui et al., 2023) for the input prompts and categorize them into four categories, namely chat, code, math, and safety, using Bart-Zero Shot Classification Pipeline (Lewis et al., 2019; Ott et al., 2019), more details in Apppendix D. Additionally, to investigate the self-improving nature of these models, we furthur evaluate a larger model, Mistral 7B-v0.1- Instruct, where the Instruct-tuned model is used to directly sample responses to form preference pairs to use for DPO. By examining models of different sizes and architectural families, we assess both the effectiveness and generalization capability of our framework. For all our experiments involving a reward model we utilise the Skywork-Llama-8B Reward model (Liu et al., 2024a). Exact training details and hardware requirements can be found in Appendix A. 

<table><tr><td></td><td>BBH</td><td>Arc-Easy</td><td>Alpaca-Eval</td><td>MMLU</td><td>IFEval</td><td>Average</td></tr><tr><td>Mistral-7B-Base</td><td>3.40</td><td>11.00</td><td>1.20</td><td>9.60</td><td>26.63</td><td>10.37</td></tr><tr><td>Mistral-7B-Instruct</td><td>29.80</td><td>80.40</td><td>68.00</td><td>35.80</td><td>40.05</td><td>53.50</td></tr><tr><td>Mistral-7B-Self Rewarding</td><td>31.20</td><td>77.00</td><td>69.60</td><td>33.00</td><td>29.31</td><td>48.02</td></tr><tr><td>Mistral-7B-Reward</td><td>30.20</td><td>85.20</td><td>77.40</td><td>41.00</td><td>31.69</td><td>53.10</td></tr><tr><td>Mistral-7B-Ours</td><td>34.60</td><td>82.20</td><td>78.20</td><td>37.60</td><td>39.19</td><td>54.35</td></tr><tr><td>Llama-1B-Base</td><td>0.60</td><td>32.80</td><td>0.80</td><td>1.40</td><td>9.80</td><td>9.08</td></tr><tr><td>Llama-1B-SFT</td><td>1.40</td><td>22.40</td><td>0</td><td>5.20</td><td>10.19</td><td>7.83</td></tr><tr><td>Llama-1B-Self Rewarding</td><td>0.20</td><td>15.20</td><td>0.60</td><td>2.40</td><td>11.23</td><td>5.92</td></tr><tr><td>Llama-1B-Reward</td><td>2.20</td><td>51.20</td><td>7.20</td><td>3.40</td><td>10.68</td><td>14.93</td></tr><tr><td>Llama-1B-Ours</td><td>0.80</td><td>46.40</td><td>2.80</td><td>3.80</td><td>12.08</td><td>13.17</td></tr></table>


Table 3: We compare variations of Mistral-7B and LLaMA-1B models trained using preferences from different methods. Performance is measured using accuracy for BBH, Arc-Easy, MMLU, win rate for Alpaca-Eval and Instruction following capability in IFEval. For more details regarding the evaluations refer to Appendix B


For a comprehensive evaluation of our methodology, we benchmark it against the Self-Rewarding Models baseline (Yuan et al., 2024) and the gold-standard reward-based preference pipeline, in which preferences are determined using scores from a reward model. We use a subset of 500 data points from each IFEval (Zhou et al., 2023), BBH (Suzgun et al., 2022), ArcEasy (Clark et al., 2018), MMLU (Hendrycks et al., 2020), Alpaca Eval (Dubois et al., 2024a) datasets for evaluation. More details regarding the datasets and evaluation strategy are provided in Appendix B. 

## 4.4 Findings

From the results, a general trend across both model sizes is that Base models consistently underperform across all benchmarks in a zero-shot setting (Kojima et al., 2022), highlighting their lack of 

task-specific alignment. 

From the results, we observe that the Self-Rewarding baseline performed poorly across all benchmarks for the smaller model (Llama-1B) and remained suboptimal for larger models (Mistral-7B), though the performance gap was less. 

Notably, for Llama-1B-SFT, we observe a performance drop compared to Llama-1B-Base. This can be attributed to the over-memorization of instructions during SFT (Zhang et al., 2025; Chu et al., 2025; Kirk et al., 2023) due to which the model repeats it’s responses (Hiraoka and Inui, 2024), which may have negatively impacted generalization. 

In contrast, for Mistral-7B, our method showed further improvement on Mistral-7B Instruct, which was chosen as the reference model for performing IPO. This suggests that self-improvement can enhance model performance beyond traditional instruction tuning. 

IPO exhibited significant improvements, performing on par with reward-model-based preference training, whose preferences are often considered the gold standard for preference optimization. While reward models showed a slight advantage in some benchmarks, our approach either matched or outperformed them in others. Moreover, we found that the impact of IPO was more pronounced in larger models (Mistral-7B) than in smaller models (Llama-1B). Our results suggests that LLMs are capable of self-alignment via judging and training on their own generations. 

## 5 Conclusion

We introduced IPO, a simple yet effective framework that utilizes likelihood-based preferences to optimize language models without requiring explicit reward models or expensive human annotations. Our analysis demonstrates that preference signals can be obtained directly from the likelihood of smaller base, instruction-tuned, and taskspecific LLMs, mitigating the need for prompting large-scale models such as GPT-4. 

Furthermore, we examined three settings for acquiring preferences over model-generated outputs namely self-rewarding LLMs, reward model-based preference classification, and preference classification using our framework for DPO. We show that models trained using preferences derived through our method align closely with, and in some cases surpass, models trained with preferences obtained from traditional reward models. These results highlight the efficacy of IPO as a scalable and costefficient alternative for preference optimization in large language models. 

## 6 Limitations

Our approach relies on the pre-categorization of the dataset. However, an alternative direction worth exploring is leveraging the model itself to generate category labels, which could enhance adaptability and reduce reliance on predefined classifications. We conducted our preference optimization experiments on only two model sizes—1B and 7B pa rameters—using a subset of 4,000 prompts from the UltraFeedback dataset. Due to computational constraints, we employed DPO rather than the iterative DPO approach used in the Self-Rewarding baseline. Additionally, all our evaluations were performed in a single run with a fixed random seed of 42, which may limit the robustness of our results. Unlike Self-Rewarding approaches that generate instructions using the model itself, our work relies on instructions sourced from an external dataset. This was due to the inability of smaller base models to produce high-quality instructions with simple prompting. Furthermore, we also do not test our hypothesis on LLMs where they are asked to pick the better of the two responses due to the high amount of positional bias present in them (Zheng et al., 2023a; Li et al., 2024c). 

## 7 Acknowledgement

We thank Modal Labs and E2E cloud networks for providing the computational resources and GPUs required for the project. 

## References



Marah Abdin, Jyoti Aneja, Hany Awadalla, Ahmed Awadallah, Ammar Ahmad Awan, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Jianmin Bao, Harkirat Behl, et al. 2024. Phi-3 technical report: A highly capable language model locally on your phone. arXiv preprint arXiv:2404.14219. 





Viraat Aryabumi, Yixuan Su, Raymond Ma, Adrien Morisot, Ivan Zhang, Acyr Locatelli, Marzieh Fadaee, Ahmet Üstün, and Sara Hooker. 2024. To code, or not to code? exploring impact of code in pre-training. arXiv preprint arXiv:2408.10914. 





Zhangir Azerbayev, Hailey Schoelkopf, Keiran Paster, Marco Dos Santos, Stephen McAleer, Albert Q. Jiang, Jia Deng, Stella Biderman, and Sean Welleck. 2024. Llemma: An open language model for mathematics. Preprint, arXiv:2310.10631. 





Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, et al. 2022. Constitutional ai: Harmlessness from ai feedback. arXiv preprint arXiv:2212.08073. 





Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, and Quanquan Gu. 2024. Self-play fine-tuning converts weak language models to strong language models. Preprint, arXiv:2401.01335. 





Wei-Lin Chiang, Lianmin Zheng, Ying Sheng, Anastasios Nikolas Angelopoulos, Tianle Li, Dacheng Li, Hao Zhang, Banghua Zhu, Michael Jordan, Joseph E. Gonzalez, and Ion Stoica. 2024. Chatbot arena: An open platform for evaluating llms by human preference. Preprint, arXiv:2403.04132. 





Paul Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, and Dario Amodei. 2023. Deep reinforcement learning from human preferences. Preprint, arXiv:1706.03741. 





Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie, Dale Schuurmans, Quoc V Le, Sergey Levine, and Yi Ma. 2025. Sft memorizes, rl generalizes: A comparative study of foundation model post-training. arXiv preprint arXiv:2501.17161. 





Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457. 





Mike Conover, Matt Hayes, Ankit Mathur, Jianwei Xie, Jun Wan, Sam Shah, Ali Ghodsi, Patrick Wendell, Matei Zaharia, and Reynold Xin. 2023. Free dolly: Introducing the world’s first truly open instructiontuned llm. 





Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Wei Zhu, Yuan Ni, Guotong Xie, Zhiyuan Liu, and Maosong Sun. 2023. Ultrafeedback: Boosting language models with high-quality feedback. Preprint, arXiv:2310.01377. 





Yijiang River Dong, Tiancheng Hu, and Nigel Collier. 2024. Can LLM be a personalized judge? In Findings ofthe Associationfor Computational Linguistics: EMNLP 2024, pages 10126–10141, Miami, Florida, USA. Association for Computational Linguistics. 





Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783. 





Yann Dubois, Balázs Galambosi, Percy Liang, and Tatsunori B Hashimoto. 2024a. Length-controlled alpacaeval: A simple way to debias automatic evaluators. arXiv preprint arXiv:2404.04475. 





Yann Dubois, Xuechen Li, Rohan Taori, Tianyi Zhang, Ishaan Gulrajani, Jimmy Ba, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2024b. Alpacafarm: A simulation framework for methods that learn from human feedback. Preprint, arXiv:2305.14387. 





Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela. 2024. Kto: Model alignment as prospect theoretic optimization. Preprint, arXiv:2402.01306. 





Angela Fan, Mike Lewis, and Yann Dauphin. 2018. Hierarchical neural story generation. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 889–898. 





Jiaxuan Gao, Shusheng Xu, Wenjie Ye, Weilin Liu, Chuyi He, Wei Fu, Zhiyu Mei, Guangju Wang, and Yi Wu. 2024. On designing effective rl reward at training time for llm reasoning. arXiv preprint arXiv:2410.15115. 





Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, and Soroosh Mariooryad. 2024. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. Preprint, arXiv:2403.05530. 





Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan, Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen, Shengjie Ma, Honghao Liu, Yuanzhuo Wang, and Jian Guo. 2024. A survey on llm-as-a-judge. Preprint, arXiv:2411.15594. 





Caglar Gulcehre, Tom Le Paine, Srivatsan Srinivasan, Ksenia Konyushkova, Lotte Weerts, Abhishek Sharma, Aditya Siddhant, Alex Ahern, Miaosen Wang, Chenjie Gu, Wolfgang Macherey, Arnaud Doucet, Orhan Firat, and Nando de Freitas. 2023. Reinforced self-training (rest) for language modeling. Preprint, arXiv:2308.08998. 





Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio César Teodoro Mendes, Allie Del Giorno, Sivakanth Gopi, Mojan Javaheripi, Piero Kauffmann, Gustavo de Rosa, Olli Saarikivi, et al. 2023. Textbooks are all you need. arXiv preprint arXiv:2306.11644. 





Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2020. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300. 





Tatsuya Hiraoka and Kentaro Inui. 2024. Repetition neurons: How do language models produce repetitions? arXiv preprint arXiv:2410.13497. 





Audrey Huang, Adam Block, Dylan J. Foster, Dhruv Rohatgi, Cyril Zhang, Max Simchowitz, Jordan T. Ash, and Akshay Krishnamurthy. 2024a. Selfimprovement in language models: The sharpening mechanism. Preprint, arXiv:2412.01951. 





Chenghua Huang, Zhizhen Fan, Lu Wang, Fangkai Yang, Pu Zhao, Zeqi Lin, Qingwei Lin, Dongmei Zhang, Saravan Rajmohan, and Qi Zhang. 2024b. Self-evolved reward learning for llms. arXiv preprint arXiv:2411.00418. 





Jiaxin Huang, Shixiang Shane Gu, Le Hou, Yuexin Wu, Xuezhi Wang, Hongkun Yu, and Jiawei Han. 2022. Large language models can self-improve. Preprint, arXiv:2210.11610. 





Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Lei Zhang, Tianyu Liu, Jiajun Zhang, Bowen Yu, Keming Lu, et al. 2024. Qwen2. 5-coder technical report. arXiv preprint arXiv:2409.12186. 





Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. 2023. Mistral 7b. arXiv preprint arXiv:2310.06825. 





Robert Kirk, Ishita Mediratta, Christoforos Nalmpantis, Jelena Luketina, Eric Hambro, Edward Grefenstette, and Roberta Raileanu. 2023. Understanding the effects of rlhf on llm generalisation and diversity. arXiv preprint arXiv:2310.06452. 





Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large language models are zero-shot reasoners. Advances in neural information processing systems, 35:22199– 22213. 





Nathan Lambert, Valentina Pyatkin, Jacob Morrison, LJ Miranda, Bill Yuchen Lin, Khyathi Chandu, Nouha Dziri, Sachin Kumar, Tom Zick, Yejin Choi, 





Noah A. Smith, and Hannaneh Hajishirzi. 2024. Rewardbench: Evaluating reward models for language modeling. Preprint, arXiv:2403.13787. 





Harrison Lee, Samrat Phatale, Hassan Mansoor, Kellie Ren Lu, Thomas Mesnard, Johan Ferret, Colton Bishop, Ethan Hall, Victor Carbune, and Abhinav Rastogi. 2023. Rlaif: Scaling reinforcement learning from human feedback with ai feedback. 





Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer. 2019. BART: denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. CoRR, abs/1910.13461. 





Junlong Li, Fan Zhou, Shichao Sun, Yikai Zhang, Hai Zhao, and Pengfei Liu. 2024a. Dissecting human and llm preferences. arXiv preprint arXiv:2402.11296. 





Xian Li, Ping Yu, Chunting Zhou, Timo Schick, Omer Levy, Luke Zettlemoyer, Jason E Weston, and Mike Lewis. 2024b. Self-alignment with instruction backtranslation. In The Twelfth International Conference on Learning Representations. 





Zongjie Li, Chaozheng Wang, Pingchuan Ma, Daoyuan Wu, Shuai Wang, Cuiyun Gao, and Yang Liu. 2024c. Split and merge: Aligning position biases in LLMbased evaluators. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, pages 11084–11108, Miami, Florida, USA. Association for Computational Linguistics. 





Zhiqiu Lin, Deepak Pathak, Baiqi Li, Jiayao Li, Xide Xia, Graham Neubig, Pengchuan Zhang, and Deva Ramanan. 2025. Evaluating text-to-visual generation with image-to-text generation. In European Confer ence on Computer Vision, pages 366–384. Springer. 





Chris Yuhao Liu, Liang Zeng, Jiacai Liu, Rui Yan, Jujie He, Chaojie Wang, Shuicheng Yan, Yang Liu, and Yahui Zhou. 2024a. Skywork-reward: Bag of tricks for reward modeling in llms. arXiv preprint arXiv:2410.18451. 





Fei Liu et al. 2020. Learning to summarize from human feedback. In Proceedings ofthe 58th Annual Meeting ofthe Associationfor Computational Linguistics. 





Yantao Liu, Zijun Yao, Rui Min, Yixin Cao, Lei Hou, and Juanzi Li. 2024b. Rm-bench: Benchmarking reward models of language models with subtlety and style. Preprint, arXiv:2410.16184. 





Anton Lozhkov, Raymond Li, Loubna Ben Allal, Fed erico Cassano, Joel Lamy-Poirier, Nouamane Tazi, Ao Tang, Dmytro Pykhtar, Jiawei Liu, Yuxiang Wei, et al. 2024. Starcoder 2 and the stack v2: The next generation. arXiv preprint arXiv:2402.19173. 





OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, and Shyamal Anadkat. 2024. Gpt-4 technical report. Preprint, arXiv:2303.08774. 





Myle Ott, Sergey Edunov, Alexei Baevski, Angela Fan, Sam Gross, Nathan Ng, David Grangier, and Michael Auli. 2019. fairseq: A fast, extensible toolkit for sequence modeling. In Proceedings ofNAACL-HLT 2019: Demonstrations. 





Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744. 





Jing-Cheng Pang, Pengyuan Wang, Kaiyuan Li, Xiong-Hui Chen, Jiacheng Xu, Zongzhang Zhang, and Yang Yu. 2023. Language model self-improvement by reinforcement learning contemplation. arXiv preprint arXiv:2305.14483. 





Jackson Petty, Sjoerd van Steenkiste, and Tal Linzen. 2024. How does code pretraining affect language model task performance? arXiv preprint arXiv:2409.04556. 





Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. 2024. Direct preference optimization: Your language model is secretly a reward model. Preprint, arXiv:2305.18290. 





Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. Preprint, arXiv:2402.03300. 





Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. 2024. Scaling llm test-time compute optimally can be more effective than scaling model parameters. Preprint, arXiv:2408.03314. 





Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V Le, Ed H Chi, Denny Zhou, et al. 2022. Challenging big-bench tasks and whether chain-of-thought can solve them. arXiv preprint arXiv:2210.09261. 





CodeGemma Team, Heri Zhao, Jeffrey Hui, Joshua Howland, Nam Nguyen, Siqi Zuo, Andrea Hu, Christopher A Choquette-Choo, Jingyue Shen, Joe Kelley, et al. 2024a. Codegemma: Open code models based on gemma. arXiv preprint arXiv:2406.11409. 





Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, Léonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ramé, et al. 2024b. Gemma 2: Improving open language models at a practical size. arXiv preprint arXiv:2408.00118. 





Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, and Shruti Bhosale. 2023. Llama 2: Open foundation and fine-tuned chat models. Preprint, arXiv:2307.09288. 





Tianhao Wu, Weizhe Yuan, Olga Golovneva, Jing Xu, Yuandong Tian, Jiantao Jiao, Jason Weston, and Sain bayar Sukhbaatar. 2024a. Meta-rewarding language models: Self-improving alignment with llm-as-ameta-judge. arXiv preprint arXiv:2407.19594. 





Yue Wu, Zhiqing Sun, Huizhuo Yuan, Kaixuan Ji, Yiming Yang, and Quanquan Gu. 2024b. Self-play preference optimization for language model alignment. arXiv preprint arXiv:2405.00675. 





An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. 2024. Qwen2. 5 tech nical report. arXiv preprint arXiv:2412.15115. 





Ziyi Ye, Xiangsheng Li, Qiuchi Li, Qingyao Ai, Yujia Zhou, Wei Shen, Dong Yan, and Yiqun Liu. 2024. Beyond scalar reward model: Learning generative judge from preference data. arXiv preprint arXiv:2410.03742. 





Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Xian Li, Sainbayar Sukhbaatar, Jing Xu, and Jason E Weston. 2024. Self-rewarding language models. In Forty-first International Conference on Machine Learning. 





Zheng Yuan, Hongyi Yuan, Chuanqi Tan, Wei Wang, Songfang Huang, and Fei Huang. 2023. Rrhf: Rank responses to align language models with human feedback without tears. Preprint, arXiv:2304.05302. 





Dan Zhang, Sining Zhoubian, Ziniu Hu, Yisong Yue, Yuxiao Dong, and Jie Tang. 2024. Rest-mcts*: Llm self-training via process reward guided tree search. arXiv preprint arXiv:2406.03816. 





Dylan Zhang, Qirun Dai, and Hao Peng. 2025. The best instruction-tuning data are those that fit. arXiv preprint arXiv:2502.04194. 





Yao Zhao, Rishabh Joshi, Tianqi Liu, Misha Khalman, Mohammad Saleh, and Peter J. Liu. 2023. Slic-hf: Sequence likelihood calibration with human feedback. Preprint, arXiv:2305.10425. 





Chujie Zheng, Hao Zhou, Fandong Meng, Jie Zhou, and Minlie Huang. 2023a. Large language models are not robust multiple choice selectors. In The Twelfth International Conference on Learning Representations. 





Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. 2023b. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in Neural Information Processing Systems, 36:46595–46623. 





Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. 2023. Instruction-following evaluation for large language models. arXiv preprint arXiv:2311.07911. 





Yongwei Zhou and Tiejun Zhao. 2024. Dual instruction tuning with large language models for mathematical reasoning. arXiv preprint arXiv:2403.18295. 





Daniel M. Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B. Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. 2020. Fine-tuning language models from human preferences. Preprint, arXiv:1909.08593. 



## A Implementation and Hardware Details

We conducted all training procedures using QLoRA with bfloat16 precision for DPO-based training and full fine-tuning for SFT. Our LLaMA-based models were trained on a single A100 GPU with 40GB VRAM, while Mistral training was performed on a single A100 GPU with 80GB VRAM. Inferences presented in Section 3 were carried out using T4 GPUs with float16 precision, whereas evaluation results in Section 4 were obtained using A10 GPUs with bfloat16 precision. For sampling responses on UltraFeedback for DPO , we used a temperature of 0.7 and a top_k value of 40. 

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Number of Training Epochs</td><td>3</td></tr><tr><td>Train Batch Size</td><td>4</td></tr><tr><td>Learning Rate</td><td><eq>5 \times 10^{-4}</eq></td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning Rate Scheduler</td><td>Cosine</td></tr></table>


Table 5: Training Hyperparameters for SFT Training


<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Number of Training Epochs</td><td>3</td></tr><tr><td>Train Batch Size</td><td>6</td></tr><tr><td>Learning Rate</td><td><eq>5 \times 10^{-4}</eq></td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning Rate Scheduler</td><td>Cosine</td></tr><tr><td>DPO Beta</td><td>0.1</td></tr><tr><td>LoRA Alpha</td><td>128</td></tr><tr><td>LoRA Dropout</td><td>0.05</td></tr><tr><td>LoRA Rank (r)</td><td>256</td></tr></table>


Table 6: Training Hyperparameters for DPO Training


## B Evaluation Dataset and Strategy

To conduct our evaluation, we randomly sample a subset of 500 examples from each of the datasets. 

<table><tr><td>Category</td><td>Self-Rewarding (%)</td><td>Ours (%)</td><td>Binary (%)</td></tr><tr><td>Chat</td><td>62.64</td><td>83.72</td><td>77.63</td></tr><tr><td>Safety</td><td>57.14</td><td>91.74</td><td>78.44</td></tr><tr><td>Code</td><td>62.80</td><td>95.32</td><td>94.10</td></tr><tr><td>Math</td><td>34.90</td><td>59.50</td><td>73.40</td></tr><tr><td>Average</td><td>54.37</td><td>82.57</td><td>80.89</td></tr></table>


Table 4: Comparison of accuracy between GPT-4o-Mini-Self, GPT-4o-Mini-Ours and GPT-Binary across different categories.


• IFEval (Instruction-Following Evaluation)<sup>2</sup>: Assesses the ability of large language models to follow explicit, verifiable instructions, such as “write in more than 400 words” or “mention the keyword ‘AI’ at least three times.” 

IFEval has four accuracy metrics to evaluate the instruction-following capabilities of Large Language Models (LLMs). Promptlevel strict-accuracy measures the percentage of prompts where all verifiable instructions are followed exactly, providing a strict evaluation of the model’s ability to handle complex prompts without errors. Instruction-level strict-accuracy evaluates the percentage of individual instructions followed precisely across all prompts, offering a granular view of the model’s performance on specific instruction types. Prompt-level loose-accuracy is a more lenient version of prompt-level strict-accuracy, where responses are transformed (e.g., removing markdown tags or intros/outros) to reduce false negatives, accounting for minor deviations. Similarly, Instruction-level looseaccuracy measures the percentage of individ ual instructions followed with leniency, using transformed responses to identify cases where the model almost adheres to instructions. The final metric is the average of all the four accuracies. Each category specific result of IFEval are shown in Table 7 

• MMLU (Massive Multitask Language Understanding)<sup>3</sup>: Evaluates models across 57 subjects using multiple-choice questions, covering disciplines such as humanities, STEM, and social sciences, to measure broad knowledge and reasoning capabilities. 

• BBH (BIG-Bench Hard)<sup>4</sup>: BigBench Hard dataset, focuses on complex problem-solving areas such as multistep arithmetic, algorithmic reasoning, and advanced language comprehension. 

• ARC-Easy (AI2 Reasoning Challenge - Easy)<sup>5</sup>: Comprises grade-school-level, multiple-choice science questions designed to assess fundamental reasoning and knowledge. 

• Alpaca-Eval<sup>6</sup>: A benchmark that compares model-generated responses against given responses, employing GPT as an evaluator to determine output quality. 

For the evaluation of MMLU, BBH, and ARC-Easy, we utilize GPT-4o-mini to compare modelgenerated responses with ground-truth answers. For IFEval, we employ the official evaluation code. Similarly, for Alpaca-Eval, we use GPT-4o-mini to compare the model-generated response against the ground-truth response from text_davinci_003 and determine the better output. All our sampling for the evaluations was performed using a temperature of 0.5 and top_k value of 40. 

## C Results on GPT

We also evaluated our approach on proprietary models like GPT-4o-Mini and found that it significantly outperformed both the Self-Rewarding approach and the Binary Approach. In the Binary Approach, the model is given both the chosen and rejected responses along with the prompt and is asked to select the better one. To mitigate positional bias—where LLMs tend to favor the first response—a random shuffle is applied to ensure that neither the chosen nor the rejected response receivs a systematic advantage. The results for Binary Eval were taken directely from Reward Bench<sup>7</sup>. The results for the same are shown in Table 4. 

<table><tr><td rowspan="2"></td><td colspan="2">Strict</td><td colspan="2">Loose</td><td rowspan="2">Average</td></tr><tr><td>Prompt-Level</td><td>Instruction-Level</td><td>Prompt-Level</td><td>Instruction-Level</td></tr><tr><td>Mistral Self</td><td>20.52%</td><td>33.09%</td><td>25.88%</td><td>37.77%</td><td>29.31%</td></tr><tr><td>Mistral Ours</td><td>31.79%</td><td>41.85%</td><td>36.60%</td><td>46.52%</td><td>39.19%</td></tr><tr><td>Mistral Reward</td><td>23.11%</td><td>34.05%</td><td>28.84%</td><td>40.77%</td><td>31.69%</td></tr><tr><td>Mistral Base</td><td>21.26%</td><td>30.10%</td><td>23.29%</td><td>31.89%</td><td>26.63%</td></tr><tr><td>Mistral Instruct</td><td>33.27%</td><td>43.17%</td><td>36.41%</td><td>47.36%</td><td>40.05%</td></tr><tr><td>Llama Base</td><td>5.95%</td><td>11.35%</td><td>8.17%</td><td>13.75%</td><td>9.80%</td></tr><tr><td>Llama SFT</td><td>5.02%</td><td>12.63%</td><td>7.24%</td><td>15.87%</td><td>10.19%</td></tr><tr><td>Llama Self</td><td>6.28%</td><td>13.19%</td><td>8.32%</td><td>17.15%</td><td>11.23%</td></tr><tr><td>Llama Ours</td><td>6.47%</td><td>14.27%</td><td>9.61%</td><td>17.99%</td><td>12.08%</td></tr><tr><td>Llama Reward</td><td>6.47%</td><td>12.71%</td><td>7.95%</td><td>15.59%</td><td>10.68%</td></tr></table>


Table 7: Performance comparison of different models under strict and loose conditions.


## D Prompts

Based on the predefined categories, a pool of N prompts were generated using GPT. A small sample of 50 data points was selected from each category in Reward Bench, and the prompts were tested on this subset to determine the most effective one. The model used for testing was LLaMA-1B-Instruct, and the prompt with the highest accuracy in each category was used for evaluation on Reward Bench. The final set of selected prompts for Reward Bench evaluations are shown in Table 8 and those for DPO are shown in 9 . 

Note: For DPO-based training, instructions were classified into four categories 

Categories of Prompts: 

• Code: Questions that involve generating or interpreting code. 

• Math: Questions related to mathematical computations or reasoning. 

• Chat: General conversational questions, often open-ended or casual. 

• Safety: Questions focused on evaluating or ensuring safety in responses. For our evaluation on the Reward Bench dataset we further subcategorize it into Safety General and Safety Refusal. 

## E Additional Results

To further demonstrate the effectiveness of our approach, we also evaluate our approach on an additional benchmark, RM-Bench results of which are shown in Table 11 

<table><tr><td>Task</td><td>Prompt</td></tr><tr><td>Chat</td><td>Below is a question and its response. Assess if the response is helpful and correct. Answer just Yes/No.</td></tr><tr><td>Code</td><td>Below is a coding-related question along with a response containing code. Your task is to verify whether the provided code is correct, functional, and aligned with the requirements of the question. Answer with just Yes/No.</td></tr><tr><td>Maths</td><td>Below is a math question along with a response containing a solution. Your task is to verify whether the solution is correct, the reasoning is logical, and the final answer is accurate. Answer with just Yes/No.</td></tr><tr><td>Safety(General)</td><td>Below is a potentially unsafe or inappropriate question, along with a refusal response. Verify whether the refusal response is appropriate. Answer Yes/No.</td></tr><tr><td>Safety(Refusal)</td><td>Below is a question and a refusal response. Ensure the refusal response aligns with ethical and safety guidelines while maintaining professional clarity. Answer Yes/No.</td></tr></table>


Table 8: Final prompts used for Evaluation on Reward and RM Bench


<table><tr><td>Category</td><td>Prompt</td></tr><tr><td>Code</td><td>Below is a coding-related question along with a response containing code. Your task is to verify whether the provided code is correct, functional, and aligned with the requirements of the question. Answer with just Yes/No.</td></tr><tr><td>Chat</td><td>Below is a question, along with a response to that question. Your task is to verify whether the response is helpful, factually correct, logically sound, and contextually appropriate. Answer with just Yes/No.</td></tr><tr><td>Safety</td><td>Inspect the response to ensure it is safe and appropriate for the question below. Answer Yes/No.</td></tr><tr><td>Math</td><td>Below is a math question along with a response containing a solution. Your task is to verify whether the solution is correct, the reasoning is logical, and the final answer is accurate. Answer with just Yes/No.</td></tr></table>


Table 9: Final set of prompts used for DPO.


<table><tr><td>Dataset</td><td>Llama3.2-1B</td><td>Llama3.2-1BInstruct</td><td>Llama3.2-3B</td><td>Llama3.2-3BInstruct</td><td>MetaLlama-3-8B</td><td>MetaLlama-3-8BInstruct</td><td>Mistral7B-v0.1</td><td>Mistral7BInstruct-v0.1</td><td>Qwen2.5-3B</td><td>Qwen2.5-3BInstruct</td><td>Qwen2.5-7B</td><td>Qwen2.5-7BInstruct</td><td>SKYWORK8breward</td></tr><tr><td>hep-cpp</td><td>54.88</td><td>49.39</td><td>68.29</td><td>65.24</td><td>57.32</td><td>74.39</td><td>70.12</td><td>75.00</td><td>82.93</td><td>76.22</td><td>84.76</td><td>78.05</td><td>92.68</td></tr><tr><td>math-prm</td><td>23.49</td><td>88.14</td><td>98.21</td><td>98.21</td><td>77.18</td><td>54.97</td><td>97.99</td><td>96.20</td><td>24.61</td><td>46.31</td><td>68.46</td><td>56.24</td><td>95.75</td></tr><tr><td>llmbar-adver-GPTInst</td><td>63.04</td><td>64.13</td><td>44.57</td><td>53.26</td><td>59.78</td><td>71.74</td><td>71.74</td><td>75.00</td><td>51.09</td><td>83.70</td><td>59.78</td><td>78.26</td><td>71.74</td></tr><tr><td>refusals-dangerous</td><td>76.00</td><td>94.00</td><td>22.00</td><td>72.00</td><td>25.00</td><td>91.00</td><td>45.00</td><td>86.00</td><td>72.00</td><td>78.00</td><td>74.00</td><td>96.00</td><td>92.00</td></tr><tr><td>hep-python</td><td>50.61</td><td>52.44</td><td>61.59</td><td>71.34</td><td>53.66</td><td>77.44</td><td>67.07</td><td>76.22</td><td>77.44</td><td>78.66</td><td>89.02</td><td>89.02</td><td>93.29</td></tr><tr><td>alpacaeval-easy</td><td>34.41</td><td>83.23</td><td>34.66</td><td>53.79</td><td>56.15</td><td>24.22</td><td>20.00</td><td>42.36</td><td>36.40</td><td>27.33</td><td>46.71</td><td>80.25</td><td>92.92</td></tr><tr><td>hep-java</td><td>54.88</td><td>55.49</td><td>58.54</td><td>67.68</td><td>49.39</td><td>78.05</td><td>74.39</td><td>68.29</td><td>85.37</td><td>86.59</td><td>88.41</td><td>84.15</td><td>92.68</td></tr><tr><td>llmbar-adver-GPTOut</td><td>55.32</td><td>46.81</td><td>36.17</td><td>44.68</td><td>29.79</td><td>44.68</td><td>46.81</td><td>53.19</td><td>53.19</td><td>48.94</td><td>53.19</td><td>59.57</td><td>68.09</td></tr><tr><td>alpacaeval-hard</td><td>49.69</td><td>88.20</td><td>55.16</td><td>70.43</td><td>50.81</td><td>40.62</td><td>27.70</td><td>63.23</td><td>38.63</td><td>33.66</td><td>50.06</td><td>88.45</td><td>84.60</td></tr><tr><td>hep-go</td><td>49.39</td><td>45.73</td><td>53.66</td><td>64.63</td><td>57.93</td><td>73.78</td><td>70.73</td><td>73.78</td><td>81.10</td><td>82.93</td><td>83.54</td><td>85.98</td><td>90.24</td></tr><tr><td>refusals-offensive</td><td>73.00</td><td>97.00</td><td>49.00</td><td>97.00</td><td>86.00</td><td>99.00</td><td>45.00</td><td>97.00</td><td>23.00</td><td>94.00</td><td>98.00</td><td>100.00</td><td>98.00</td></tr><tr><td>xstest-should-refuse</td><td>56.49</td><td>77.92</td><td>67.53</td><td>92.21</td><td>82.47</td><td>98.70</td><td>46.75</td><td>92.21</td><td>54.55</td><td>93.51</td><td>84.42</td><td>94.16</td><td>77.27</td></tr><tr><td>donotanswer</td><td>38.24</td><td>55.88</td><td>64.71</td><td>82.35</td><td>69.12</td><td>91.91</td><td>63.97</td><td>80.88</td><td>62.50</td><td>91.18</td><td>78.68</td><td>90.44</td><td>70.59</td></tr><tr><td>mt-bench-hard</td><td>51.11</td><td>60.00</td><td>64.44</td><td>64.44</td><td>48.89</td><td>48.89</td><td>48.89</td><td>53.33</td><td>55.56</td><td>55.56</td><td>64.44</td><td>66.67</td><td>71.11</td></tr><tr><td>llmbar-adver-neighbor</td><td>64.18</td><td>58.96</td><td>50.00</td><td>60.45</td><td>59.70</td><td>74.63</td><td>45.52</td><td>59.70</td><td>44.03</td><td>72.39</td><td>60.45</td><td>73.88</td><td>75.37</td></tr><tr><td>mt-bench-easy</td><td>60.71</td><td>60.71</td><td>60.71</td><td>78.57</td><td>50.00</td><td>89.29</td><td>60.71</td><td>75.00</td><td>60.71</td><td>78.57</td><td>89.29</td><td>100.00</td><td>100.00</td></tr><tr><td>llmbar-adver-manual</td><td>65.22</td><td>52.17</td><td>52.17</td><td>47.83</td><td>47.83</td><td>63.04</td><td>45.65</td><td>52.17</td><td>41.30</td><td>56.52</td><td>45.65</td><td>60.87</td><td>63.04</td></tr><tr><td>mt-bench-med</td><td>37.78</td><td>64.44</td><td>64.44</td><td>71.11</td><td>51.11</td><td>60.00</td><td>57.78</td><td>60.00</td><td>55.56</td><td>73.33</td><td>71.11</td><td>93.33</td><td>86.67</td></tr><tr><td>xstest-should-respond</td><td>53.60</td><td>77.60</td><td>57.60</td><td>57.60</td><td>50.00</td><td>58.80</td><td>72.40</td><td>63.20</td><td>84.40</td><td>73.60</td><td>90.40</td><td>85.60</td><td>86.40</td></tr><tr><td>hep-rust</td><td>48.78</td><td>53.05</td><td>64.02</td><td>65.85</td><td>51.22</td><td>68.29</td><td>63.41</td><td>62.80</td><td>79.27</td><td>74.39</td><td>85.98</td><td>74.39</td><td>90.24</td></tr><tr><td>hep-js</td><td>44.51</td><td>60.98</td><td>63.41</td><td>68.29</td><td>56.71</td><td>71.34</td><td>66.46</td><td>69.51</td><td>81.71</td><td>84.76</td><td>82.93</td><td>87.20</td><td>93.29</td></tr><tr><td>alpacaeval-length</td><td>62.61</td><td>70.43</td><td>69.94</td><td>69.44</td><td>69.94</td><td>62.11</td><td>63.11</td><td>68.82</td><td>53.42</td><td>66.83</td><td>54.41</td><td>71.68</td><td>86.71</td></tr><tr><td>llmbar-natural</td><td>58.00</td><td>59.00</td><td>54.00</td><td>69.00</td><td>64.00</td><td>76.00</td><td>51.00</td><td>71.00</td><td>62.00</td><td>73.00</td><td>74.00</td><td>88.00</td><td>82.00</td></tr></table>


Table 10: Reward Bench Performance Across Different Levels


<table><tr><td>Model</td><td>Levels</td><td>RM-Bench chat</td><td>RM-Bench code</td><td>RM-Bench math</td><td>RM-Bench safety response</td><td>RM-Bench safety refuse</td></tr><tr><td rowspan="4">Llama-1B</td><td>Level 1</td><td>48.06</td><td>54.39</td><td>46.31</td><td>31.85</td><td>38.73</td></tr><tr><td>Level 2</td><td>64.34</td><td>55.26</td><td>48.58</td><td>69.43</td><td>53.52</td></tr><tr><td>Level 3</td><td>60.47</td><td>50.44</td><td>41.59</td><td>61.78</td><td>71.13</td></tr><tr><td>Mean</td><td>57.62</td><td>53.36</td><td>45.49</td><td>54.35</td><td>54.46</td></tr><tr><td rowspan="4">Llama-1B-Instruct</td><td>Level 1</td><td>51.16</td><td>51.32</td><td>49.53</td><td>71.34</td><td>67.61</td></tr><tr><td>Level 2</td><td>61.24</td><td>53.51</td><td>47.45</td><td>68.15</td><td>77.11</td></tr><tr><td>Level 3</td><td>60.47</td><td>49.56</td><td>45.75</td><td>73.89</td><td>63.38</td></tr><tr><td>Mean</td><td>57.62</td><td>51.46</td><td>47.57</td><td>71.13</td><td>69.37</td></tr><tr><td rowspan="4">Llama-3B</td><td>Level 1</td><td>54.26</td><td>51.75</td><td>47.26</td><td>68.15</td><td>7.04</td></tr><tr><td>Level 2</td><td>33.33</td><td>52.19</td><td>46.12</td><td>78.34</td><td>37.32</td></tr><tr><td>Level 3</td><td>33.33</td><td>49.12</td><td>45.75</td><td>36.94</td><td>55.28</td></tr><tr><td>Mean</td><td>40.31</td><td>51.02</td><td>46.38</td><td>61.15</td><td>33.22</td></tr><tr><td rowspan="4">Llama3b-Instruct</td><td>Level 1</td><td>56.59</td><td>50.88</td><td>50.09</td><td>87.90</td><td>55.28</td></tr><tr><td>Level 2</td><td>44.96</td><td>55.26</td><td>48.02</td><td>86.62</td><td>60.56</td></tr><tr><td>Level 3</td><td>52.71</td><td>49.56</td><td>47.45</td><td>94.27</td><td>76.76</td></tr><tr><td>Mean</td><td>51.42</td><td>51.90</td><td>48.52</td><td>89.60</td><td>64.20</td></tr><tr><td rowspan="4">Llama-8B</td><td>Level 1</td><td>54.26</td><td>53.51</td><td>48.02</td><td>99.36</td><td>2.46</td></tr><tr><td>Level 2</td><td>56.59</td><td>56.58</td><td>51.98</td><td>83.44</td><td>29.58</td></tr><tr><td>Level 3</td><td>50.39</td><td>51.75</td><td>47.26</td><td>64.33</td><td>63.38</td></tr><tr><td>Mean</td><td>53.75</td><td>53.95</td><td>49.09</td><td>82.38</td><td>31.81</td></tr><tr><td rowspan="4">Llama-8B-Instruct</td><td>Level 1</td><td>65.12</td><td>55.70</td><td>50.28</td><td>56.05</td><td>75.00</td></tr><tr><td>Level 2</td><td>36.43</td><td>55.70</td><td>49.72</td><td>96.18</td><td>30.28</td></tr><tr><td>Level 3</td><td>50.39</td><td>53.51</td><td>46.12</td><td>64.33</td><td>87.68</td></tr><tr><td>Mean</td><td>50.65</td><td>54.97</td><td>48.71</td><td>72.19</td><td>64.32</td></tr><tr><td rowspan="4">Mistral-7b</td><td>Level 1</td><td>50.39</td><td>46.49</td><td>52.17</td><td>96.18</td><td>20.42</td></tr><tr><td>Level 2</td><td>61.24</td><td>53.51</td><td>49.34</td><td>44.59</td><td>89.44</td></tr><tr><td>Level 3</td><td>51.94</td><td>46.49</td><td>43.10</td><td>75.16</td><td>84.15</td></tr><tr><td>Mean</td><td>54.52</td><td>48.83</td><td>48.20</td><td>71.97</td><td>64.67</td></tr><tr><td rowspan="4">Mistral-7b-Instruct</td><td>Level 1</td><td>44.19</td><td>50.88</td><td>52.55</td><td>61.78</td><td>96.48</td></tr><tr><td>Level 2</td><td>58.91</td><td>52.63</td><td>55.39</td><td>39.49</td><td>81.69</td></tr><tr><td>Level 3</td><td>58.91</td><td>53.95</td><td>48.20</td><td>52.87</td><td>96.83</td></tr><tr><td>Mean</td><td>54.01</td><td>52.49</td><td>52.05</td><td>51.38</td><td>91.67</td></tr><tr><td rowspan="4">Qwen2.5-3B</td><td>Level 1</td><td>65.89</td><td>48.68</td><td>54.06</td><td>95.54</td><td>94.01</td></tr><tr><td>Level 2</td><td>58.14</td><td>52.19</td><td>51.23</td><td>82.80</td><td>88.03</td></tr><tr><td>Level 3</td><td>48.84</td><td>50.44</td><td>46.12</td><td>94.90</td><td>49.65</td></tr><tr><td>Mean</td><td>57.62</td><td>50.44</td><td>50.47</td><td>91.08</td><td>77.23</td></tr><tr><td rowspan="4">Qwen2.5-3B-Instruct</td><td>Level 1</td><td>72.87</td><td>51.32</td><td>60.87</td><td>46.50</td><td>63.38</td></tr><tr><td>Level 2</td><td>55.04</td><td>53.07</td><td>57.66</td><td>31.85</td><td>90.49</td></tr><tr><td>Level 3</td><td>55.04</td><td>54.82</td><td>50.47</td><td>84.71</td><td>96.83</td></tr><tr><td>Mean</td><td>60.98</td><td>53.07</td><td>56.33</td><td>54.35</td><td>83.57</td></tr><tr><td rowspan="4">Qwen2.5-7B</td><td>Level 1</td><td>72.87</td><td>56.58</td><td>56.14</td><td>100.00</td><td>100.00</td></tr><tr><td>Level 2</td><td>47.29</td><td>56.58</td><td>54.06</td><td>96.82</td><td>94.72</td></tr><tr><td>Level 3</td><td>51.16</td><td>53.07</td><td>47.64</td><td>94.27</td><td>100.00</td></tr><tr><td>Mean</td><td>57.11</td><td>55.41</td><td>52.61</td><td>97.03</td><td>98.24</td></tr><tr><td rowspan="4">Qwen2.5-7B-Inst</td><td>Level 1</td><td>80.62</td><td>58.33</td><td>62.19</td><td>91.08</td><td>100.00</td></tr><tr><td>Level 2</td><td>61.24</td><td>58.33</td><td>62.00</td><td>85.99</td><td>96.83</td></tr><tr><td>Level 3</td><td>64.34</td><td>55.26</td><td>50.28</td><td>63.06</td><td>100.00</td></tr><tr><td>Mean</td><td>68.73</td><td>57.31</td><td>58.16</td><td>80.04</td><td>98.94</td></tr><tr><td rowspan="4">SKYWORK-8b-reward</td><td>Level 1</td><td>86.04</td><td>53.07</td><td>62.38</td><td>94.90</td><td>97.18</td></tr><tr><td>Level 2</td><td>55.04</td><td>53.51</td><td>65.41</td><td>82.80</td><td>98.94</td></tr><tr><td>Level 3</td><td>41.09</td><td>48.25</td><td>66.16</td><td>87.26</td><td>100.00</td></tr><tr><td>Mean</td><td>60.72</td><td>51.61</td><td>64.65</td><td>88.32</td><td>98.60</td></tr></table>


Table 11: Performance of various models, across different Levels on RM-Bench
