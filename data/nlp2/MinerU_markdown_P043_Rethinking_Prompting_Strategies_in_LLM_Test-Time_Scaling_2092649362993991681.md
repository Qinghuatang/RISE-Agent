# Rethinking the Role of Prompting Strategies in LLM Test-Time Scaling: A Perspective of Probability Theory

Yexiang Liu<sup>1,2</sup>, Zekun Li<sup>3</sup>, Zhi Fang<sup>1,2</sup>, Nan Xu<sup>1,4</sup>, Ran He<sup>1,2</sup>*, Tieniu Tan<sup>1,2,5</sup> 

<sup>1</sup>MAIS, Institute of Automation, Chinese Academy of Sciences 

<sup>2</sup>School of Artificial Intelligence, University of Chinese Academy of Sciences 

<sup>3</sup>University of California, Santa Barbara 

<sup>4</sup>Beijing Wenge Technology Co., Ltd <sup>5</sup>Nanjing University {liuyexiang2023, fangzhi2023, xunan2015, ran.he, tieniu.tan}@ia.ac.cn zekunli@cs.ucsb.edu 

## Abstract

Recently, scaling test-time compute on Large Language Models (LLM) has garnered wide attention. However, there has been limited investigation of how various reasoning prompting strategies perform as scaling. In this paper, we focus on a standard and realistic scaling setting: majority voting. We systematically conduct experiments on 6 LLMs 8 prompting strategies 6 benchmarks. Experiment results consistently show that as the sampling time and computational overhead increase, complicated prompting strategies with superior initial performance gradually fall behind simple Chain-of-Thought. We analyze this phenomenon and provide theoretical proofs. Ad ditionally, we propose a probabilistic method to efficiently predict scaling performance and identify the best prompting strategy under large sampling times, eliminating the need for resource-intensive inference processes in practical applications. Furthermore, we introduce two ways derived from our theoretical analysis to significantly improve the scaling performance. We hope that our research can promote to re-examine the role of complicated prompting, unleash the potential of simple prompting strategies, and provide new insights for enhancing test-time scaling performance. Code is available at https://github.com/MraDonk ey/rethinking_prompting. 

## 1 Introduction

Over the past few years, how to enhance the reasoning abilities of large language models (LLMs) has been a topic of widespread interest (Dubey et al., 2024; Anil et al., 2023; Touvron et al., 2023; Open AI, 2024a; Team et al., 2024). Researchers have introduced various prompting strategies to improve the reasoning capacity of LLMs, such as Chain of Thought (CoT) (Wei et al., 2022) and so on (Zheng et al., 2024; Yasunaga et al., 2024; Madaan et al., 

2023). Recently, many studies have shown that scaling LLM test-time compute can also effectively improve reasoning (Snell et al., 2025; Open AI, 2024b; Ji et al., 2025; Bi et al., 2024). 

However, how different prompting strategies behave when scaling test-time compute is less explored. In this paper, we focus on a standard and effective scaling setting: majority voting.We comprehensively evaluate the performance of 8 mainstream prompting strategies under equivalent sampling time or computation overhead. We test 4 open-sourced and 2 closed-sourced LLMs on 6 reasoning benchmarks, finding that simple CoT consistently performs best on all LLMs across benchmarks with given budgets as scaling increases, even if it falls behind at the beginning. This indicates that current LLMs can achieve remarkable reasoning capabilities by only relying on simple CoT without other complicated prompting strategies. It also reminds us to reflect on the necessity of complicated prompting for scaling and fairly compare different strategies under the same budget. 

We systematically analyze this phenomenon and provide theoretical and experimental proofs. We conclude that this is caused by two reasons. One is that there are more easy questions and fewer hard questions for CoT compared to other strategies. Easy questions are more likely to get right solutions, and the error possibility decreases until 0% as scaling. In comparison, hard questions are the opposite. The other is that CoT is less likely to be affected by wrong answers. Although CoT sometimes has lower pass@1 accuracy, its probability of obtaining the correct answer is more prominent in the result distribution. In contrast, other strategies have higher disturbed peaks in the distribution of incorrect answers. These two reasons enable CoT to improve reasoning performance more rapidly and gradually dominate as scaling. 

What’s more, we propose a method with the complexity O(1) according to probability theory to quickly predict the scaling performance, which can serve as the test-time scaling law for major ity voting. Experiments show that our method can accurately estimate the scaling performance and select the best strategy with arbitrary sampling time. 

Furthermore, we explore two ways to significantly improve scaling performance with our theories. (1) Adaptively scaling according to the question difficulty. (2) Dynamically selecting the optimal prompting strategy. Extensive experiments verify their general effectiveness and superiority, e.g., improving Majority@10 accuracy from 86.0% to 97.4% and 15.2% to 61.0% for LLaMA-3-8B-Instruct (Dubey et al., 2024) on GSM8K (Cobbe et al., 2021) and MATH-500 (Hendrycks et al., 2021b) by combining (1) and (2), respectively. 

Our contributions can be summarized as follows: 

• We comprehensively study the test-time scaling performance on 6 LLMs 8 prompting strategies 6 benchmarks. (Section 2) 

• We find that CoT consistently performs best under the equivalent sampling time and computation overhead. (Section 3) 

• We analyze this phenomenon and provide theoretical and experimental proofs. (Section 4) 

• We propose a method to quickly predict the scaling performance and the best strategy under given sampling times. (Section 5) 

• Based on the above analysis, we introduce two ways to significantly improve the scaling performance. (Section 6) 

## 2 Scaling System Designs

We focus on a straight and effective setting of testtime scaling, majority voting, i.e., Self-Consistency (Wang et al., 2023b), which selects the most consistent answer among several samples. Our goal is to study what prompting strategy performs best under the equivalent scaling overhead, particularly when largely increasing the scaling extent. 

## 2.1 Models

We conduct experiments on 4 open-sourced LLMs including Qwen2.5-7B-Instruct (Yang et al., 2024a), LLaMA-3-8B-Instruct (Dubey et al., 2024), GLM-4-9B-Chat (GLM et al., 2024) and Phi-3.5- mini-Instruct, and 2 closed-sourced LLMs including Gemini-1.5-Flash (Team et al., 2024) and GPT-4o-mini (Open AI, 2024a). 

## 2.2 Prompting Strategies

We mainly focus on generalizable reasoning prompting strategies, excluding those individually designed for specific tasks or involving fine-tuning, training auxiliary models, or incorporating other models, tools, or human assistance. In this setting, the model’s performance is only related to the input prompt, thus making it fairly compare the scaling performance of those prompting strategies. The prompting strategies we test are listed as follows. 

Direct Prompting (DiP): Directly input the question to the model, without any additional instruction or restrictions to the output. 

Chain-of-Thought (CoT) (Wei et al., 2022; Kojima et al., 2022): Use the prompt “Let’s think step by step.” to solve the problem step by step. 

Least-to-Most (L2M) (Zhou et al., 2023): Break down the question into progressive subquestions. Answer the sub-questions and get the final result according to them and their answers. 

Tree-of-Thoughts (ToT) (Yao et al., 2023): Explore multiple reasoning paths to get several solutions, then analyze each solution and decide which one is the most promising. 

Self-Refine (S-RF) (Madaan et al., 2023): First, answer the question to get an initial answer. Next, evaluate the previous answer and get feedback. Finally, refine the previous answer according to feedback. This will last for several rounds. 

Step-Back Prompting (SBP) (Zheng et al., 2024): First, extract the discipline concepts and principles involved in solving the problem. Then, solve the problem step by step by following the principles. 

Analogous Prompting (AnP) (Yasunaga et al., 2024): Recall relevant problems as examples. Afterward, solve the analogous problems and proceed to solve the initial problem according to them. 

Multi-Agent Debate (MAD) (Du et al., 2024): Set three model instances as different agents to debate for several rounds, and select the most consistent result among them. 

## 2.3 Benchmarks

We evaluate across 6 reasoning benchmarks used in the original papers of the above prompting strategies, including GSM8K (Cobbe et al., 2021), GSM-Hard (Gao et al., 2023), MATH-500 (Hendrycks et al., 2021b; Lightman et al., 2024), MMLU-highschool-biology, chemistry and physics (Hendrycks et al., 2021a). 

## 2.4 Formal Expression

We divided the prompting strategies into two groups: iterative methods (S-RF, MAD, and ToT) and the other non-iterative methods. For S-RF and MAD, we run them N rounds and get the final result in the Nth round. For ToT, we set the model to explore and evaluate N different reasoning paths to get the best one. For others, we parallel sample N generations and get their most consistent answer with majority voting. For convenience, we refer to all of the above processes as sampling N times. Therefore, we can categorize those iterative strategies that require multiple rounds or reasoning paths as $\mathcal { P } _ { 2 } =$ S-RF, MAD, ToT , and other non-iterative ones as $\mathcal { P } _ { 1 } =$ DiP, CoT, L2M, SBP, AnP . 

Formally, assuming that we have n prompting strategies $\{ \mathbf { P } _ { i } | i = 1 , 2 , . . . , n \}$ , when using the prompt strategy $\mathbf { P } _ { i }$ to answer a text question x on a model , we can get the answered result of one sample with an answer extractor $\phi ,$ which extracts the answer in the output sentence using regular expressions. Then we can formalize the process of getting the final answer when sampling N times as $\phi [ \mathcal { M } ( x | \mathbf { P } _ { i } ) ; N ]$ 

$$
\left\{ \begin{array}{c} \operatorname{mode} \{\phi [ \mathcal {M} (x \mid \mathbf {P} _ {i}) ] \} _ {1} ^ {N}, \mathbf {P} _ {i} \in \mathcal {P} _ {1} \\ \phi [ \mathcal {M} (x \mid \mathbf {P} _ {i}; N) ], \quad \mathbf {P} _ {i} \in \mathcal {P} _ {2} \end{array} \right\}\tag{1}
$$

With a fixed sampling time N, the best prompting strategy $\mathbf { P } _ { N } ^ { * }$ on the dataset D is 

$$
\mathbf {P} _ {N} ^ {*} = \underset {\mathbf {P} _ {i}} {\operatorname{argmax}}   \mathbb {E} _ {x \in \mathfrak {D}}   \mathbb {1} \{\phi [ \mathcal {M} (x   |   \mathbf {P} _ {i}); N ] = y \},\tag{2}
$$

where y is the ground truth answer for $x .$ . However, sampling with distinct $\mathbf { P } _ { i }$ may cause different computation overhead. It would be fairer to compare them with a fixed overhead O. To calculate the overhead of using a model to answer a question x by sampling N times with the prompting strategy $\mathbf { P } _ { i }$ , we can consider it as a function of $x , \mathcal { M } , \mathbf { P } _ { i } , N$ , noted as $\mathscr { C } ( x | \mathscr { M } ; \mathbf { P } _ { i } ; N )$ . Under a fixed overhead O, the best prompting strategy $\mathbf { P } _ { O } ^ { * }$ on the dataset D is 

$$
\mathbf {P} _ {O} ^ {*} = \underset {\mathbf {P} _ {i}} {\operatorname{argmax}} \underset {N} {\max} \mathbb {E} _ {x \in \mathfrak {D}} \mathbb {1} \{\phi [ \mathcal {M} (x | \mathbf {P} _ {i}); N ] = y \}
$$

$$
s. t. \sum_ {x \in \mathfrak {D}} \mathcal {C} (x \mid \mathcal {M}; \mathbf {P} _ {i}; N) \leq O.\tag{3}
$$

Given that completion tokens are more computationally expensive than prompt tokens, we define the overhead as the weighted sum of prompt tokens and completion tokens (Cost). For the models Gemini-1.5-Flash and GPT-4o-mini, we utilize their respective pricing metrics.<sup>1</sup> For other opensourced models, we adopt the pricing of GPT-4omini as a proxy. 

## 3 CoT Dominates as Test-Time Scaling

Under each sampling time $N ,$ we test five times to obtain the average performance of majority voting. We evaluate under two kinds of budget constraints: (1) a fixed sampling time budget N, and (2) a fixed inference cost budget O. Figure 1 and 2 summarize the average performances across benchmarks of different $\mathbf { P } _ { i }$ under constrained sampling time $N$ and cost O on each model, and display the best prompting strategy $\mathbf { P } _ { N } ^ { * }$ under different values of N and $\mathbf { P } _ { O } ^ { * }$ under different values of $O ,$ respectively.<sup>2</sup> We can see that when scaling test-time compute, CoT performs best among all prompting strategies under a constrained N and O most of the time. Although some complicated prompting strategies perform best under lower N and O, CoT dominates without exception on all models when largely scaling. We theoretically and experimentally analyze this phenomenon, whose reasons come from two aspects. We explain these in detail in Section 4. 

What’s more, we find that about 80% of the results conform to this trend on each model and each benchmark. On certain datasets and LLMs, DiP also performs best as largely scaling. This is particularly evident on powerful models, such as Gemini-1.5-Flash and GPT-4o-mini. More detailed results can be found in Appendix C. These indicate that simple CoT is more efficient and has the potential to surpass other complicated prompting strategies under the same scaling setting. Current LLMs can achieve remarkable reasoning capabilities by only relying on simple prompting strategies. Complicated prompting with superior pass@1 accuracy may not always be better as test-time scaling. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/935649009d2a9c09f54f59ebf73e6529f487bc59d1ab0cd86431a24356b6b90a.jpg)



Figure 1: Average performances of distinct prompting strategies and the best one $\mathbf { P } _ { N } ^ { * }$ across benchmarks on each LLM under constrained sampling time N. As increasing the sample time N, the accuracy of CoT grows rapidly and it dominates on all models when N is large enough.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/aa018bcad699b329d98675397f970a263eca1694b5bd61cbb38f804c299018fe.jpg)



Figure 2: Average performances of distinct prompting strategies and the best one $\mathbf { P } _ { O } ^ { * }$ across benchmarks on each LLM under constrained cost O. Under the equal cost O, CoT performs best most of the time. When O grows larger, CoT gradually becomes the best prompt strategy $\mathbf { P } _ { O } ^ { * }$ on all models.


## 4 Why CoT Performs Worse with Lower N while Better with Larger N?

Let us consider a specific input question x, note the answer space $\mathcal { A } = \{ a _ { 1 } , a _ { 2 } , \ldots , a _ { m } \}$ as the set of all probable values of $\phi [ \mathcal { M } ( \boldsymbol { x } | \mathbf { P } _ { i } ) ]$ for all $\mathbf { P } _ { i } ,$ $i . e . , \phi [ \mathcal { M } ( x | \mathbf { P } _ { i } ) ] \in \mathcal { A } f o r \ \forall \ \mathbf { P } _ { i }$ . We omit $N = 1$ in $\phi [ \mathcal { M } ( \boldsymbol { x } | \mathbf { P } _ { i } ) ]$ for brevity. $\{ p _ { i , 1 } , p _ { i , 2 } , \dots , p _ { i , m } \}$ denotes the corresponding probabilities, $i . e . , p _ { i , j } =$ Pr $\left( \phi [ \mathcal { M } ( x \mid \mathbf { P } _ { i } ) ] \ = \ a _ { j } \right)$ . Note $a _ { i } ^ { * }$ as the final result of $\mathbf { P } _ { i }$ by scaling sampling N times, $i . e .$ $a _ { i } ^ { * } = \phi [ \mathcal { M } ( x \mid \mathbf { P } _ { i } ) ; N ]$ . Then the occurrence number $\mathbf { X } _ { i } ~ = ~ ( \mathbf { x } _ { i , 1 } , \ldots , \mathbf { x } _ { i , m } )$ of each probable answer for $\mathbf { P } _ { i }$ follows a multinomial distribution, $i . e .$ $\mathbf { X } _ { i } \sim M u l t ( N , p _ { i , 1 } , p _ { i , 2 } , \dotsc , p _ { i , m } )$ . The process of getting the final result $a _ { i } ^ { * }$ of $\mathbf { P } _ { i }$ by sampling N times can be formalized as: 

$$
\begin{array}{c} \mathcal {J} _ {i} = \{j   |   \mathbf {x} _ {i, j} = \max \{\mathbf {X} _ {i} \} \} \\ k \sim \text {Uniform} (\mathcal {J} _ {i}), \quad a _ {i} ^ {*} = a _ {k} \end{array}\tag{4}
$$

Next, we will introduce several lemmas and theorems to explain the two reasons why CoT sometimes performs worse with lower N while better with larger N. In the following proof, we omit the input $x ,$ assume $a _ { 1 }$ is the correct answer, and note the probability of getting $a _ { 1 }$ when sampling N times with $\mathbf { P } _ { i }$ as $\mathrm { P r } ( a _ { 1 } | \mathbf { P } _ { i } ; N )$ , which can be regarded as the expectation of the accuracy $\mathbb { 1 } \{ \phi [ \mathcal { M } ( x | \mathbf { P } _ { i } ) ; N ] = y \}$ . Details about the proof process can be found in Appendix B. 

Definition 1. Note $p _ { m a x } = \operatorname* { m a x } \{ p _ { i , 1 } , . . . , p _ { i , m } \}$ ${ \mathcal { S } } = \{ a _ { j } | p _ { i , j } = p _ { m a x } \}$ , we can define the $d i f { \bf - }$ ficulty of the input question x for $\mathbf { P } _ { i } .$ . If $a _ { 1 } ~ \in$ $s$ and $| { \cal S } | = 1$ , we call x an easy question for $\mathbf { P } _ { i \cdot } ~ H a _ { 1 } \in \mathcal { S }$ and $\vert { \cal S } \vert > 1$ , we call x a moderate question for P<sub>i</sub>. If a<sub>1</sub> $\notin S ,$ , we call x a hard questionfor P . 

Theorem 1. If x is an easy question $f o r$ $\mathbf P _ { i } , \ P r ( a _ { 1 } | \mathbf P _ { i } ; N )$ is non-decreasing w.r.t. $N ,$ lim $P r ( a _ { 1 } | \mathbf { P } _ { i } ; N ) = 1$ $N { \xrightarrow { } } + \infty$ 

Theorem 2. If x is a moderate question for $\mathbf P _ { i } , \ P r ( a _ { 1 } | \mathbf P _ { i } ; N )$ is non-decreasing w.r.t. N, $\operatorname* { l i m } _ { \substack { \mathrm { { n } } \tau } } P r ( a _ { 1 } | \mathbf { P } _ { i } ; N ) = 1 / | S |$ N + 

Theorem 3. If x is a hard question for $\mathbf { P } _ { i } ,$ $P r ( a _ { 1 } | \mathbf { P } _ { i } ; N )$ exhibits a general declining trend w.r.t. N, lim $P r ( a _ { 1 } | \mathbf { P } _ { i } ; N ) = 0$ N + 

Lemma 1. Consider a specific condition with answer space $| \mathcal { A } | \quad = \quad 3 .$ For $\begin{array} { r l r } { N } & { { } = } & { 3 , } \end{array}$ $P r ( a _ { 1 } | \mathbf { P } _ { i } ; N ) = 3 p _ { i , 1 } ^ { 2 } - 2 p _ { i , 1 } ^ { 3 } + 2 p _ { i , 1 } p _ { i , 2 } p _ { i , 3 }$ . For $N = 5 , P r ( a _ { 1 } | \mathbf { P } _ { i } ; N ) = 6 p _ { i , 1 } ^ { 5 } - 1 5 p _ { i , 1 } ^ { 4 } + 1 0 p _ { i , 1 } ^ { 3 } +$ $1 5 p _ { i , 1 } ^ { 2 } p _ { i , 2 } p _ { i , 3 } ( p _ { i , 2 } + p _ { i , 3 } ) .$ 

Theorem 4. For two prompting strategies $\mathbf { P } _ { i }$ and $\mathbf { P } _ { i ^ { \prime } } ,$ , note $p _ { i , q } ~ = ~ \operatorname* { m a x } \{ p _ { i , 2 } , \ldots , p _ { i , m } \} , ~ p _ { i ^ { \prime } , q ^ { \prime } } ~ =$ max $\{ p _ { i ^ { \prime } , 2 } , \ldots , p _ { i ^ { \prime } , m } \} , i f p _ { i , 1 } - p _ { i , q } < p _ { i ^ { \prime } , 1 } - p _ { i ^ { \prime } , q ^ { \prime } }$ and $p _ { i , 1 } + p _ { i , q } - p _ { i , 1 } ^ { 2 } - p _ { i , q } ^ { 2 } > p _ { i ^ { \prime } , 1 } + p _ { i ^ { \prime } , q ^ { \prime } } - p _ { i ^ { \prime } , 1 } ^ { 2 } -$ $p _ { i ^ { \prime } , q ^ { \prime } } ^ { 2 }$ , there exits a sufficiently large $N _ { 0 }$ such that $f o r \ ' { N } > N _ { 0 } , P r ( a _ { 1 } | \mathbf { P } _ { i } ; N ) < P r ( a _ { 1 } | \mathbf { P } _ { i ^ { \prime } } ; N )$ 

## 4.1 CoT Has More Easy Questions and Fewer Hard Questions

We identify two primary reasons why CoT sometimes performs worse with lower sample sizes (N) but achieves better performance among these prompting approaches with larger N. The first reason relates to the distribution of question difficulty for CoT. CoT has more easy questions and fewer hard questions. When sampling with lower $N , \mathbf { P } _ { i }$ still has a small probability of obtaining the right answer for hard questions, while the probability diminishes to zero as increasing N. This is the opposite of easy questions. Figure 39 shows an example of the accuracy changes on easy/hard questions. The prompting strategy with fewer hard questions and more easy questions will improve performance more rapidly when scaling. According to Theorem 1 to 3, we can calculate the extreme performance of $\mathbf { P } _ { i }$ according to the difficulty proportion of questions, $i . e . , \sum _ { x \in \mathfrak { D } } \operatorname* { l i m } _ { N  + \infty } \operatorname* { P r } ( a _ { 1 } | \mathbf { \dot { P _ { i } } } ; \mathbf { \tilde { N } } )$ Table 1 summarizes the difficulty proportion of the questions and extreme performance for each $\mathbf { P } _ { i }$ on each model. It can be observed that CoT has more easy questions and fewer hard questions, and can reach the best extreme performance on all models, thus making CoT gradually dominate as increasing N even if it has a lower pass@1 accuracy. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/7f3a8f4d24f8c98624c1f8410610c352d0a923c5ea6ee265fbee790e6f0df940.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/50b3f269b1bcb09b63b72e6011b217fc6283f3f5040bfd4de78831f63cacdb5c.jpg)



Figure 3: Illustration of the two reasons why CoT sometimes performs worse with lower N while better with larger N. Left: CoT has more easy questions and fewer hard questions. For example, the probability distribution of L2M is $\{ 0 . 4 , 0 . 5 , 0 . 1 , 0 . 0 , 0 . 0 \}$ (hard question), and $\{ 0 . 3 , 0 . 2 , 0 . 2 , 0 . 2 , 0 . 1 \}$ (easy question) for CoT. Although L2M has higher pass@1 accuracy, its accuracy reduces until 0% as scaling while CoT increases until 100%. Right: CoT is less likely to be affected by wrong answers due to their relatively uniform distribution. The probability of obtaining the right answer $a _ { 1 }$ grows more rapidly as increasing N.


## 4.2 CoT is Less Likely to be Affected by Wrong Answers

The second reason for this phenomenon is that CoT is less likely to be affected by wrong answers. $\mathrm { P r } ( a _ { 1 } | \mathbf { P } _ { i } ; N )$ is not a function of only the probability $p _ { i , 1 }$ of the right answer $a _ { 1 }$ , but also related to the probability distribution of other wrong answers. According to Theorem 4, even if $p _ { i , 1 } > p _ { i ^ { \prime } , 1 } , i . e .$ $\mathrm { P r } ( a _ { 1 } | \mathbf { P } _ { i } ; N = 1 ) > \mathrm { P r } ( a _ { 1 } | \mathbf { P } _ { i ^ { \prime } } ; N = 1 )$ , there still may exist an $N _ { 0 }$ that ${ \mathrm { P r } } ( a _ { 1 } | \mathbf { P } _ { i } ; N = N _ { 0 } ) >$ $\mathrm { P r } ( a _ { 1 } | \mathbf { P } _ { i ^ { \prime } } ; N = N _ { 0 } )$ . Considering a question x in GSM8K as an example and $a _ { 1 }$ is the correct answer, the result probability distribution of $\mathbf { P } _ { i } =$ SBP is $\{ 0 . 6 4 , 0 . 3 5 , 0 . 0 1 \}$ , and $\lbrace 0 . 6 , 0 . 2 , 0 . 2 \rbrace$ for $\mathbf P _ { i ^ { \prime } } = \mathbf C \mathbf o \mathrm { T }$ , which satisfies the condition in Theorem 4. According to Lemma 1, $\mathrm { P r } ( a _ { 1 } | \mathbf { P } _ { i } ; N =$ $1 ) \ = \ 0 . 6 4 0 > \ \operatorname* { P r } ( a _ { 1 } | \mathbf { P } _ { i ^ { \prime } } ; N \ = \ 1 ) \ = \ 0 . 6 0 0$ $\mathrm { P r } ( a _ { 1 } | \mathbf { P } _ { i } ; N = 3 ) = 0 . 7 0 9 > \mathrm { P r } ( a _ { 1 } | \mathbf { P } _ { i ^ { \prime } } ; N$ $3 ) = 0 . 6 9 6 , \mathrm { w h i l e ~ P r } ( a _ { 1 } | \mathbf { P } _ { i } ; N = 5 ) = 0 . 7 5 7 <$ $\mathrm { P r } ( a _ { 1 } | \mathbf { P } _ { i ^ { \prime } } ; N = 5 ) = 0 . 7 6 9$ . This means, although 


Table 1: Difficulty proportion of questions and extreme peformance (denote by $\mathbf { \mu ^ { 6 6 } } \mathbf { A c c } ^ { 9 9 } )$ for each $\mathbf { P } _ { i }$ and LLM across benchmarks. CoT has more easy questions and fewer hard questions, and can reach the best extreme performance on all LLMs.


<table><tr><td><eq>P_i</eq></td><td>Easy</td><td>Moderate</td><td>Hard</td><td>Acc</td><td>Easy</td><td>Moderate</td><td>Hard</td><td>Acc</td><td>Easy</td><td>Moderate</td><td>Hard</td><td>Acc</td></tr><tr><td></td><td colspan="4">Qwen2.5-7B-Instruct</td><td colspan="4">LLaMA-3-8B-Instruct</td><td colspan="4">GLM-4-9B-Chat</td></tr><tr><td>DiP</td><td>86.3%</td><td>0.3%</td><td>13.4%</td><td>86.4</td><td>69.7%</td><td>1.0%</td><td>29.3%</td><td>70.2</td><td>79.8%</td><td>0.6%</td><td>19.6%</td><td>80.1</td></tr><tr><td>CoT</td><td>88.1%</td><td>0.2%</td><td>11.6%</td><td>88.2</td><td>70.9%</td><td>0.9%</td><td>28.2%</td><td>71.3</td><td>82.8%</td><td>0.8%</td><td>16.5%</td><td>83.1</td></tr><tr><td>L2M</td><td>87.4%</td><td>0.3%</td><td>12.3%</td><td>87.6</td><td>70.3%</td><td>1.6%</td><td>28.1%</td><td>71.0</td><td>81.9%</td><td>0.4%</td><td>17.7%</td><td>82.1</td></tr><tr><td>SBP</td><td>87.1%</td><td>0.1%</td><td>12.8%</td><td>87.2</td><td>67.3%</td><td>1.3%</td><td>31.3%</td><td>68.0</td><td>81.4%</td><td>0.9%</td><td>17.6%</td><td>81.9</td></tr><tr><td>AnP</td><td>81.1%</td><td>0.5%</td><td>18.4%</td><td>81.3</td><td>67.5%</td><td>1.4%</td><td>31.1%</td><td>68.2</td><td>76.4%</td><td>1.2%</td><td>22.4%</td><td>77.0</td></tr><tr><td></td><td colspan="4">Phi-3.5-mini-Instruct</td><td colspan="4">Gemini-1.5-Flash</td><td colspan="4">GPT-4o-mini</td></tr><tr><td>DiP</td><td>78.4%</td><td>0.6%</td><td>21.1%</td><td>78.6</td><td>91.0%</td><td>0.0%</td><td>9.0%</td><td>91.0</td><td>89.7%</td><td>0.4%</td><td>9.9%</td><td>89.9</td></tr><tr><td>CoT</td><td>81.2%</td><td>0.4%</td><td>18.4%</td><td>81.4</td><td>91.2%</td><td>0.2%</td><td>8.6%</td><td>91.3</td><td>89.8%</td><td>0.3%</td><td>9.9%</td><td>90.0</td></tr><tr><td>L2M</td><td>80.2%</td><td>0.6%</td><td>19.2%</td><td>80.5</td><td>90.9%</td><td>0.2%</td><td>89.8%</td><td>90.9</td><td>89.8%</td><td>0.3%</td><td>10.0%</td><td>89.9</td></tr><tr><td>SBP</td><td>79.0%</td><td>0.6%</td><td>20.4%</td><td>79.3</td><td>90.6%</td><td>0.4%</td><td>9.0%</td><td>90.8</td><td>81.4%</td><td>0.2%</td><td>10.4%</td><td>89.5</td></tr><tr><td>AnP</td><td>77.0%</td><td>1.2%</td><td>21.8%</td><td>77.6</td><td>80.5%</td><td>0.6%</td><td>18.8%</td><td>90.9</td><td>81.4%</td><td>1.1%</td><td>17.5%</td><td>81.9</td></tr></table>


Table 2: Quantity of questions described in Section 4.2. The value $v _ { i i ^ { \prime } }$ in the ith row and i′th column represents the quantity of data that satisfies Theorem 4. Results prove that CoT has greater potential to signifi cantly increase performance as scaling.


<table><tr><td colspan="7">Qwen2.5-7B-Instruct</td></tr><tr><td><eq>\mathbf{P_{i&#x27;}}</eq></td><td>DiP↓</td><td>CoT↓</td><td>L2M↓</td><td>SBP↓</td><td>AnP↓</td><td>Sum ↓</td></tr><tr><td>DiP ↑</td><td>-</td><td>447</td><td>414</td><td>457</td><td>393</td><td>1711</td></tr><tr><td>CoT ↑</td><td>423</td><td>-</td><td>374</td><td>416</td><td>361</td><td>1574</td></tr><tr><td>L2M ↑</td><td>505</td><td>510</td><td>-</td><td>494</td><td>403</td><td>1912</td></tr><tr><td>SBP ↑</td><td>599</td><td>601</td><td>564</td><td>-</td><td>429</td><td>2193</td></tr><tr><td>AnP ↑</td><td>800</td><td>817</td><td>799</td><td>776</td><td>-</td><td>3192</td></tr><tr><td>Sum ↑</td><td>2327</td><td>2375</td><td>2151</td><td>2143</td><td>1586</td><td>-</td></tr><tr><td colspan="7">LLaMA-3-8B-Instruct</td></tr><tr><td><eq>\mathbf{P_{i&#x27;}}</eq></td><td>DiP↓</td><td>CoT↓</td><td>L2M↓</td><td>SBP↓</td><td>AnP↓</td><td>Sum↓</td></tr><tr><td>DiP ↑</td><td>-</td><td>816</td><td>794</td><td>459</td><td>513</td><td>2582</td></tr><tr><td>CoT ↑</td><td>620</td><td>-</td><td>646</td><td>382</td><td>408</td><td>2056</td></tr><tr><td>L2M ↑</td><td>639</td><td>677</td><td>-</td><td>432</td><td>393</td><td>2141</td></tr><tr><td>SBP ↑</td><td>1316</td><td>1433</td><td>1398</td><td>-</td><td>923</td><td>5070</td></tr><tr><td>AnP ↑</td><td>1243</td><td>1381</td><td>1380</td><td>871</td><td>-</td><td>4875</td></tr><tr><td>Sum ↑</td><td>3818</td><td>4307</td><td>4218</td><td>2144</td><td>2237</td><td>-</td></tr></table>

complicated prompting strategies may have higher pass@1 accuracy, they are easier to be affected by wrong answers. In contrast, simple CoT has a relatively flat distribution on wrong answers, thus making it focus more on the correct answer, which makes it more rapidly improve performance in easy questions and more slowly reduce accuracy in hard questions as increasing N, as shown in Figure 3. We record the quantity of such questions for each two prompting strategies and display the results of Qwen2.5-7B-Instruct and LLaMA-3-8B-Instruct in Table 2. If CoT is $\mathbf { P } _ { i ^ { \prime } }$ , there are the most data satisfying Theorem 4. If CoT is $\mathbf { P } _ { i } ,$ there are the least such questions. These demonstrate that CoT has greater potential to significantly increase scaling performance compared with other strategies. 


Table 3: Average KL divergence between the erroneous answer distribution and uniform distribution of each $\mathbf { P } _ { i }$ across all tested 6 benchmarks on each LLM.


<table><tr><td></td><td>Qwen</td><td>LLaMA</td><td>GLM</td><td>Phi</td><td>Gemini</td><td>GPT</td></tr><tr><td>DiP</td><td>0.0774</td><td>0.0830</td><td>0.0655</td><td>0.0770</td><td>0.0679</td><td>0.0649</td></tr><tr><td>CoT</td><td>0.0668</td><td>0.0829</td><td>0.0575</td><td>0.0678</td><td>0.0674</td><td>0.0624</td></tr><tr><td>L2M</td><td>0.0708</td><td>0.0857</td><td>0.0572</td><td>0.0724</td><td>0.0692</td><td>0.0647</td></tr><tr><td>SBP</td><td>0.0794</td><td>0.0952</td><td>0.0603</td><td>0.0821</td><td>0.0757</td><td>0.0647</td></tr><tr><td>AnP</td><td>0.1357</td><td>0.1154</td><td>0.1137</td><td>0.1097</td><td>0.1990</td><td>0.1593</td></tr></table>

To quantify the uniformity of the erroneous answer distributions, we measure the KL divergence between them and the uniform distribution, where lower values indicate closer alignment with greater uniformity. Table 3 shows the average KL divergence on each $\mathbf { P } _ { i }$ and LLM across all benchmarks. We can see that CoT has the most uniform distribution in most cases. While L2M achieves the lowest KL divergence on GLM-4-9B-Chat, CoT exhibits a marginal difference of merely 0.0003, indicating near-identical uniformity. 

As for why CoT has a more uniform probability distribution, we speculate this stems from its simpler approach that avoids imposing specific constraints or guidance on reasoning patterns, compared with other specifically designed prompting approaches. By minimizing explicit guidance, it preserves the model’s natural exploration of the solution space. While DiP is also simple, it fails to adequately activate the model’s step-by-step reasoning capabilities and free exploration potential. In contrast, more complex prompting strategies introduce explicit guidance mechanisms that constrain reasoning patterns and narrow search directions. This results in probability mass concentrating around specific potential correct solutions at the expense of broader exploratory behavior, ultimately leading to less uniform answer distributions. 

## 5 Predicting Scaling Performance and $\mathbf { P } _ { N } ^ { * }$

In practice, evaluating the test-time scaling performance requires significantly intensive resource consumption, especially with very large sampling time N. For pretraining, it is feasible to predict the train-time scaling performance based on the scaling law (Kaplan et al., 2020) through a series of low-cost experiments, while maintaining the model architecture largely unchanged and minimizing the risks associated with large-scale training. Similarly, we can also use the sample results of $\mathbf { P } _ { i }$ with fewer N to approximately get the distribution $\{ p _ { i , 1 } , p _ { i , 2 } , \dots , p _ { i , m } \}$ to predict the test-time scaling performance with larger N. Directly, one can utilize the multinomial distribution probability calculation formula (Equation 13 and 14) to calculate $\mathrm { P r } ( a _ { 1 } | \mathbf { P } _ { i } ; N )$ with enumeration or leverage numerical simulation to estimate. However, their computational complexities are both $O ( N )$ , and the former needs to traverse all situations and is dif ficult to operate. Therefore, we propose a method with the computational complexity $O ( 1 )$ to quickly predict the scaling performance of majority voting for arbitrary $\mathbf { P } _ { i } ,$ which can serve as the test-time scaling law for majority voting. It can also select the best prompting strategy $\mathbf { P } _ { N } ^ { * }$ according to the predicted performances of each $\mathbf { P } _ { i }$ 

Here we omit the prompting index i and input question $x ,$ and assume $a _ { 1 }$ is the correct answer in the following. According to Khinchin’s Law of Large Numbers and Lindeberg-Levy Central Limit Theorem, when N is large enough, each occurrence number $\mathbf { x } _ { j }$ can be approximated by a normal distribution. Specifically, for $\mathbf { x } _ { 1 }$ , we have 

$$
\mathbf {x} _ {1} \sim \mathcal {N} (N p _ {1}, N p _ {1} (1 - p _ {1})),\tag{5}
$$

$i . e .$ , a normal distribution with mean $N p _ { 1 }$ and variance $N p _ { 1 } ( 1 - p _ { 1 } )$ . Considering the maximum value among all other x<sub>j</sub> $( j ~ \neq ~ 1 )$ , denoted as $M = \operatorname* { m a x } ( \mathbf { x } _ { 2 } , . . . , \mathbf { x } _ { m } )$ , when N is large enough, the distribution of M can be approximated by 

$$
M \sim \mathcal {N} (N p _ {m a x}, N p _ {m a x} (1 - p _ {m a x})),\tag{6}
$$

where $p _ { m a x }$ is the second highest probability excluding $p _ { 1 }$ . We now need to calculate $P ( \mathbf { x } _ { 1 } > M )$ 

which can be approximated by comparing two normal distributions. Let $Z = \mathbf { x } _ { 1 } - M$ , then the distribution of $Z$ is 

$$
\mathcal {N} (N (p _ {1} - p _ {m a x}), N (p _ {1} (1 - p _ {1}) + p _ {m a x} (1 - p _ {m a x})).\tag{7}
$$

Therefore, 

$$
\operatorname * {P r} (a ^ {*} = a _ {1}) \approx \operatorname * {P r} (Z > 0),\tag{8}
$$

where $a ^ { * }$ is the final sample result. Using properties of the standard normal distribution, we can write 

$$
\begin{array}{c} \operatorname * {P r} (Z > 0) = \operatorname * {P r} \left(\frac {Z - E [ Z ]}{\sqrt {\operatorname{Var} [ Z ]}} > \frac {- E [ Z ]}{\sqrt {\operatorname{Var} [ Z ]}}\right) \\ = 1 - \Phi \left(\frac {- E [ Z ]}{\sqrt {\operatorname{Var} [ Z ]}}\right), \\ \mathbb {E} [ Z ] = N (p _ {1} - p _ {m a x}), \\ \mathbb {V} [ Z ] = N (p _ {1} (1 - p _ {1}) + p _ {m a x} (1 - p _ {m a x})) \end{array}\tag{9}
$$

where Φ is the standard normal cumulative distribution function. Thus, we can quickly predict the scaling performance and select the best prompting strategy $\mathbf { P } _ { N } ^ { * }$ with given N by 

$$
\operatorname * {P r} (a _ {1} | \mathbf {P} _ {i}; N) \approx 1 - \Phi \left(\frac {- (p _ {1} - p _ {\text {max}})}{\sqrt {\frac {p _ {1} (1 - p _ {1}) + p _ {\text {max}} (1 - p _ {\text {max}})}{N}}}\right),
$$

$$
\operatorname{Accuracy} \left(\mathbf {P} _ {i}, N\right) = \mathbb {E} _ {x \in \mathfrak {D}} \operatorname * {P r} \left(a _ {1} \mid \mathbf {P} _ {i}, N\right),\tag{10}
$$

$$
\mathbf {P} _ {N} ^ {*} = \underset {\mathbf {P} _ {i}} {\operatorname{argmax}}   \text { Accuracy } (\mathbf {P} _ {i}, N).\tag{11}
$$

(12) 

Experiment. We verify our method on LLaMA-3-8B-Instruct on GSM8K, only using 40 samples to estimate $p _ { i , j }$ . Results are shown in Figure 4. We can see that our method can accurately estimate the scaling performance, and the error decreases until 0% as scaling sampling time. When $N \geq 1 0 .$ , the error is already less than 1%. This makes sense as our method is based on the assumption that N is large enough. Although the prediction accuracy is not very high when N is small, the difference in predicted performances between distinct $\mathbf { P } _ { i }$ is similar to that in true performances, so our method can correctly select the best prompting strategy $\mathbf { P } _ { N } ^ { * }$ with arbitrary N, as shown in Table 4. 

## 6 Improving Scaling Peformance

According to the analysis in Section 4, we can further improve the scaling performance in two ways. Extensive experiments confirm their effectiveness, leading to significant improvements. We will further explore them in the future. All following results are conducted on Qwen-2.5-7B-Instruct on GSM8K. Please refer to Appendix D for more results on other LLMs and benchmarks. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/6ccc42115472fb56984a93d15c9cbcaba668b967914dc1c38918a946c7cd353d.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/09d684dda3b11a3c0e32f6582b208bd0e0115287b0bd335d95c5426cdea915bb.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/b1c771160c603b7cd963a9bba5bbd8b2a7d47c9b6a69207b2a0483b49b2bb80b.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/19da1b16c3aff01c1a36d310c117633658cbb97e086f66995ee75e1ad64d5c2d.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/f7b9a599eae8541e67d552dde01673a8145b6d3ab4222e32b484d7c7925b08b6.jpg)



Figure 4: Real and predicted performance using our method of different $\mathbf { P } _ { i }$ under various sampling time constraints. Our method can accurately estimate the scaling performance of arbitrary $\mathbf { P } _ { i }$ , especially with large $N .$



Table 4: The true best prompting strategy $\mathbf { P } _ { N } ^ { * }$ and the predicted $\mathbf { P } _ { N } ^ { * }$ using our method under various sampling time constraints. Our method can correctly predict the best prompting strategy under any constraints evaluated.


<table><tr><td rowspan="2"><eq>P_{N}^{*}</eq></td><td colspan="11">Sampling Time N</td></tr><tr><td>1</td><td>3</td><td>5</td><td>10</td><td>20</td><td>30</td><td>40</td><td>50</td><td>60</td><td>100</td><td>1000</td></tr><tr><td>Oracle</td><td>L2M</td><td>CoT</td><td>CoT</td><td>CoT</td><td>SBP</td><td>SBP</td><td>SBP</td><td>SBP</td><td>SBP</td><td>SBP</td><td>SBP</td></tr><tr><td>Predicted (Ours)</td><td>L2M</td><td>CoT</td><td>CoT</td><td>CoT</td><td>SBP</td><td>SBP</td><td>SBP</td><td>SBP</td><td>SBP</td><td>SBP</td><td>SBP</td></tr><tr><td>Correctness</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

## 6.1 Adaptively Scaling Based on the Difficulty

According to Theorem 1 to 4, it will lead to decreased performance when scaling sampling time on hard questions. Performances only continuously improve on easy questions. Therefore, when facing a hard question, we can force LLMs to only answer it once without scaling more. If the question is a moderate or easy question, LLMs scale sampling time as usual. We evaluate the performance both when forcing the LLM to determine the question difficulty itself (noted as “Adaptive”) and providing the difficulty oracle to the LLM as an upper bound reference (noted as “Oracle”), as shown in Figure 5. “Adaptive” performance is almost equal to the usual scaling performance (noted as “Vanilla”), which is because the LLM is more inclined to believe a question is easy, especially on more complicated $\mathbf { P } _ { i }$ such as SBP and AnP. Nevertheless, all $\mathbf { P } _ { i }$ can significantly improve their scaling performances with question difficulty oracles, proving the potential of this method. 

## 6.2 Dynamically Choosing the Optimal $\mathbf { P } _ { i }$

For a question $x ,$ it may be a hard question for a prompting strategy $\mathbf { P } _ { i }$ with higher accuracy, while an easy question for another strategy $\mathbf { P } _ { i ^ { \prime } }$ with lower accuracy. So if we can choose the opti mal prompting strategy for each question, it will largely improve the performance. We test the scaling performance both when forcing the LLM to choose the most suitable $\mathbf { P } _ { i }$ (noted as “Dynamic”) and providing the oracles as an upper bound (noted as “Oracle”), i.e., telling the LLM which $\mathbf { P } _ { i }$ maximizes $\mathrm { P r } ( a _ { 1 } | \mathbf { P } _ { i } ; N )$ , as shown in Figure $6 . \ { } ^ { 6 } \mathrm { D y } .$ namic” performance is almost equal to CoT. This is because Qwen believes CoT is the best $\mathbf { P } _ { i }$ among 8 prompting strategies in 99.7% of the questions. However, it can achieve significant improvement with oracles. This means that selecting the best $\mathbf { P } _ { i }$ for each question is more effective than majority voting, as “Oracle” performance with only $N = 1$ is much higher than $\forall \mathbf { P } _ { : }$ with even $N \to + \infty$ However, “Oracle” performance does not increase with scaling. This is because there are questions that are hard for all $\mathbf { P } _ { i } .$ . Even if we select the best $\mathbf { P } _ { i }$ on a question, its accuracy still reduces with scaling. So if we can combine the two methods in Section 6.1 and 6.2, it would lead to much more improvement. 

## 6.3 Combining Adaptively Scaling and Dynamically Choosing the Optimal $\mathbf { P } _ { i }$

Figure 7 reports the peformance upper bounds of each $\mathbf { P } _ { i } \in \mathcal { P } _ { 1 } + { } ^ {  } \mathrm { A d a p t i v e } ^ { , , }$ , “Dynamic”, and combining “Adaptive” and “Dynamic”. Experiment results demonstrate the powerful potential of the combined method. We will explore more feasible methods to reach this upper bound in future work. 

## 7 Related Work

Reasoning Prompting Strategies. CoT series carefully design exemplars or 0-shot prompts to unleash the potential of step-by-step solving (Wei et al., 2022; Kojima et al., 2022; Zhang et al., 2023; Fu et al., 2023). (Zhou et al., 2023; Dua et al., 2022; Khot et al., 2023) break down the question into smaller, more manageable subproblems. (Madaan et al., 2023; Kim et al., 2023) force LLMs to selfevaluate and correct. (Du et al., 2024; Liang et al., 2024; Chan et al., 2024; Liu et al., 2025; Huang et al., 2025) utilize multi-agent debate to collaborate reasoning. (Yasunaga et al., 2024; Yu et al., 2024) guide LLMs to draw experience from analogous problems. (Zheng et al., 2024; Gao et al., 2025) promote LLMs on abstract reasoning. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/66ccfb7545dd2b11e241deb9a4d71e0542ea06f9338174fa0f27eb42e309c004.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/f4312d10275250c4e8b8fae9c42451a01ec0613e5d4d9f4c2eca4c3ea1b8dbc1.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/2d6bab4230e7d8f74c4ed6f336ceccf4e817f9a65ae000fcb60968d30f759ff6.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/8efd537763eb8ab3817534aca508dbc1ab9eabf9cb51eddc55270ebde0b55e73.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/7366418bef607fed02ba289057a281d3dd2d61e21e49e84315d80afc921f112d.jpg)



Figure 5: Results of adaptively scaling for each $\mathbf { P } _ { i } \in \mathcal { P } _ { 1 }$ based on oracle and predicted question difficulty.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/384a8f8038964cf293d091b5c738f24f27b429a1257885808c809bbae47bed49.jpg)



Figure 6: Results of dynamically choosing the optimal $\mathbf { P } _ { i }$


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/8cf7c87f7dc0a0d116a63bc0b6c2029a454d0e8faea43bf7ca85e33e6b60afe2.jpg)



Figure 7: Results of combining adaptively scaling and dynamically choosing the optimal $\mathbf { P } _ { i }$ with oracles.


Scaling Test-Time Compute. Self-Consistency is a simple but effective scaling method (Wang et al., 2023b). (Li et al., 2023; Hosseini et al., 2024) train a verifier to evaluate samples and select the best solution. Some use iterative refinement (Madaan et al., 2023) or multiple rounds of debate(Du et al., 2024). Others leverage the theory of tree search (Yao et al., 2023; Ding et al., 2024; Zhang et al., 2024a) and graph search (Besta et al., 2024a; Jin et al., 2024a) to expand and aggregate reasoning paths (Besta et al., 2024b). 

Several studies have shown that scaling test-time compute optimally can be more effective than scaling model parameters (Snell et al., 2025; Open AI, 2024b). (Snell et al., 2025) investigates the most effective test-time scaling approach with the basic fixed standard prompting given auxiliary resources, e.g., available datasets for training, specifically trained verifiers, and fine-tuned models. They analyze two kinds of test-time scaling approaches: 1) searching against verifier reward models, and 2) sequential revisions with specifically trained models. In contrast, our aim is, completely relying on the LLM itself, which prompting strategy is the most effective with the basic majority voting scaling. We provide new perspectives and theories to 1) understand, 2) predict, and 3) improve the scaling performance of different prompting strategies with the most basic test-time scaling setting. 

## 8 Conclusion

We comprehensively study the behavior of various prompting strategies when scaling majority voting. Our experiments on 6 LLMs  8 prompting strategies 6 benchmarks consistently show that CoT has the potential to perform best as scaling. Theoretical analysis reveals that it benefits from fewer hard questions, more easy questions, and less susceptibility to incorrect answers, enabling more rapid performance gains. Additionally, our proposed method for predicting scaling performance offers a practical tool to select the optimal prompting strategy under given sampling time budgets. What’s more, we introduce two effective methods to further improve scaling performance. 

We also extend experiments on two more challenging reasoning benchmarks, GPQA (Rein et al., 2024) and AIME (Mathematical Association of America, 2024), further verifying the generality of our findings and methods. Our combined method can achieve a significant boost, elevating accuracy from just over 30% (majority@100) to 75.7%. Please refer to Appendix E for more details. 

## Limitations

In this paper, we mainly focus on majority voting, which is a simple but effective scaling approach. However, we don’t test on other more complex scaling approaches such as Monte Carlo Tree Search. Our finding that CoT dominates as scaling most of the time does not always hold for every LLM on every dataset, $e . g .$ , Table 4. Nevertheless, 80% of the results conform to this rule. In fact, it depends on the composition of the dataset. If we specifically collect hard questions for $\mathbf { P } _ { i }$ as a dataset, it will lead to a continuous decline performance of $\mathbf { P } _ { i } .$ Our experiments and analysis indicate that, even though some $\mathbf { P } _ { i }$ may perform poorly with lower sampling time, they hold the potential to exhibit superior performance than other prompting strategies as test-time scaling. We propose two superior methods according to rigorous theories, which can significantly improve scaling performance on each model and each benchmark we test, and we are confident in the universality of our methods. However, our experiment results indicate that LLMs alone cannot readily achieve the intended effects, pushing us to explore more practicable and effective methods in our future work. 

## Ethical Considerations

There are many potential societal consequences of our work, none which we feel must be specifically highlighted here. The sole potential risk we acknowledge is that scaling compute may result in substantial electricity consumption and carbon dioxide emissions. 

## Acknowledgment

We thank Jie Cao and Huaibo Huang for their insightful discussions. The work is supported by National Natural Science Foundation of China (Grant No. 62425606, 32341009, U21B2045) and the Strategic Priority Research Program of Chinese Academy of Sciences (Grant No. XDA0480302). 

## References



Pranjal Aggarwal, Aman Madaan, Yiming Yang, et al. 2023. Let’s sample step by step: Adaptiveconsistency for efficient reasoning and coding with llms. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 12375–12396. 





Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak 





Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. 2023. Palm 2 technical report. arXiv preprint arXiv:2305.10403. 





Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Michal Podstawski, Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Hubert Niewiadomski, Piotr Nyczyk, et al. 2024a. Graph of thoughts: Solving elaborate problems with large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 17682–17690. 





Maciej Besta, Florim Memedi, Zhenyu Zhang, Robert Gerstenberger, Nils Blach, Piotr Nyczyk, Marcin Copik, Grzegorz Kwasniewski, Jürgen Müller, Lukas´ Gianinazzi, et al. 2024b. Topologies of reasoning: Demystifying chains, trees, and graphs of thoughts. arXiv preprint arXiv:2401.14295. 





Zhenni Bi, Kai Han, Chuanjian Liu, Yehui Tang, and Yunhe Wang. 2024. Forest-of-thought: Scaling testtime compute for enhancing llm reasoning. arXiv preprint arXiv:2412.09078. 





Chi-Min Chan, Weize Chen, Yusheng Su, Jianxuan Yu, Wei Xue, Shanghang Zhang, Jie Fu, and Zhiyuan Liu. 2024. Chateval: Towards better llm-based evaluators through multi-agent debate. In The Twelfth International Conference on Learning Representations. 





Lingjiao Chen, Jared Quincy Davis, Boris Hanin, Peter Bailis, Ion Stoica, Matei Zaharia, and James Zou. 2024a. Are more llm calls all you need? towards the scaling properties of compound ai systems. In The Thirty-eighth Annual Conference on Neural Information Processing Systems. 





Lingjiao Chen, Matei Zaharia, and James Zou. 2024b. Frugalgpt: How to use large language models while reducing cost and improving performance. Transactions on Machine Learning Research. 





Qiguang Chen, Libo Qin, WANG Jiaqi, Jingxuan Zhou, and Wanxiang Che. 2024c. Unlocking the capabil ities of thought: A reasoning boundary framework to quantify and optimize chain-of-thought. In The Thirty-eighth Annual Conference on Neural Information Processing Systems. 





Xingyu Chen, Jiahao Xu, Tian Liang, Zhiwei He, Jianhui Pang, Dian Yu, Linfeng Song, Qiuzhi Liu, Mengfei Zhou, Zhuosheng Zhang, et al. 2024d. Do not think that much for 2+ 3=? on the overthinking of o1-like llms. arXiv preprint arXiv:2412.21187. 





Yanxi Chen, Xuchen Pan, Yaliang Li, Bolin Ding, and Jingren Zhou. 2024e. A simple and provable scaling law for the test-time compute of large language models. arXiv preprint arXiv:2411.19477. 





Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168. 





Yingqian Cui, Pengfei He, Xianfeng Tang, Qi He, Chen Luo, Jiliang Tang, and Yue Xing. 2024. A theoretical understanding of chain-of-thought: Coherent reasoning and error-aware demonstration. In The 28th International Conference on Artificial Intelligence and Statistics. 





Mehul Damani, Idan Shenfeld, Andi Peng, Andreea Bobu, and Jacob Andreas. 2025. Learning how hard to think: Input-adaptive allocation of lm computation. In The Thirteenth International Conference on Learning Representations. 





Ruomeng Ding, Chaoyun Zhang, Lu Wang, Yong Xu, Minghua Ma, Wei Zhang, Si Qin, Saravan Rajmohan, Qingwei Lin, and Dongmei Zhang. 2024. Everything of thoughts: Defying the law of penrose triangle for thought generation. In Findings of the Association for Computational Linguistics: ACL 2024, pages 1638–1662. Association for Computational Linguistics. 





Yilun Du, Shuang Li, Antonio Torralba, Joshua B Tenenbaum, and Igor Mordatch. 2024. Improving factuality and reasoning in language models through multiagent debate. In Forty-first International Conference on Machine Learning. 





Dheeru Dua, Shivanshu Gupta, Sameer Singh, and Matt Gardner. 2022. Successive prompting for decomposing complex questions. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 1251–1265. 





Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783. 





Guhao Feng, Bohang Zhang, Yuntian Gu, Haotian Ye, Di He, and Liwei Wang. 2024. Towards revealing the mystery behind chain of thought: a theoretical perspective. Advances in Neural Information Processing Systems, 36. 





Yao Fu, Hao Peng, Ashish Sabharwal, Peter Clark, and Tushar Khot. 2023. Complexity-based prompting for multi-step reasoning. In The Eleventh International Conference on Learning Representations. 





Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. 2023. Pal: program-aided language models. In Proceedings of the 40th International Conference on Machine Learning, pages 10764– 10799. 





Silin Gao, Jane Dwivedi-Yu, Ping Yu, Xiaoqing Ellen Tan, Ramakanth Pasunuru, Olga Golovneva, Koustuv Sinha, Asli Celikyilmaz, Antoine Bosselut, and Tianlu Wang. 2025. Efficient tool use with chain of-abstraction reasoning. In Proceedings ofthe 31st International Conference on Computational Linguistics, pages 2727–2743. 





Team GLM, Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang, Da Yin, Dan Zhang, Diego Rojas, Guanyu Feng, Hanlin Zhao, et al. 2024. Chatglm: A family of large language models from glm-130b to glm-4 all tools. arXiv preprint arXiv:2406.12793. 





Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021a. Measuring massive multitask language understanding. In International Conference on Learning Representations. 





Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021b. Measuring mathematical problem solving with the math dataset. In Thirtyfifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2). 





Arian Hosseini, Xingdi Yuan, Nikolay Malkin, Aaron Courville, Alessandro Sordoni, and Rishabh Agarwal. 2024. V-star: Training verifiers for self-taught reasoners. In Conference On Language Modeling. 





Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song, and Denny Zhou. 2024. Large language models cannot self-correct reasoning yet. In The Twelfth International Conference on Learning Representations. 





Ziyang Huang, Jun Zhao, and Kang Liu. 2025. Towards adaptive mechanism activation in language agent. In Proceedings ofthe 31st International Conference on Computational Linguistics, pages 2867–2885. 





Yixin Ji, Juntao Li, Hai Ye, Kaixin Wu, Jia Xu, Linjian Mo, and Min Zhang. 2025. Test-time computing: from system-1 thinking to system-2 thinking. arXiv preprint arXiv:2501.02497. 





Bowen Jin, Chulin Xie, Jiawei Zhang, Kashob Kumar Roy, Yu Zhang, Zheng Li, Ruirui Li, Xianfeng Tang, Suhang Wang, Yu Meng, and Jiawei Han. 2024a. Graph chain-of-thought: Augmenting large language models by reasoning on graphs. In Findings ofthe Associationfor Computational Linguistics: ACL 2024, pages 163–184. 





Mingyu Jin, Qinkai Yu, Dong Shu, Haiyan Zhao, Wenyue Hua, Yanda Meng, Yongfeng Zhang, and Mengnan Du. 2024b. The impact of reasoning step length on large language models. In Findings of the Associationfor Computational Linguistics: ACL 2024, pages 1830–1842. 





Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. 2020. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361. 





Tushar Khot, Harsh Trivedi, Matthew Finlayson, Yao Fu, Kyle Richardson, Peter Clark, and Ashish Sabharwal. 2023. Decomposed prompting: A modular approach for solving complex tasks. In The Eleventh 





International Conference on Learning Representations. 





Geunwoo Kim, Pierre Baldi, and Stephen McAleer. 2023. Language models can solve computer tasks. In Advances in Neural Information Processing Systems, volume 36, pages 39648–39677. 





Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large language models are zero-shot reasoners. In Advances in neural information processing systems, volume 35, pages 22199–22213. 





Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serv ing with pagedattention. In Proceedings ofthe 29th Symposium on Operating Systems Principles, pages 611–626. 





Yifei Li, Zeqi Lin, Shizhuo Zhang, Qiang Fu, Bei Chen, Jian-Guang Lou, and Weizhu Chen. 2023. Making language models better reasoners with step-aware verifier. In Proceedings of the 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 5315–5333. 





Yiwei Li, Peiwen Yuan, Shaoxiong Feng, Boyuan Pan, Xinglin Wang, Bin Sun, Heda Wang, and Kan Li. 2024. Escape sky-high cost: Early-stopping selfconsistency for multi-step reasoning. In The Twelfth International Conference on Learning Representations. 





Tian Liang, Zhiwei He, Wenxiang Jiao, Xing Wang, Yan Wang, Rui Wang, Yujiu Yang, Shuming Shi, and Zhaopeng Tu. 2024. Encouraging divergent thinking in large language models through multi-agent debate. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing. 





Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. 2024. Let’s verify step by step. In The Twelfth International Conference on Learning Representations. 





Yexiang Liu, Jie Cao, Zekun Li, Ran He, and Tieniu Tan. 2025. Breaking mental set to improve reasoning through diverse multi-agent debate. In The Thirteenth International Conference on Learning Representations. 





Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. 2023. Self-refine: iterative refinement with self-feedback. In Proceedings of the 37th International Conference on Neural Information Processing Systems, pages 46534–46594. 





Mathematical Association of America. 2024. American invitational mathematics examination (AIME). 





Open AI. 2024a. GPT-4o-mini. https://openai.com /ja-JP/index/gpt-4o-mini-advancing-cost-e fficient-intelligence. 





Open AI. 2024b. Introducing openai o1. https://op enai.com/o1/. 





Zhenting Qi, Mingyuan Ma, Jiahang Xu, Li Lyna Zhang, Fan Yang, and Mao Yang. 2025. Mutual reasoning makes smaller llms stronger problem-solvers. In The Thirteenth International Conference on Learning Representations. 





David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. 2024. Gpqa: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling. 





Marija Šakota, Maxime Peyrard, and Robert West. 2024. Fly-swat or cannon? cost-effective language model choice via meta-modeling. In Proceedings of the 17th ACM International Conference on Web Search and Data Mining, pages 606–615. 





Andries Petrus Smit, Nathan Grinsztajn, Paul Duckworth, Thomas D Barrett, and Arnu Pretorius. 2024. Should we be going mad? a look at multi-agent debate strategies for llms. In Forty-first International Conference on Machine Learning. 





Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. 2025. Scaling llm test-time compute optimally can be more effective than scaling model parameters. In The Thirteenth International Conference on Learning Representations. 





Zayne Sprague, Fangcong Yin, Juan Diego Rodriguez, Dongwei Jiang, Manya Wadhwa, Prasann Singhal, Xinyu Zhao, Xi Ye, Kyle Mahowald, and Greg Durrett. 2025. To cot or not to cot? chain-of-thought helps mainly on math and symbolic reasoning. In The Thirteenth International Conference on Learning Representations. 





Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. 2024. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530. 





Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288. 





Boshi Wang, Sewon Min, Xiang Deng, Jiaming Shen, You Wu, Luke Zettlemoyer, and Huan Sun. 2023a. Towards understanding chain-of-thought prompting: An empirical study of what matters. In The 61st Annual Meeting OfThe Association For Computational Linguistics. 





Peiyi Wang, Lei Li, Zhihong Shao, Runxin Xu, Damai Dai, Yifei Li, Deli Chen, Yu Wu, and Zhifang Sui. 2024. Math-shepherd: Verify and reinforce llms stepby-step without human annotations. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9426–9439. 





Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2023b. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations. 





Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. 2022. Chain-of-thought prompting elicits reasoning in large language models. In Advances in neural information processing systems, volume 35, pages 24824–24837. 





An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. 2024a. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115. 





Chenxiao Yang, Zhiyuan Li, and David Wipf. 2024b. An in-context learning theoretic analysis of chainof-thought. In ICML 2024 Workshop on In-Context Learning. 





Wenkai Yang, Shuming Ma, Yankai Lin, and Furu Wei. 2025. Towards thinking-optimal scaling of test-time compute for llm reasoning. arXiv preprint arXiv:2502.18080. 





Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L Griffiths, Yuan Cao, and Karthik Narasimhan. 2023. Tree of thoughts: deliberate problem solving with large language models. In Proceedings of the 37th International Conference on Neural Information Processing Systems, pages 11809–11822. 





Michihiro Yasunaga, Xinyun Chen, Yujia Li, Panupong Pasupat, Jure Leskovec, Percy Liang, Ed H Chi, and Denny Zhou. 2024. Large language models as analogical reasoners. In The Twelfth International Conference on Learning Representations. 





Junchi Yu, Ran He, and Zhitao Ying. 2024. Thought propagation: An analogical approach to complex rea soning with large language models. In The Twelfth International Conference on Learning Representations. 





Zishun Yu, Tengyu Xu, Di Jin, Karthik Abinav Sankararaman, Yun He, Wenxuan Zhou, Zhouhao Zeng, Eryk Helenowski, Chen Zhu, Sinong Wang, et al. 2025. Think smarter not harder: Adaptive reasoning with inference aware optimization. arXiv preprint arXiv:2501.17974. 





Murong Yue, Jie Zhao, Min Zhang, Liang Du, and Ziyu Yao. 2024. Large language model cascades with 





mixture of thought representations for cost-efficient reasoning. In The Twelfth International Conference on Learning Representations. 





Di Zhang, Xiaoshui Huang, Dongzhan Zhou, Yuqiang Li, and Wanli Ouyang. 2024a. Accessing gpt-4 level mathematical olympiad solutions via monte carlo tree self-refine with llama-3 8b. arXiv preprint arXiv:2406.07394. 





Kexun Zhang, Shang Zhou, Danqing Wang, William Yang Wang, and Lei Li. 2024b. Scaling llm inference with optimized sample compute allocation. arXiv preprint arXiv:2410.22480. 





Zhuosheng Zhang, Aston Zhang, Mu Li, and Alex Smola. 2023. Automatic chain of thought prompting in large language models. In The Eleventh International Conference on Learning Representations. 





Huaixiu Steven Zheng, Swaroop Mishra, Xinyun Chen, Heng-Tze Cheng, Ed H Chi, Quoc V Le, and Denny Zhou. 2024. Take a step back: Evoking reasoning via abstraction in large language models. In The Twelfth International Conference on Learning Representations. 





Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc V Le, et al. 2023. Least-to-most prompting enables complex reasoning in large language models. In The Eleventh International Conference on Learning Representations. 



## Appendix

In Appendix, we present broader related work, proofs for our theorems, detailed results on each benchmark, more discussions on improving the scaling performance, results of extended experiments, implementation details and prompts. The content structure is outlined as follows: 

• Appendix A - Broader Related Work 

• Appendix B - Proofs 

• Appendix C - Detailed Results 

• Appendix D - More Discussions on Improving the Scaling Performance 

– Appendix D.1 - Adaptively Scaling Based on the Difficulty 

– Appendix D.2 - Dynamically Choosing the Optimal $\mathbf { P } _ { i }$ 

– Appendix D.3 - Combining Adaptively Scaling and Dynamically Choosing the Optimal $\mathbf { P } _ { i }$ 

• Appendix E - Extended Experiments 

• Appendix F - Implementation Details and Prompts 

## A Broader Related Work

Efficient Reasoning. (Aggarwal et al., 2023; Li et al., 2024; Chen et al., 2024a) improve the reasoning efficiency with majority voting by adjusting the sampling time. (Damani et al., 2025; Zhang et al., 2024b) learn to dynamically allocate resources under limited sampling time budgets. (Chen et al., 2024b; Yue et al., 2024; Šakota et al., 2024) leverage multiple models with different prices to reduce cost while maintaining performance. (Yang et al., 2025; Chen et al., 2024d; Yu et al., 2025) reduce the length of the thinking process to alleviate the overthinking issue, to achieve efficient reasoning. 

Role and Mechanism of CoT and Test-Time Scaling. (Jin et al., 2024b) studies the impact of reasoning step length of CoT. (Wang et al., 2023a) studies what makes CoT prompting effective, indicating that being relevant to the query and correctly ordering the reasoning steps are more important. (Feng et al., 2024; Cui et al., 2024) analyze the mechanism of CoT from a theoretical perspective. (Sprague et al., 2025) points out that CoT helps mainly on math and symbolic reasoning by sorting and analyzing a large number of experimental results. (Chen et al., 2024c) proposes a framework to quantify the reasoning boundary of CoT. (Yang et al., 2024b) provides an in-context learning analysis of CoT. (Chen et al., 2024a) investigates and analyzes the performance changes with more LLM calls. (Chen et al., 2024e) proves that the failure probability of test-time scaling decays to zero exponentially or by a power law. 

## B Proofs

Theorem 1. If x is an easy question for $\mathbf P _ { i } , \ \operatorname* { P r } ( a _ { 1 } | \mathbf P _ { i } ; N )$ is non-decreasing w.r.t. N, $\operatorname* { l i m } _ { N  + \infty } \mathrm { P r } ( a _ { 1 } | \mathbf { P } _ { i } ; N ) = 1$ 

Theorem 2. If x is a moderate question for $\mathbf P _ { i } , \ \mathrm { P r } ( a _ { 1 } | \mathbf P _ { i } ; N )$ is non-decreasing w.r.t. N, $\operatorname* { l i m } _ { \tau } \mathrm { P r } ( a _ { 1 } | \mathbf { P } _ { i } ; N ) = 1 / | \cal { S } |$ N + 

Theorem 3. If x is a hard question for $\mathbf { P } _ { i }$ $\mathrm { P r } ( a _ { 1 } | \mathbf { P } _ { i } ; N )$ exhibits a general declining trend w.r.t. ${ \cal N } , \operatorname * { l i m } _ { N  + \infty } \mathrm { P r } ( a _ { 1 } | { \bf P } _ { i } ; N ) = 0$ 

Proof. The occurrence number $\mathbf { X } _ { i }$ $\left( \mathbf { x } _ { i , 1 } , \ldots , \mathbf { x } _ { i , m } \right)$ of each probable answer for $\mathbf { P } _ { i }$ follows a multinomial distribution, $i . e . , \ \mathbf { X } _ { i } \ \sim$ $M u l t ( N , p _ { i , 1 } , p _ { i , 2 } , \dots , p _ { i , m } )$ . When sampling N times, the specific probability of a certain occurrence number can be calculated with Equation 13. For brevity, we omit the input x, sampling time N, and the prompting index i in $\mathbf { x } _ { i , j }$ and $p _ { i , j }$ in the following equations. 

$$
\operatorname * {P r} \left(\mathbf {x} _ {1} = k _ {1}, \mathbf {x} _ {2} = k _ {2}, \ldots , \mathbf {x} _ {m} = k _ {m}\right)
$$

$$
= \underbrace {\frac {N !}{k _ {1} ! k _ {2} ! \cdots k _ {m} !}} _ {\text {coefficient}} \underbrace {p _ {1} ^ {k _ {1}} p _ {2} ^ {k _ {2}} \cdots p _ {m} ^ {k _ {m}}} _ {\text {probability term}}   a \text {term in} \operatorname * {P r} (a _ {1} | \mathbf {P} _ {i}; N)\tag{13}
$$

$$
s. t. \sum_ {j = 1} ^ {m} k _ {j} = N, \sum_ {j = 1} ^ {m} p _ {j} = 1
$$

Assuming the correct answer is $a _ { 1 }$ , M = max $\cdot ( k _ { 2 } , \ldots , k _ { m } )$ , the probability of obtaining the right answer by sampling N times with $\mathbf { P } _ { i }$ is 

$$
\begin{array}{l} \operatorname * {P r} \left(a _ {1} | \mathbf {P} _ {i}\right) = \operatorname * {P r} \left(\mathbf {x} _ {1} > M\right) + \sum_ {| \mathcal {J} | = 1} ^ {m - 1} \\ \frac {\operatorname * {P r} \left(\mathbf {x} _ {1} = \mathbf {x} _ {j} > \mathbf {x} _ {q} f o r a l l j \in \mathcal {J} , q \notin \{1 \} \cup \mathcal {J}\right)}{| \mathcal {J} | + 1}, \end{array} \tag {14}\tag{14}
$$

where $\mathcal { I }$ is the set of all indexes $j \left( j \neq 1 \right)$ of $\mathbf { x } _ { j }$ 

that satisfies $\mathbf { x } _ { j } = \mathbf { x } _ { 1 }$ 

$$
\begin{array}{l} \operatorname * {P r} _ {1} = \operatorname * {P r} \left(\mathbf {x} _ {1} > M\right) = \\ \sum_ {\sum_ {j = 1} ^ {m} k _ {j} = N} \frac {N !}{k _ {1} ! \cdots k _ {m} !} p _ {1} ^ {k _ {1}} \prod_ {j = 2} ^ {m} p _ {j} \mathbb {1} (k _ {j} <   k _ {1}), \end{array}\tag{15}
$$

$$
\begin{array}{l} \operatorname * {P r} _ {2} = \operatorname * {P r} (\mathbf {x} _ {1} = \mathbf {x} _ {j} > \mathbf {x} _ {q} f o r a l l j \in \mathcal {J}, q \notin \{1 \} \cup \mathcal {J})) \\ = \sum_ {\sum_ {j = 1} ^ {m} k _ {j} = N} \frac {N !}{k _ {1} ! | \mathcal {I} | \prod_ {q \notin \{1 \} \cup J} k _ {k} !} p _ {1} \prod_ {j \in \mathcal {J}} p _ {j} \prod_ {q \notin \{1 \} \cup \mathcal {J}} p _ {k} \mathbb {1} (k _ {k} <   k _ {1}), \end{array}\tag{16}
$$

$\mathrm { P r } _ { 1 }$ represents the probability that $\mathbf { x } _ { 1 }$ is the only maximum number in $\mathbf { X } _ { i } . \mathrm { P r } _ { 2 }$ denotes the probability that there exists more than one maximum number and correctly obtains $a _ { 1 }$ by randomly choosing from them. 

Here we present a generalized representation. As shown in Equation 14, Pr $\left( a _ { 1 } | \mathbf { P } _ { i } \right)$ only includes the cases where ${ \bf X } _ { i , 1 }$ is the maximum value (maybe not the only one). Therefore, for a certain occurrence number $\mathbf { X } _ { i } ~ = ~ \left( \mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { m } \right)$ of each probable answer $\{ a _ { 1 } , \ldots , a _ { m } \}$ , we can reorder $a _ { 2 } , \ldots , a _ { m }$ to obtain ${ \bf x } _ { 1 } = { \bf x } _ { 2 } = \cdot \cdot \cdot = { \bf x } _ { l } >$ $\mathbf { x } _ { l + 1 } , \mathbf { x } _ { l + 2 } , \ldots , \mathbf { x } _ { m }$ , where $1 \leq l \leq m$ . When $l = 1 , \mathbf { x } _ { 1 }$ is the only maximum value. So each term in $\mathrm { P r } ( a _ { 1 } | \mathbf { P } _ { i } ; N )$ can be written as 

$$
\frac {1}{l} \cdot \frac {\left(l k + \sum_ {j = l + 1} ^ {m} k _ {j}\right) !}{(k !) ^ {l} \cdot \prod_ {j = l + 1} ^ {m} (k _ {j} !)} p _ {1} ^ {k} p _ {2} ^ {k} \dots p _ {l} ^ {k} p _ {l + 1} ^ {k _ {l + 1}} p _ {l + 2} ^ {k _ {l + 2}} \dots p _ {m} ^ {k _ {m}}\tag{17}
$$

where $k _ { 1 } = k _ { 2 } = \dots = k _ { l } = k > k _ { j } , j =$ $l + 1 , . . . , m$ and $\begin{array} { r } { l k + \sum _ { i = l + 1 } ^ { m } k _ { j } = N } \end{array}$ 

Now we prove Theorem 1 and 2. We aim to prove that given the set of answers $\{ a _ { 1 } , a _ { 2 } , \ldots , a _ { m } \}$ with associated probabilities $\{ p _ { 1 } , p _ { 2 } , \dots , p _ { m } \}$ from $\mathbf { P } _ { i }$ , we have $\mathrm { P r } ( a _ { 1 } | \mathbf { P } _ { i } ; N +$ $1 ) \geq \operatorname* { P r } ( a _ { 1 } | \mathbf { P } _ { i } ; N )$ for any $N ~ \in ~ \mathbb { N } ^ { + }$ . Due to $\begin{array} { r } { \sum _ { j = 1 } ^ { m } p _ { j } = 1 } \end{array}$ , the given proposition can be restated as 

$$
\operatorname * {P r} (a _ {1} | \mathbf {P} _ {i}; N + 1) - (\sum_ {j = 1} ^ {m} p _ {j}) \cdot \operatorname * {P r} (a _ {1} | \mathbf {P} _ {i}; N) \geq 0.\tag{18}
$$

We consider the probability term $p _ { 1 } ^ { k } p _ { 2 } ^ { k } \cdot \cdot \cdot p _ { l } ^ { k } p _ { l + 1 } ^ { k _ { l + 1 } } p _ { l + 2 } ^ { k _ { l + 2 } } \cdot \cdot \cdot p _ { m } ^ { k _ { m } }$ in each term in $\mathrm { P r } ( a _ { 1 } | \mathbf { P } _ { i } ; N )$ ), i.e, Equation 17. When it times $\textstyle \sum _ { j = 1 } ^ { m } p _ { j }$ , there will be three cases. 

Case 1: When it times p<sub>1</sub>, x<sub>1</sub> is the only maximum value. The probability term becomes 

$$
p _ {1} ^ {k + 1} p _ {2} ^ {k} \cdot \cdot \cdot p _ {l} ^ {k} p _ {l + 1} ^ {k _ {l + 1}} p _ {l + 2} ^ {k _ {l + 2}} \cdot \cdot \cdot p _ {m} ^ {k _ {m}},
$$

Case 2: When it times $p _ { s }$ , where $s \in \{ 2 , . . . , l \}$ $\mathbf { x } _ { s }$ become the only maximum value. In this situation, its final result would be an incorrect answer. If $l = 1$ , case 2 will not exist. The probability term becomes 

$$
p _ {1} ^ {k} p _ {2} ^ {k} \cdot \cdot \cdot p _ {s - 1} ^ {k} p _ {s} ^ {k + 1} p _ {s + 1} ^ {k} \cdot \cdot \cdot p _ {l} ^ {k} p _ {l + 1} ^ {k _ {l + 1}} p _ {l + 2} ^ {k _ {l + 2}} \cdot \cdot \cdot p _ {m} ^ {k _ {m}}
$$

Case 3: When it times $p _ { t }$ , where $t \in \{ l +$ $1 , . . . , m \}$ , the value of $\mathbf { x } _ { t }$ changes from $k _ { t }$ to $k _ { t } + 1$ If $k _ { t } = k - 1$ $\mathbf { x } _ { t }$ becomes a new maximum value. If $l = m$ , case 3 will not exist. The probability term becomes 

$$
p _ {1} ^ {k} p _ {2} ^ {k} \cdot \cdot \cdot p _ {l} ^ {k} p _ {l + 1} ^ {k _ {l + 1}} p _ {l + 2} ^ {k _ {l + 2}} \cdot \cdot \cdot p _ {t - 1} ^ {k _ {t - 1}} p _ {t} ^ {k _ {t} + 1} p _ {t + 1} ^ {k _ {t + 1}} \cdot \cdot \cdot p _ {m} ^ {k _ {m}}.
$$

It can be seen that case 1 and case 3 are also present in $\mathrm { P r } ( a _ { 1 } | \mathbf { P } _ { i } ; N + 1 )$ , whereas case 2 does not. We begin by considering case 1 and case 2. The terms in $\mathrm { P r } ( a _ { 1 } | \mathbf { P } _ { i } ; N + 1 )$ corresponding to case $1 p _ { 1 } ^ { k + 1 } p _ { 2 } ^ { k } \cdot \cdot \cdot p _ { l } ^ { k } p _ { l + 1 } ^ { k _ { l + 1 } } p _ { l + 2 } ^ { k _ { l + 2 } } \cdot \cdot \cdot p _ { m } ^ { k _ { m } }$ are shown in Equation 19, and no term in $\mathrm { P r } ( a _ { 1 } | \mathbf { P } _ { i } ; N \ + \ 1 )$ involves case 2. The terms in $\begin{array} { r } { \big ( \sum _ { i = 1 } ^ { m } p _ { j } \big ) \ \cdot \ \mathrm { P r } \big ( a _ { 1 } | \mathrm { \bf P } _ { i } ; N \big ) } \end{array}$ involving case 1 are shown in Equation 20. The terms in $\begin{array} { r } { ( \sum _ { i = 1 } ^ { m } p _ { j } ) \cdot \mathrm { P r } ( a _ { 1 } | \bar { \bf P } _ { i } ; N ) } \end{array}$ involving case $\mathrm { ~ ? ~ } p _ { 1 } ^ { k } p _ { 2 } ^ { k } \cdot \cdot \cdot p _ { s - 1 } ^ { k } p _ { s } ^ { k + 1 } p _ { s + 1 } ^ { k } \cdot \cdot \cdot p _ { l } ^ { k } p _ { l + 1 } ^ { k _ { l + 1 } } p _ { l + 2 } ^ { k _ { l + 2 } } \cdot \cdot \cdot p _ { m } ^ { k _ { m } }$ are shown in Equation 21. Based on the fact of Equation 22, we can establish the inequality Equation $2 3 , i . e .$ , the terms corresponding to case 1 and case 2 in $\operatorname* { P r } ( a _ { 1 } | \mathbf { P } _ { i } ; N + 1 )$ are greater than or equal to those in $\begin{array} { r l } { ~ } & { { } \big ( \sum _ { i = 1 } ^ { m } p _ { j } \big ) \cdot \mathrm { P r } \big ( a _ { 1 } | \mathrm { \bf P } _ { i } ; N \big ) } \end{array}$ 

Now we consider case 3 $p _ { 1 } ^ { k } p _ { 2 } ^ { k } \cdot \cdot \cdot p _ { l } ^ { k } p _ { l + 1 } ^ { k _ { l + 1 } } p _ { l + 2 } ^ { k _ { l + 2 } } \cdot \cdot \cdot p _ { t - 1 } ^ { k _ { t - 1 } } p _ { t } ^ { k _ { t } + 1 } p _ { t + 1 } ^ { k _ { t + 1 } } \cdot \cdot \cdot p _ { m } ^ { k _ { m } } ,$ which can be analyzed by splitting it into two distinct scenarios: $k _ { t } + 1 < k$ and $k _ { t } + 1 = k$ 

For scenario $k _ { t } + 1 < k _ { \mathrm { : } }$ , the terms in $\mathrm { P r } ( a _ { 1 } | \mathbf { P } _ { i } ; N + 1 )$ corresponding to case 3 are shown in Equation 24. The terms in $\textstyle { \bigl ( } \sum _ { j = 1 } ^ { m } p _ { j } { \bigr ) }$ $\mathrm { P r } ( a _ { 1 } | \mathbf { P } _ { i } ; N )$ corresponding to case 3 are shown in Equation 25. Evidently, we can obtain Equation 26 similar to Equation 22, and then we can get Equation 27, which proves the terms corresponding to the scenario $k _ { t } + 1 < k$ in $\operatorname* { P r } ( a _ { 1 } | \mathbf { P } _ { i } ; N + 1 )$ are equal to those in $\begin{array} { r } { ( \sum _ { i = 1 } ^ { m } p _ { j } ) \cdot \mathrm { P r } ( a _ { 1 } | \mathrm { \bf P } _ { i } ; N ) } \end{array}$ 

For scenario $ k _ { t } + 1 ~ = ~ k _ {  }$ , in a similar manner, according to the Equation 28, we obtain the same result as the above scenario, i.e., the terms corresponding to the scenario $k _ { t } + 1 ~ = ~ k$ in $\operatorname* { P r } ( a _ { 1 } | \mathbf { P } _ { i } ; N + 1 )$ are equal to those in $\textstyle { \bigl ( } \sum _ { j = 1 } ^ { m } p _ { j } { \bigr ) }$ 

$$
\frac {\left(l k + 1 + \sum_ {j = l + 1} ^ {m} k _ {j}\right) !}{(k + 1) \cdot (k !) ^ {l} \cdot \prod_ {j = l + 1} ^ {m} (k _ {j} !)} p _ {1} ^ {k + 1} p _ {2} ^ {k} \dots p _ {l} ^ {k} p _ {l + 1} ^ {k _ {l + 1}} p _ {l + 2} ^ {k _ {l + 2}} \dots p _ {m} ^ {k _ {m}}\tag{19}
$$

$$
\begin{array}{r l} & p _ {1} \cdot \frac {1}{l} \cdot \frac {\left(l k + \sum_ {j = l + 1} ^ {m} k _ {j}\right) !}{(k !) ^ {l} \cdot \prod_ {j = l + 1} ^ {m} (k _ {j} !)} p _ {1} ^ {k} p _ {2} ^ {k} \dots p _ {l} ^ {k} p _ {l + 1} ^ {k _ {l + 1}} p _ {l + 2} ^ {k _ {l + 2}} \dots p _ {m} ^ {k _ {m}} \\ & + \sum_ {s = 2} ^ {l} p _ {s} \cdot \frac {\left(l k + \sum_ {j = l + 1} ^ {m} k _ {j}\right) !}{\frac {k + 1}{k} \cdot (k !) ^ {l} \cdot \prod_ {j = l + 1} ^ {m} (k _ {j} !)} p _ {1} ^ {k + 1} p _ {2} ^ {k} \dots p _ {s - 1} ^ {k} p _ {s} ^ {k - 1} p _ {s + 1} ^ {k} \dots p _ {l} ^ {k} p _ {l + 1} ^ {k _ {l + 1}} p _ {l + 2} ^ {k _ {l + 2}} \dots p _ {m} ^ {k _ {m}} \\ & + \sum_ {t = l + 1} ^ {m} p _ {t} \cdot \frac {\left(l k + \sum_ {j = l + 1} ^ {m} k _ {j}\right) !}{\frac {k + 1}{k _ {t}} \cdot (k !) ^ {l} \cdot \prod_ {j = l + 1} ^ {m} (k _ {j} !)} p _ {1} ^ {k + 1} p _ {2} ^ {k} \dots p _ {l} ^ {k} p _ {l + 1} ^ {k _ {l + 1}} p _ {l + 2} ^ {k _ {l + 2}} \dots p _ {t - 1} ^ {k _ {t - 1}} p _ {t} ^ {k _ {t - 1}} p _ {t + 1} ^ {k _ {t + 1}} \dots p _ {m} ^ {k _ {m}} \end{array}\tag{20}
$$

$$
\sum_ {s = 2} ^ {l} p _ {s} \cdot \frac {1}{l} \cdot \frac {\left(l k + \sum_ {j = l + 1} ^ {m} k _ {j}\right) !}{(k !) ^ {l} \cdot \prod_ {j = l + 1} ^ {m} (k _ {j} !)} p _ {1} ^ {k} p _ {2} ^ {k} \dots p _ {s - 1} ^ {k} p _ {s} ^ {k} p _ {s + 1} ^ {k} \dots p _ {l} ^ {k} p _ {l + 1} ^ {k _ {l + 1}} p _ {l + 2} ^ {k _ {l + 2}} \dots p _ {m} ^ {k _ {m}}\tag{21}
$$

$$
\begin{array}{r l} \frac {\Big (l k + 1 + \sum_ {j = l + 1} ^ {m} k _ {j} \Big) !}{(k + 1) \cdot (k !) ^ {l} \cdot \prod_ {j = l + 1} ^ {m} (k _ {j} !)} & = \frac {\Big [ (k + 1) + (l - 1) k + \sum_ {t = l + 1} ^ {m} k _ {t} ] (l k + \sum_ {j = l + 1} ^ {m} k _ {j}) !}{(k + 1) \cdot (k !) ^ {l} \cdot \prod_ {j = l + 1} ^ {m} (k _ {j} !)} \\ & = \frac {1}{l} \cdot \frac {\Big (l k + \sum_ {j = l + 1} ^ {m} k _ {j} \Big) !}{(k !) ^ {l} \cdot \prod_ {j = l + 1} ^ {m} (k _ {j} !)} + (l - 1) \cdot \frac {\Big (l k + \sum_ {j = l + 1} ^ {m} k _ {j} \Big) !}{\frac {k + 1}{k} \cdot (k !) ^ {l} \cdot \prod_ {j = l + 1} ^ {m} (k _ {j} !)} \\ & + \sum_ {t = l + 1} ^ {m} \frac {\Big (l k + \sum_ {j = l + 1} ^ {m} k _ {j} \Big) !}{\frac {k + 1}{k _ {t}} \cdot (k !) ^ {l} \cdot \prod_ {j = l + 1} ^ {m} (k _ {j} !)} + (l - 1) \cdot \frac {1}{l} \cdot \frac {\Big (l k + \sum_ {j = l + 1} ^ {m} k _ {j} \Big) !}{(k !) ^ {l} \cdot \prod_ {j = l + 1} ^ {m} (k _ {j} !)} \end{array}\tag{22}
$$

<sup>Pr(a</sup>1|<sup>P</sup>i<sup>;</sup> <sup>N).</sup> 

Thus far, let us revisit the proof steps. In the first step, we expand the expression $\scriptstyle ( \sum _ { j = 1 } ^ { m } p _ { j } )$ $\mathrm { P r } ( a _ { 1 } | \mathbf { P } _ { i } ; N )$ and divide it into three cases, where case 1 and case 3 are present in $\operatorname* { P r } ( a _ { 1 } | \mathbf { P } _ { i } ; N + 1 )$ whereas case 2 does not. It has been proven that the coefficients of the terms in $\operatorname* { P r } ( a _ { 1 } | \mathbf { P } _ { i } ; N + 1 )$ where case 3 appears as the probability term part, are identical to those in $\begin{array} { r } { ( \sum _ { i = 1 } ^ { m } p _ { j } ) \cdot \mathrm { P r } ( a _ { 1 } | \mathrm { \bf P } _ { i } ; N ) } \end{array}$ Consequently, these terms cancel out in the expression $\begin{array} { r l } { \mathrm { P r } ( a _ { 1 } | \mathbf { P } _ { i } ; N + 1 ) - ( \sum _ { i = 1 } ^ { m } p _ { j } ) \cdot \mathrm { P r } ( a _ { 1 } | \mathbf { P } _ { i } ; N ) } & { { } } \end{array}$ However, case 2 is not present in $\mathrm { P r } ( a _ { 1 } | \mathbf { P } _ { i } ; N +$ 1), which implies that the terms in $\scriptstyle ( \sum _ { j = 1 } ^ { m } p _ { j } )$ $\mathrm { P r } ( a _ { 1 } | \mathbf { P } _ { i } ; N )$ , where case 2 appears as the probability term part, cannot be combined with any terms in $\operatorname* { P r } ( a _ { 1 } | \mathbf { P } _ { i } ; N + 1 )$ by extracting the exponent and performing subtraction on the coefficients like the terms containing case 3. It is fortunate that the terms in $\operatorname* { P r } ( a _ { 1 } | \mathbf { P } _ { i } ; N + 1 )$ , where case 1 appears as the probability term part, cancel out with the corresponding terms in $\begin{array} { r } { ( \sum _ { i = 1 } ^ { m } p _ { j } ) \cdot \mathrm { P r } ( a _ { 1 } | \mathrm { \bf P } _ { i } ; N ) } \end{array}$ which share the same probability terms and the remaining terms have coefficients identical to those of the terms in $\begin{array} { r } { ( \sum _ { i = 1 } ^ { m } p _ { j } ) \cdot \mathrm { P r } ( a _ { 1 } | \mathrm { \bf P } _ { i } ; N ) } \end{array}$ , where case 2 appears as the probability terms. Therefore, these terms can be combined by factoring out the shared coefficients and partial probability terms. The remaining part after factoring out the common factor is $\textstyle \sum _ { s = 2 } ^ { l } ( p _ { 1 } - p _ { s } )$ . It is undeniable that $p _ { 1 } ~ \geq ~ p _ { s }$ when x is an easy question or moderate question for $\mathbf { P } _ { i } ,$ , therefore $\begin{array} { r } { \operatorname* { P r } ( a _ { 1 } | \mathbf { P } _ { i } ; N + 1 ) - ( \sum _ { i = 1 } ^ { m } p _ { j } ) \cdot \operatorname* { P r } ( a _ { 1 } | \mathbf { P } _ { i } ; N ) \ge 0 } \end{array}$ Namely, $\mathrm { P r } ( a _ { 1 } | \mathbf { P } _ { i } ; N )$ is strictly non-decreasing w.r.t. N if x is an easy or moderate question. 

$$
\begin{array} { r l } & { \frac { ( l k + 1 + \sum _ { j = l + 1 } ^ { m } k _ { j } ) ! } { ( k + 1 ) \cdot ( k ! ) ^ { l } \cdot \prod _ { j = l + 1 } ^ { m } ( k _ { j } ! ) } p _ { 1 } ^ { k + 1 } p _ { 2 } ^ { k } \cdots p _ { l } ^ { k } p _ { l + 1 } ^ { k _ { l + 1 } } p _ { l + 2 } ^ { k _ { l + 2 } } \cdots p _ { m } ^ { k _ { m } } } \\ & { - p _ { 1 } \cdot \frac { 1 } { l } \cdot \frac { ( l k + \sum _ { j = l + 1 } ^ { m } k _ { j } ) ! } { ( k ! ) ^ { l } \cdot \prod _ { j = l + 1 } ^ { m } ( k _ { j } ! ) } p _ { 1 } ^ { k } p _ { 2 } ^ { k } \cdots p _ { l } ^ { k } p _ { l + 1 } ^ { k _ { l + 1 } } p _ { l + 2 } ^ { k _ { l + 2 } } \cdots p _ { m } ^ { k _ { m } } } \\ & { - \sum _ { s = 2 } ^ { l } p _ { s } \cdot \frac { ( l k + \sum _ { j = l + 1 } ^ { m } k _ { j } ) ! } { \frac { k + 1 } { k } \cdot ( k ! ) ^ { l } \cdot \prod _ { j = l + 1 } ^ { m } ( k _ { j } ! ) } p _ { 1 } ^ { k + 1 } p _ { 2 } ^ { k } \cdots p _ { s - 1 } ^ { k } p _ { s } ^ { k - 1 } p _ { s + 1 } ^ { k } \cdots p _ { l } ^ { k } p _ { l + 1 } ^ { k _ { l + 1 } } p _ { l + 2 } ^ { k _ { l + 2 } } \cdots p _ { m } ^ { k _ { m } } } \\ & { - \sum _ { t = l + 1 } ^ { m } p _ { t } \cdot \frac { ( l k + \sum _ { j = l + 1 } ^ { m } k _ { j } ) ! } { \frac { k + 1 } { k _ { t } } \cdot ( k ! ) ^ { l } \cdot \prod _ { j = l + 1 } ^ { m } ( k _ { j } ! ) } p _ { 1 } ^ { k + 1 } p _ { 2 } ^ { k } \cdots p _ { l } ^ { k } p _ { l + 1 } ^ { k _ { l + 1 } } p _ { l + 2 } ^{ k _ { l + 2 } } \cdots p _ { t - 1 } ^ { k _ { t - 1 } } p _ { t } ^ { k _ { t - 1 } } p _ { t + 1 } ^ { k _ { t + 1 } } \cdots p _ { m } ^ { k _ { m } } } \\ & { - \sum _ { s = 2 } ^ { l } p _ { s } \cdot \frac { 1 } { l } \cdot \frac { ( l k + \sum _ { j = l + 1 } ^ { m } k _ { j } ) ! } { ( k ! ) ^ { l } \cdot \prod _ { j = l + 1 } ^ { m } ( k _ { j } ! ) } p _ { 1 } ^ { k } p _ { 2 } ^ { k } ⋯ p _ { s - 1 } ^ { k } p _ { s } ^ { k } p _ { s + 1 } ⋯ p _ { l } ^ { k } p _ { l + 1 } ^ { k _ { l + 1 } } p _ { l + 2 } ^ { k _ { l + 2 } } ⋯ p _ { m } ^ { k _ { m } } } \\ & { = \sum _ { s = 2 } ^ { l } \left[ \frac 1l \cdot \frac ( l k + \sum _ { j = l + 1 } ^ { m } k _ { j } ) !}{ ( k ! ) ^ { l } \cdot \prod _ { j = l + 1 } ^ { m } ( k _ { j} ! )} p _ { 1 } ^ { k } p _ { 2 } ^ { k } ⋯ p _ { s - 1 } ^ { k } p _ { s } ^ { k } p _ { s + 1 } ⋯ p _ { l } ^ { k } p _ { l + 1 } ^ { k _ { l + 1} } p _ { l + 2 } ^ { k _ { l + 2} } ⋯ p _ { m } ^ { k _ { m } } \right] \cdot ( p _ { 1 } - p _ { s } ) \geq 0 .  \end{array}\tag{23}
$$

$$
\frac {1}{l} \cdot \frac {\left(l k + 1 + \sum_ {j = l + 1} ^ {m} k _ {j}\right) !}{(k _ {t} + 1) \cdot (k !) ^ {l} \cdot \prod_ {j = l + 1} ^ {m} (k _ {j} !)} p _ {1} ^ {k} p _ {2} ^ {k} \dots p _ {l} ^ {k} p _ {l + 1} ^ {k _ {l + 1}} p _ {l + 2} ^ {k _ {l + 2}} \dots p _ {t - 1} ^ {k _ {t - 1}} p _ {t} ^ {k _ {t + 1}} p _ {t + 1} ^ {k _ {t + 1}} \dots p _ {m} ^ {k _ {m}}\tag{24}
$$

$$
\begin{array}{l} \sum_ {s = 2} ^ {l} p _ {s} \frac {1}{l - 1} \frac {\left(l k + \sum_ {j = l + 1} ^ {m} k _ {j}\right) !}{\frac {k _ {t} + 1}{k} (k !) ^ {l} \prod_ {j = l + 1} ^ {m} (k _ {j} !)} p _ {1} ^ {k} p _ {2} ^ {k}... p _ {s - 1} ^ {k} p _ {s} ^ {k - 1} p _ {s + 1} ^ {k}... p _ {l} ^ {k} p _ {l + 1} ^ {k _ {l + 1}} p _ {l + 2} ^ {k _ {l + 2}}... p _ {t - 1} ^ {k _ {t - 1}} p _ {t} ^ {k _ {t + 1}} p _ {t + 1} ^ {k _ {t + 1}}... p _ {m} ^ {k _ {m}} \\ + \sum_ {r = l + 1, r \neq t} ^ {m} p _ {r} \frac {1}{l} \frac {\left(l k + \sum_ {j = l + 1} ^ {m} k _ {j}\right) !}{\frac {k _ {t} + 1}{k _ {r}} (k !) ^ {l} \prod_ {j = l + 1} ^ {m} (k _ {j} !)} p _ {1} ^ {k} p _ {2} ^ {k}... p _ {l} ^ {k} p _ {l + 1} ^ {k _ {l + 1}} p _ {l + 2} ^ {k _ {l + 2}}... p _ {r - 1} ^ {k _ {r - 1}} p _ {r} ^ {k _ {r} - 1} p _ {r + 1} ^ {k _ {r + 1}}... p _ {t - 1} ^ {k _ {t - 1}} p _ {t} ^ {k _ {t + 1}} p _ {t + 1} ^ {k _ {t + 1}}... p _ {m} ^ {k _ {m}} \\ + p _ {t} \frac {1}{l} \frac {\left(l k + \sum_ {j = l + 1} ^ {m} k _ {j}\right) !}{(k !) ^ {l} \prod_ {j = l + 1} ^ {m} (k _ {j} !)} p _ {1} ^ {k} p _ {2} ^ {k}... p _ {l} ^ {k} p _ {l + 1} ^ {k _ {l + 1}} p _ {l + 2} ^ {k _ {l + 2}}... p _ {m} ^ {k _ {m}} \end{array}\tag{25}
$$

$$
\begin{array}{l} \frac {1}{l} \cdot \frac {\Big (l k + 1 + \sum_ {j = l + 1} ^ {m} k _ {j} \Big) !}{(k _ {t} + 1) \cdot (k !) ^ {l} \cdot \prod_ {j = l + 1} ^ {m} (k _ {j} !)} = \sum_ {s = 2} ^ {l} \frac {1}{l - 1} \cdot \frac {\Big (l k + \sum_ {j = l + 1} ^ {m} k _ {j} \Big) !}{\frac {k _ {t} + 1}{k} \cdot (k !) ^ {l} \cdot \prod_ {j = l + 1} ^ {m} (k _ {j} !)} \\ \qquad + \sum_ {r = l + 1, r \neq t} ^ {m} p _ {r} \cdot \frac {1}{l} \cdot \frac {\Big (l k + \sum_ {j = l + 1} ^ {m} k _ {j} \Big) !}{\frac {k _ {t} + 1}{k _ {r}} \cdot (k !) ^ {l} \cdot \prod_ {j = l + 1} ^ {m} (k _ {j} !)} \\ \qquad + \frac {1}{l} \cdot \frac {\Big (l k + \sum_ {j = l + 1} ^ {m} k _ {j} \Big) !}{(k !) ^ {l} \cdot \prod_ {j = l + 1} ^ {m} (k _ {j} !)} \end{array}\tag{26}
$$

$$
\begin{array}{l} \frac {1}{l} \cdot \frac {\left(l k + 1 + \sum_ {j = l + 1} ^ {m} k _ {j}\right) !}{(k _ {t} + 1) \cdot (k !) ^ {l} \cdot \prod_ {j = l + 1} ^ {m} (k _ {j} !)} p _ {1} ^ {k} p _ {2} ^ {k} \dots p _ {l} ^ {k} p _ {l + 1} ^ {k _ {l + 1}} p _ {l + 2} ^ {k _ {l + 2}} \dots p _ {t - 1} ^ {k _ {t - 1}} p _ {t} ^ {k _ {t + 1}} p _ {t + 1} ^ {k _ {t + 1}} \dots p _ {m} ^ {k _ {m}} \\ - \sum_ {s = 2} ^ {l} p _ {s} \frac {1}{l - 1} \frac {\left(l k + \sum_ {j = l + 1} ^ {m} k _ {j}\right) !}{\frac {k _ {t} + 1}{k} (k !) ^ {l} \prod_ {j = l + 1} ^ {m} (k _ {j} !)} p _ {1} ^ {k} p _ {2} ^ {k}... p _ {s - 1} ^ {k} p _ {s} ^ {k - 1} p _ {s + 1} ^ {k}... p _ {l} ^ {k} p _ {l + 1} ^ {k _ {l + 1}} p _ {l + 2} ^ {k _ {l + 2}}... p _ {t - 1} ^ {k _ {t - 1}} p _ {t} ^ {k _ {t + 1}} p _ {t + 1} ^ {k _ {t + 1}}... p _ {m} ^ {k _ {m}} \\ - \sum_ {r = l + 1, r \neq t} ^ {m} p _ {r} \frac {1}{l} \frac {\left(l k + \sum_ {j = l + 1} ^ {m} k _ {j}\right) !}{\frac {k _ {t + 1}}{k _ {r}} (k !) ^ {l} \prod_ {j = l + 1} ^ {m} (k _ {j} !)} p _ {1} ^ {k} p _ {2} ^ {k}... p _ {l} ^ {k} p _ {l + 1} ^ {k _ {l + 1}} p _ {l + 2} ^ {k _ {l + 2}}... p _ {r - 1} ^ {k _ {r - 1}} p _ {r} ^ {k _ {r - 1}} p _ {r + 1} ^ {k _ {r + 1}}... p _ {t - 1} ^ {k _ {t - 1}} p _ {t} ^ {k _ {t + 1}} p _ {t + 1} ^ {k _ {t + 1}}... p _ {m} ^ {k _ {m}} \\ - p _ {t} \cdot \frac {1}{l} \cdot \frac {\left(l k + \sum_ {j = l + 1} ^ {m} k _ {j}\right) !}{(k !) ^ {l} \cdot \prod_ {j = l + 1} ^ {m} (k _ {j} !)} p _ {1} ^ {k} p _ {2} ^ {k}... p _ {l} ^ {k} p _ {l + 1} ^ {k _ {l + 1}} p _ {l + 2} ^ {k _ {l + 2}}... p _ {m} ^ {k _ {m}} = 0 \end{array}\tag{27}
$$

$$
\begin{array}{l} \frac {1}{l + 1} \cdot \frac {\left(l k + 1 + \sum_ {j = l + 1} ^ {m} k _ {j}\right) !}{(k _ {t} + 1) \cdot (k !) ^ {l} \cdot \prod_ {j = l + 1} ^ {m} (k _ {j} !)} = \sum_ {s = 2} ^ {l} \frac {1}{l} \cdot \frac {\left(l k + \sum_ {j = l + 1} ^ {m} k _ {j}\right) !}{(k !) ^ {l} \cdot \prod_ {j = l + 1} ^ {m} (k _ {j} !)} \\ \qquad + \sum_ {r = l + 1, r \neq t} ^ {m} p _ {r} \cdot \frac {1}{l + 1} \cdot \frac {\left(l k + \sum_ {j = l + 1} ^ {m} k _ {j}\right) !}{\frac {k _ {t} + 1}{k _ {r}} \cdot (k !) ^ {l} \cdot \prod_ {j = l + 1} ^ {m} (k _ {j} !)} \\ \qquad + \frac {1}{l} \cdot \frac {\left(l k + \sum_ {j = l + 1} ^ {m} k _ {j}\right) !}{(k !) ^ {l} \cdot \prod_ {j = l + 1} ^ {m} (k _ {j} !)} \end{array}\tag{28}
$$

For sufficiently large N, the strong law of large numbers implies that $\operatorname* { P r } ( \operatorname* { l i m } _ { N  + \infty } \mathbf { x } _ { j } / N = p _ { j } ) = 1$ When x is an easy question, $p _ { 1 } > p _ { j } , { \bf x } _ { 1 } / N >$ $\mathbf { x } _ { j } / N$ for $j = 2 , 3 , . . . , m$ . As N is sufficiently large, it is sure that $\mathbf { x } _ { 1 }$ is the only maximum value, making the final result must be the correct answer $a _ { 1 }$ . Therefore, if x is an easy question, N, lim $\operatorname* { P r } ( a _ { 1 } | \mathbf { P } _ { i } ; N ) = 1$ . If x is a moderate ques-N + tion, there are equivalent answers in the probability sense, whose probabilities are all the maximum value. Therefore, $\operatorname * { l i m } _ { N  + \infty } \mathrm { P r } ( a _ { 1 } | { \bf P } _ { i } ; N ) =$ $1 / | S |$ . Similarly, if x is a hard question, the maximum probability is not $p _ { 1 }$ , the final result must be a wrong answer, so $\operatorname * { l i m } _ { N  + \infty } \mathrm { P r } ( a _ { 1 } | { \bf P } _ { i } ; N ) = 0$ Theorem 1 to 3 is proved. 

Lemma 1. Consider a specific condition with answer space $| { \mathcal { A } } | \ = \ 3 .$ For $\begin{array} { r l r } { N } & { { } = } & { 3 . } \end{array}$ $\mathrm { P r } ( a _ { 1 } | \mathbf { P } _ { i } ; N ) = 3 p _ { i , 1 } ^ { 2 } - 2 p _ { i , 1 } ^ { 3 } + 2 p _ { i , 1 } p _ { i , 2 } p _ { i , 3 }$ . For $N = 5 , \mathrm { { P r } } ( a _ { 1 } | \mathbf { P } _ { i } ; N ) = 6 p _ { i , 1 } ^ { 5 } - 1 5 p _ { i , 1 } ^ { 4 } + 1 0 p _ { i , 1 } ^ { 3 } +$ $1 5 p _ { i , 1 } ^ { 2 } p _ { i , 2 } p _ { i , 3 } ( p _ { i , 2 } + p _ { i , 3 } )$ 

Proof. For $\begin{array} { r l r } { N } & { { } = } & { 3 . } \end{array}$ , we can calculate 

$\mathrm { P r } ( a _ { 1 } | \mathbf { P } _ { i } ; N )$ with Equation 14 as follows: 

$$
\begin{array}{c} \operatorname * {P r} (a _ {1} | \mathbf {P} _ {i}; N = 3) = \binom {3} {3} p _ {i, 1} ^ {3} + \binom {3} {2} p _ {i, 1} ^ {2} (1 - p _ {i, 1}) \\ \qquad + A (3, 1)   p _ {i, 1} p _ {i, 2} p _ {i, 3} \\ \qquad = 3 p _ {i, 1} ^ {2} - 2 p _ {i, 1} ^ {3} + 2 p _ {i, 1} p _ {i, 2} p _ {i, 3}, \end{array}\tag{29}
$$

where $A ( n , k )$ is the permutation number formula $\begin{array} { r } { A ( n , k ) = \frac { \dot { n } ! } { ( n - k ) ! } } \end{array}$ . For $N = 5$ , we can get 

$$
\begin{array}{l} \operatorname * {P r} (a _ {1} | \mathbf {P} _ {i}; N = 5) = \binom {5} {5} p _ {i, 1} ^ {5} + \binom {5} {4} p _ {i, 1} ^ {4} (1 - p _ {i, 1}) \\ \quad + \binom {5} {3} p _ {i, 1} ^ {3} (1 - p _ {i, 1}) ^ {2} + \binom {5} {2} p _ {i, 1} ^ {2} \binom {3} {2} (p _ {i, 2} ^ {2} p _ {i, 3} + \\ p _ {i, 3} ^ {2} p _ {i, 2}) = 6 p _ {i, 1} ^ {5} - 1 5 p _ {i, 1} ^ {4} + 1 0 p _ {i, 1} ^ {3} + 1 5 p _ {i, 1} ^ {2} p _ {i, 2} p _ {i, 3} \\ (p _ {i, 2} + p _ {i, 3}). \end{array}\tag{30}
$$

Lemma 1 is proved. 

Theorem 4. For two prompting strategies $\mathbf { P } _ { i }$ and $\mathbf { P } _ { i ^ { \prime } }$ , note $p _ { i , q } = \operatorname* { m a x } \{ p _ { i , 2 } , \ldots , p _ { i , m } \} , p _ { i ^ { \prime } , q ^ { \prime } } =$ max $\{ p _ { i ^ { \prime } , 2 } , \ldots , p _ { i ^ { \prime } , m } \} , { \mathrm { i f } } p _ { i , 1 } - p _ { i , q } < p _ { i ^ { \prime } , 1 } - p _ { i ^ { \prime } , q ^ { \prime } }$ and $p _ { i , 1 } + p _ { i , q } - p _ { i , 1 } ^ { 2 } - p _ { i , q } ^ { 2 } > p _ { i ^ { \prime } , 1 } + p _ { i ^ { \prime } , q ^ { \prime } } - p _ { i ^ { \prime } , 1 } ^ { 2 } -$ $p _ { i ^ { \prime } , q ^ { \prime } } ^ { 2 }$ , there exits a sufficiently large $N _ { 0 }$ such that for $\mathbf { \bar { \chi } } N > N _ { 0 } , \operatorname* { P r } ( a _ { 1 } | \mathbf { P } _ { i } ; N ) < \operatorname* { P r } ( a _ { 1 } | \mathbf { P } _ { i ^ { \prime } } ; N )$ 

Proof. According to Khinchin’s Law of Large Numbers and Lindeberg-Levy Central Limit Theorem, when N is sufficiently large, each $X _ { i }$ can be approximated by a normal distribution. Specifically, for each $\mathbf { x } _ { i , j }$ we have $\begin{array} { r c l } { \mathbf x _ { i , j } } & { \sim } & { N ( N p _ { i , j } , N p _ { i , j } ( 1 \ - \ p _ { i , j } ) ) } \end{array}$ Note $\begin{array} { r c l } { M _ { i } } & { = } & { \operatorname* { m a x } ( \mathbf { x } _ { i , 2 } , . . . , \mathbf { x } _ { i , m } ) } \end{array}$ and $\begin{array} { r l } { p _ { i , q } } & { { } = } \end{array}$ max $\{ p _ { i , 2 } , \hdots , p _ { i , m } \}$ , the distribution of $M _ { i }$ can be approximated by $M _ { i } \sim \mathcal { N } ( N p _ { i , n } , N p _ { i , 1 } ( 1 -$ $p _ { i , 1 } ) )$ . So $\mathbf { x } _ { i , j } - M _ { i }$ obey the normal distribution $\mathcal { N } ( N ( p _ { i , 1 } - p _ { i , q } ) , N ( p _ { i , 1 } ( 1 - p _ { i , 1 } ) + p _ { i , q } ( 1 - p _ { i , q } ) ) )$ Thus, we can get Equation 31: 

$$
\begin{array}{l} \operatorname * {P r} (\mathbf {x} _ {i, 1} > M _ {i}) \\ = \operatorname * {P r} (\mathbf {x} _ {i, 1} - M _ {i} > 0) \\ = 1 - \Phi (f (p _ {i, 1}, p _ {i, q}, N)), \\ \Phi (f (p _ {i, 1}, p _ {i, q}, N)) = \\ \Phi (\sqrt {\frac {N}{p _ {i , 1} (1 - p _ {i , 1}) + p _ {i , q} (1 - p _ {i , q})}} (p _ {i, q} - p _ {i, 1})), \end{array}\tag{31}
$$

where Φ is the standard normal cumulative distribution function. This also holds for any other $\begin{array} { r } { \begin{array} { c c c c } { \operatorname { P r } ( \mathbf { x } _ { i ^ { \prime } , 1 } } & { > } & { M _ { i } ^ { \prime } ) . } & { \operatorname { I f } p _ { i , 1 } - p _ { i , q } } & { < } \end{array} } \end{array}$ $p _ { i ^ { \prime } , 1 } \ - \ p _ { i ^ { \prime } , q ^ { \prime } }$ and $p _ { i , 1 } + p _ { i , q } - p _ { i , 1 } ^ { 2 } - p _ { i , q } ^ { 2 } >$ $p _ { i ^ { \prime } , 1 } + p _ { i ^ { \prime } , q ^ { \prime } } - p _ { i ^ { \prime } , 1 } ^ { 2 } - p _ { i ^ { \prime } , q ^ { \prime } } ^ { 2 } .$ , we can get $p _ { i , q } \ -$ $p _ { i , 1 } > p _ { i ^ { \prime } , q } - p _ { i ^ { \prime } , 1 }$ and $p _ { i , 1 } \big ( 1 - p _ { i , 1 } \big ) + p _ { i , q } ( 1 -$ $p _ { i , q } ) \ < \ p _ { i ^ { \prime } , 1 } ( 1 - p _ { i ^ { \prime } , 1 } ) + p _ { i ^ { \prime } , q ^ { \prime } } ( 1 - p _ { i ^ { \prime } , q ^ { \prime } } )$ , and $\Phi ( f ( p _ { i , 1 } , p _ { i , q } , N ) ) ~ > ~ \Phi ( f ( p _ { i ^ { \prime } , 1 } , p _ { i ^ { \prime } , q ^ { \prime } } , N ) )$ . So there exists a large $N _ { 0 }$ such that for $N \_ >$ $N _ { 0 } , ~ \mathrm { P r } ( \mathbf { x } _ { i ^ { \prime } , 1 } ~ > ~ M _ { i } ^ { \prime } ) ~ > ~ \mathrm { P r } ( \mathbf { x } _ { i , 1 } ~ > ~ M _ { i } )$ $i . e . ,$ $\mathrm { P r } ( a _ { 1 } | { \bf P } _ { i } ; N ) ~ > ~ \mathrm { P r } ( a _ { 1 } | { \bf P } _ { i ^ { \prime } } ; N )$ . Theorem 4 is proved. 

## C Detailed Results

Here we display the scaling performance of different prompting strategies on each LLM and benchmark under given sampling time N and cost $O _ { \mathrm { { r } } }$ as shown in Figures 8 to 15. We find that, aside from CoT, DiP also exhibits superior performance compared to other complex prompting strategies on certain models and datasets, $e . g .$ ., GPT-4o-mini on MATH. This also comes from the two reasons, i.e., DiP has more hard questions and easy questions, and a flat probability distribution of wrong answers on the specific dataset. This phenomenon is particularly prominent on powerful LLMs such as Gemini-1.5-Flash on GSM8K and GSM-Hard, where DiP and CoT exhibit comparable performance. Almost 83% of results satisfy that CoT or DiP performs best as significantly scaling. Besides, this trend is also observed on other prompting strategies on few datasets and models. This encourages us to fully unleash the potential of simple prompting strategies, and indicates that the scaling performance does not only depend on the prompting strategies pass@1 accuracy. 

## D More Discussions on Improving the Scaling Performance

In this section, we will discuss more about our further exploration of the two ways to improve the scaling performance. We display more results of (1) adaptively scaling based on the question difficulty, (2) dynamically choosing the optimal $\mathbf { P } _ { i }$ and (3) combining adaptively scaling and dynamically choosing the optimal $\mathbf { P } _ { i }$ in Section D.1, D.2 and D.3, respectively. 

## D.1 Adaptively Scaling Based on the Difficulty

We use the following prompt to force the LLM to determine if the question is hard for given $\mathbf { P } _ { i }$ 

Question: 

{question} 

Using the method #{method}# to solve the question: 

{description} 

If the method is more likely to get the right answer, the question is easy. Otherwise, ifthe method is more likely to get the wrong answer, the question is hard. Please determine the difficulty ofthe question for the used method, and answer in the following JSONformat. 

{"Difficulty": "Easy or Hard", "Reason": ""} 

Figures 16 to 20 report the results of each prompting strategy when adaptively scaling based on the question difficulty. Our experiment results show that LLMs cannot accurately judge the difficulty of the input question most of the time, thus even leading to reduced performance. Nevertheless, this method is theoretically capable of enhancing the scaling performance, thereby motivating us to explore other approaches to accurately assess the question difficulty. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/9e8aac371feb4a03fa21780ab1182cebb471fc19597adca68cb6b86ccd25b205.jpg)



Figure 8: Performance of each prompting strategy under given sampling time N on GSM8K.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/49a5a84fbae880c7ac42338f01cb2b13066e226ca7c1d9788e56b9d42abcba38.jpg)



Figure 9: Performance of each prompting strategy under given cost O on GSM8K.


## D.2 Dynamically Choosing the Optimal $\mathbf { P } _ { i }$

Figures 21 to 25 display the results on GSM8K on LLaMA-3-8B-Instruct, GLM-4-9B-Chat, Phi-3.5- mini-Instruct, Gemini-1.5-Flash and GPT-4o-mini, respectively. It can be observed that all LLMs tend to believe that CoT is the best prompting strategy, while CoT does not excel at every question. With oracles to provide the optimal $\mathbf { P } _ { i }$ labels, all LLMs demonstrate significant performance improvements, even with only one sampling time, proving the enormous potential of this method. we will explore how to approach this upper bound in the future. 

## D.3 Combining Adaptively Scaling and Dynamically Choosing the Optimal $\mathbf { P } _ { i }$

Figures 26 to 30 display the results of combining adaptively scaling and dynamically choosing the optimal $\mathbf { P } _ { i }$ on GSM8K on LLaMA-3-8B-Instruct, GLM-4-9B-Chat, Phi-3.5-mini-Instruct, Gemini-1.5-Flash, and GPT-4o-mini, respectively. Figures 31 to 36 show the results on each LLM on MATH, respectively. Extensive experiments demonstrate the general effectiveness and superiority of this method, which has an extremely high upper bound. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/f6fb02fb4c50f36b764f23d08efa6b158488a6bad9789902c9cd1817e15c12a6.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/8d830c6fd45fa5f6c164d706a5824a634c093fe9667126f0221673f1d4cee9e9.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/7619db2ef4abd2b12e94232abf7869e0f3fad506404e75ae889707c465ca60cb.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/e2bbd60a857513903efb641b429dbe12d863e058feb303a396d870c082b3b4bf.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/7733fec0a9ce3f3c497706dfe125c72c3524faae962560a82e86996a7566e596.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/bb91c0e10d46a1392fa0a555514bebfeaa5a208e6016d6c2eafa8b72b2c8e11a.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/4432781a72d83ded4c0183093de965ebd8a37dd5e31054de6ead39dc7600f3ef.jpg)



Figure 10: Performance of each prompting strategy under given sampling time N on GSM-Hard.


Non-Iterative $\mathcal { P } _ { 1 }$ DiP CoT L2M SBP AnP Iterative P2 ToT O S-RF MAD 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/04d60a391955c0ad5bc80426806bf7d85309494fdb2111a7a7d8d3600b677d43.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/89ac47cc9172ad8a690f96c845bb376369a781283032b104b1f56ac633570151.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/1bee8d9352a42f9fbfa3493262051cfc8463cc3043a7d81cb68772a2e12932fa.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/93ecef9754028d49354cd860492e1fa8c7ce0f51439795e7bae5aaced3860c41.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/a56db65de17fb140fa0107e9d8cdd9ef3dac07de2b76a6baf5dff38cbf8a32c4.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/043798fc7484a29ca4eccf093dbe2747382d5481aef1ee4bf93df632a84e81c9.jpg)



Figure 11: Performance of each prompting strategy under given cost O on GSM-Hard.



Figure 12: Performance of each prompting strategy under given sampling time N on MATH.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/778d6ee25900bb815836112af68e8017ddecd813f7a052a5dd7a62ef0be7d256.jpg)



Figure 13: Performance of each prompting strategy under given cost O on MATH.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/669808917873288373a05638f467e406ce953d8474add9bf60049c95478b0464.jpg)



Figure 14: Performance of each prompting strategy under given sampling time N on MMLU.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/40f73193074de2a605f50efe9dcf12b6b42e57c06c95d94914090203866b278b.jpg)



Figure 15: Performance of each prompting strategy under given cost O on MMLU.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/896c84cf71bd2a8b823ef91e26447f91e96bda7613fadc0630ecd9fb855648e1.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/3d8ef4060476999cb4b74748d17f4e389086fbaf939f7e2b7847266bc47add77.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/0b7d7ec2ddf20d23fc97a51ee1ea28ceee708e03c0cb36862b532c95b027d20d.jpg)



Figure 16: Results of adaptively scaling based on the question difficulty on Llama-3-8B-Instruct on GSM8K.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/908a460eb8112c890c668dfe9bbe4aa8370ab251ba06a49f0ed1e6a7b3e77d9d.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/8e8074c8c2d45ccc1d278c9ec8ce5088797dbd64112a34867be6b121a949d8cf.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/e2831a10540aaa941e1fc5597d00c1aaab0d38e2071c4016f9856b0a90499a35.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/f8fa08f7487e512fffb9c6a3dee36b703df75210bd0507bc68dff446f377ef5c.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/860d59ad3f39ebe22e5a69af39641271109212ace94b4352863b0428261966e5.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/66e0796c35bcca5abc6b669f5f8e06367a69b031175f48ee651d22dddbf6a9a7.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/383bd03817b89497b8306f39042e3a2dd1e08e36cc9f75471a66fd888bc167ae.jpg)



Figure 17: Results of adaptively scaling based on the question difficulty on GLM-4-9B-Chat on GSM8K.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/e4c5a128ee41354d332edfbd2d5f1be405886ab6b6ac599d846c7744e6689cbe.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/f8858057bf921b0ed5fce2042c438b48055610b30aa0b6c612e569334512c5f1.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/07302bdb98b915c63d689911d9fa67e0f2126ae867be8df2816bf8521a6f43a1.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/3e6c751cc33e61193244ad0d180c5bca2c601981d6d0aaa12d57117c735288c1.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/1a75021beff3958f3e6c933a99bc22fc5bd435e52e02b91a098918a7c9382194.jpg)



Figure 18: Results of adaptively scaling based on the question difficulty on Phi-3.5-mini-Instruct on GSM8K.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/bb8c12f67c8dda901ad405b71a54fc3502076cc4bca610c797e99a1dd9f16ace.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/711f8c72f96d24f41ae5de8789706f225fdbf96def49d06bb343fc38e4263a84.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/efb86cc1331ab06d3bb52f5d30b411736cce4b62eb1d51dc89a8667adf0e50a9.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/32b0a1f8386e55d73638d6f28653567c8839cd30981f30cf0f05783788b3bd03.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/ccbc2cd0c9266f534fa9e74ea6eb00abc06307875d3fc64b26a1659f4c6fca8f.jpg)



Figure 19: Results of adaptively scaling based on the question difficulty on Gemini-1.5-Flash on GSM8K.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/8d88054be803fba7564a15522623e2c617212f7a504d7f4f685bea5043730fcf.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/c49821c89f9b48d375c6538288efa1f9f19b992c19292440c70186868ea790eb.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/4e41167268fd7d92f81ac1bd4c42a9423fa991f2322568de1ee968220b6ae872.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/93a751e78ed21faf0d47d3babfeb4806b74375266861835f341abb443a51f46c.jpg)



Figure 20: Results of adaptively scaling based on the question difficulty on GPT-4o-mini on GSM8K.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/a8d89284abe4f3b21fc01fc279e6a0df8a8a2847926213200304f1f5e33c7300.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/9d316bb85a4bbff5938abc6dd3eee1b5a5d990eefc30743d8e30f9ab3711100b.jpg)



Figure 21: Results of dynamically choosing the optimal $\mathbf { P } _ { i }$ on LLaMA-3-8B-Instruct on GSM8K.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/a9ffcba06695f235460ab9ecec708e1a51cc365bcdf6bd98236341091639c2e8.jpg)



Figure 22: Results of dynamically choosing the optimal $\mathbf { P } _ { i }$ on GLM-9B-Chat on GSM8K.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/6826c246b4b835dd98c56939a0975b53109a2d987f80222428aa2f45e81f9267.jpg)



Figure 23: Results of dynamically choosing the optimal $\mathbf { P } _ { i }$ on Phi-3.5-mini-Instruct on GSM8K.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/d8b4f02e3fc90660b364aa48b75dbe6cfae4eb8596cbb990df0ec92b9eb74a2d.jpg)



Figure 24: Results of dynamically choosing the optimal $\mathbf { P } _ { i }$ on Gemini-1.5-Flash on GSM8K.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/b2d03ef0adb9c0947fc4bc0e05916ff0c9471ef1f51d2a2447451ff2d34570a9.jpg)



Figure 25: Results of dynamically choosing the optimal $\mathbf { P } _ { i }$ on GPT-4o-mini on GSM8K.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/c4c4d9e73244e9cb3598bbc4fbcd189f12f1971a56779bf297a168c23abcafb6.jpg)



Figure 26: Results of combining adaptively scaling and dynamically choosing the optimal $\mathbf { P } _ { i }$ on LLaMA-3-8B-Instruct on GSM8K.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/61a9f1360d3c740cf560aca0b52bd6c16eed6f700a179c2941e63e60624cd1c7.jpg)



Figure 27: Results of combining adaptively scaling and dynamically choosing the optimal $\mathbf { P } _ { i }$ on GLM4-9B-Chat on GSM8K.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/5dca367467108591ff006338763071df71efda6a0ffc63c597f7fa825137261b.jpg)



Figure 28: Results of combining adaptively scaling and dynamically choosing the optimal $\mathbf { P } _ { i }$ on Phi-3.5-mini-Instruct on GSM8K.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/5f19b5c64853f5ba7c9f6f413ec25d231ff2477745d10b5044ad081d1931a5d5.jpg)



Figure 29: Results of combining adaptively scaling and dynamically choosing the optimal $\mathbf { P } _ { i }$ on Gemini-1.5- Flash on GSM8K.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/c0e862e9dd182878531d7cee4446fc2ede492829a377abe0ec08a25328fcb5cb.jpg)



Figure 30: Results of combining adaptively scaling and dynamically choosing the optimal $\mathbf { P } _ { i }$ on GPT-4o-mini on GSM8K.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/37b93a2e906ca264346c0f569e40441e38e9368090588d3e6112f4e8ec2bb8c6.jpg)



Figure 31: Results of combining adaptively scaling and dynamically choosing the optimal $\mathbf { P } _ { i }$ on Qwen2.5-7B-Instruct on MATH.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/62087b538c9ab7d638dec4a734b9009dbbc449d0c6b3a757bec57c481cada2cb.jpg)



Figure 32: Results of combining adaptively scaling and dynamically choosing the optimal $\mathbf { P } _ { i }$ on LLaMA-3-8B-Instruct on MATH.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/b5e52ffb862dda2039ab3d196ed587e84d7d1e3a2b94a271d376dc47ca9f5304.jpg)



Figure 33: Results of combining adaptively scaling and dynamically choosing the optimal $\mathbf { P } _ { i }$ on GLM-4-9B-Instruct on MATH.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/72cd84e4fb0077bf3f482c07b0a2f7c5c315d7dd47a2ceae88779ab6b9ebdd2f.jpg)



Figure 34: Results of combining adaptively scaling and dynamically choosing the optimal $\mathbf { P } _ { i }$ on Phi-3.5-mini-Instruct on MATH.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/3238dc8145172e9619626fe2e9e4f54e777ac71922287789ce899b8e6b08644c.jpg)



Figure 35: Results of combining adaptively scaling and dynamically choosing the optimal $\mathbf { P } _ { i }$ on Gemini-1.5- Flash on MATH.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/ebdfe4766b7b0917b521dc13b55d46213b5c6aee24de091c24fc654d122f2ed2.jpg)



Figure 36: Results of combining adaptively scaling and dynamically choosing the optimal $\mathbf { P } _ { i }$ on GPT-4o-mini on MATH.


## E Extended Experiments

We extend experiments with Qwen2.5-7B-Instruct on two more challenging reasoning benchmarks, GPQA (Rein et al., 2024) and AIME2024 (Mathematical Association of America, 2024), further verifying the generality of our findings and methods. Figure 37 displays the accuracy of each $\mathbf { P } _ { i }$ with Qwen2.5-7B-Instruct on GQPA and AIME. Simple CoT and DiP gradually dominate as scaling sampling times on GPQA and AIME, respectively. On GPQA, we can see that the performance fluctuates as scaling, rather than constantly increasing. This can be attributed to the ratio of easy/hard questions and their accuracies in the entire dataset. Given AnP as an example, Figure 39 reports its scaling performance on the easy/hard question subsets of GPQA. Performance on easy questions monotonically improves with increasing sampling times, while accuracy on hard questions exhibits a corresponding decline. This fundamental trade-off induces characteristic oscillation in the aggregate accuracy, with consistent replication across all tested prompting strategies, substantiating our theoretical framework. 

Figure 38 shows the results of adaptively scaling on each $\mathbf { P } _ { i }$ . “Adaptive” approach demonstrates evident performance gains over “Vanilla” baselines across CoT, L2M, SBP, and AnP, indicating the model’s intrinsic capability to assess question difficulty. Under “Oracle” conditions, it can achieve further performance amplification. 

Figure 40 reports the results of dynamically choosing the optimal $\mathbf { P } _ { i }$ on Qwen2.5-7B-Instruct on GPQA. “Dynamic” approach achieves median range performance across all tested $\mathbf { P } _ { i }$ , quantitatively confirming the model’s suboptimal strategy selection capacity. Strikingly, “Oracle” intervention enables dramatic performance elevation, with 65.4% accuracy at $N = 1$ 

Figure 41 summarizes the results of combining adaptively scaling and dynamically choosing the optimal $\mathbf { P } _ { i }$ on Qwen2.5-7B-Instrcut on GPQA, which further enormously enhances the scaling performance, with 75.7% accuracy at $N = 1 0 0$ 

## F Implementation Details and Prompts

We use vllm (Kwon et al., 2023) to deploy opensourced LLMs, with top-p = 0.9 and temperature = 0.7. For closed-sourced LLMs, we use their APIs with default settings. We set the content safety detection threshold of Gemini-1.5-Flash to zero to prevent erroneous judgments that may result in null outputs. 

Following (Wang et al., 2024; Lightman et al., 2024; Qi et al., 2025), we use MATH-500, a subset of representative problems from the MATH dataset to speed up the evaluation. We use the test split of each dataset. The license for all datasets is CC-BY 4.0 or others for open academic research. The number of samples on each dataset is shown in Table 5. We ensure our use of existing artifacts is aligned with their intended purposes. All of them are public English datasets for academic research. On GSM8K and GSM-Hard, we use the same 1-shot prompt in the original paper of Leastto-Most (Zhou et al., 2023) shown in Figure 44 and Figure 45. On other datasets, we use the 0- shot prompt shown in Figure 46. We use the same prompt in Analogous Prompting (Yasunaga et al., 2024), and guide the LLM to recall one analogous problem. We use the same 1-shot prompt in Step-Back Prompting (Zheng et al., 2024) on MMLU, and apply their prompt designed for reasoning tasks on other datasets. We use the same prompt in 0- shot Chain-of-Thought (Kojima et al., 2022), Multi-Agent Debate (Du et al., 2024) and Self-Refine (Huang et al., 2024) on all datasets. The prompts are shown in Figures 42 to 51. 


Table 5: The number of samples in each dataset.


<table><tr><td>Dataset</td><td>Samples</td></tr><tr><td>GSM8K</td><td>1318</td></tr><tr><td>GSM-Hard</td><td>1318</td></tr><tr><td>MATH-500</td><td>500</td></tr><tr><td>MMLU-Biology</td><td>310</td></tr><tr><td>MMLU-Chemistry</td><td>203</td></tr><tr><td>MMLU-Physics</td><td>151</td></tr></table>

GPQA 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/274149b47778a37ce112cb12f84b99fdf8d62e1229722e7c33ef0a3353fec7f4.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/4cff5757788fcf7528386f8e37cb36a4f803a63df7a30c06b2fb2daac409905c.jpg)



Figure 37: Accuracy of each P with Qwen2.5-7B-Instruct on GQPA and AIME2024.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/81c5a8dd15f23d672db94188d0482ccd5fb226e52943a8ea1e0ec9985898ea83.jpg)



Figure 38: Results of adaptively scaling based on the question difficulty on Qwen2.5-7B-Instruct on GPQA.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/98e0ae3d3de90b42d68093f0d0324b1a44f17c8553c3b5ee8ef548469095c4b8.jpg)



Figure 39: Scaling performance of AnP on the easy/hard question subsets of GPQA. The accuracy on easy questions is non-decreasing with the sampling time, while it exhibits a general declining trend on hard questions. This also holds for all other prompting strategies.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/6a16bfa7fe2d8c8c526eb72f825379237df826d4130200c89d5e14e9af5b8b52.jpg)



Figure 40: Results of dynamically choosing the optimal P on Qwen2.5-7B-Instruct on GPQA.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/40828bf0899e57ca6f95f8a239bdc3837dc365ec7854f7752600e830561611cc.jpg)



Figure 41: Results of combining adaptively scaling and dynamically choosing the optimal P on Qwen2.5-7B-Instrcut on GPQA.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/a051c007455d2369c80601c0880c4e9a2ac0d3a836fcc1056a88bd67232cae0c.jpg)



Figure 42: Prompt of DiP.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/2da4744dad2b50f8d183c15f6c10b81a5986446232576df7b097881bf013372c.jpg)



Figure 43: Prompt of CoT.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/aaadde049f81023ed6d330d39c8b1bc3de9422169d74a3cbac01a4c9b1883290.jpg)



Figure 44: Prompt of L2M on GSM8K.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/b44fc1f96592e8846db73845ba6220a089798605503dc3a9ee9f7cc28f0bb2cd.jpg)



Figure 45: Prompt of L2M on GSM-Hard.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/aa43224ad7d1bb6668924f553cc801eb7c2238e4196b7ba2b752df5ea90033c3.jpg)



Figure 46: Prompt of L2M on MATH and MMLU.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/b85a74f851dbaccf09e3ba02be065c01688b865c8d3c88333bdf58196222bb5d.jpg)



Figure 47: Prompt of ToT.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/cb6aba8139e722206d5b0e4a4087b9aff006f32425949dad02d00e294173a21e.jpg)



Figure 48: Prompt of S-RF.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/ec801ecd96147977a8a76cd60eb99e2438e3bd08456d55ed3f32a05d48f7da41.jpg)



Figure 49: Prompt of SBP.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/3e42b8b66fc9c336198394f47bc7aa49b4af41e667599531903c53b63e983dd9.jpg)



Figure 50: Prompt of AnP.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/36e5bde8-1a14-4d9f-a434-6f4711225faf/d68ad589409c5bee42865a25ae8ed5fcae572d06648255c571c34a5fc56fa13a.jpg)



Figure 51: Prompt of MAD.
