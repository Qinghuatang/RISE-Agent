# KS-Lottery: Finding Certified Lottery Tickets for Multilingual Transfer in Large Language Models

Fei Yuan<sup>1</sup>, Chang Ma<sup>2</sup>, Shuai Yuan<sup>1</sup>, Qiushi Sun<sup>2</sup>, Lei Li<sup>3</sup> 

<sup>1</sup> Shanghai Artificial Intelligence Laboratory <sup>2</sup> The University of Hong Kong, <sup>3</sup> Carnegie Mellon University yuanfei@pjlab.org.cn, cma@cs.hku.hk 

syuanaf@connect.ust.hk, qiushisun@u.nus.edu, leili@cs.cmu.edu 

## Abstract

The lottery ticket hypothesis posits the existence of “winning tickets” within a randomly initialized neural network. Do winning tickets exist for LLMs in fine-tuning scenarios? How can we find such winning tickets? In this paper, we propose KS-Lottery, a method to identify a small subset of LLM parameters highly effective in multilingual fine-tuning. Our key idea is to use Kolmogorov-Smirnov Test to analyze the distribution shift of parameters before and after fine-tuning. We further theoretically prove that KS-Lottery can find the certified winning tickets in the embedding layer, fine tuning on the found parameters is guaranteed to perform as well as full fine-tuning. Comparing KS-Lottery with other tuning algorithms on translation tasks, the experimental results show that KS-Lottery finds a much smaller set of parameters for fine-tuning while achieving the comparable performance as full fine-tuning LLM. Surprisingly, we find that fine-tuning 18 tokens’ embedding of LLaMA suffices to reach the fine-tuning translation performance <sup>1</sup>. 

## 1 Introduction

Can we find an ultra-small subset of a well-trained Large Language Model (LLM; Touvron et al., 2023a,b; OpenAI, 2023; Chowdhery et al., 2022) such that fine-tuning these few parameters suffices to achieve the same performance as full tuning? The lottery tickets hypothesis (Frankle and Carbin, 2019) states that a small subnetwork (less than 10- 20% of the whole model size), referred to as “winning tickets”, in a large, randomly initialized neural network can achieve comparable performance to the original network with the same amount of training. Yet, the existence of winning tickets is not investigated for fine-tuning scenarios. Prior work (Aghajanyan et al., 2021) presents evidence that there are a small number of additional parameters corresponding to an intrinsic dimension (Li et al., 2018) on which fine-tuning leads to good performance. However, it remains an unsolved challenge to uncover such a small subset of fine-tuning efficient parameters within the original model. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/622691a6-a1f2-40b5-bee7-743c94c4d403/68bc4853acbe5999c231ec44336527557ff4a7f4a188fbd11b4bb54e6b53ef99.jpg)



Figure 1: Existence of “winning tickets”. KS-Lottery identifies a small subset of embedding parameters of LLaMA-7B to maintain the translation performance of en→ca on Flores-101.


In this paper, we show that there exist key parameters - winning tickets, for transferring LLM to multiple new languages. As shown in Figure 1 (KS-Lottery), we found that as tuning as few as 18 (18/32000 = 0.0006) token embeddings of a well-trained LLM could achieve test performance comparable to full tuning on machine translation tasks. Based on the observation, we state the lottery ticket hypothesis for multilingual fine-tuning. 

Generally, the fine-tuning process can be represented by a transition in M parameters from $\pmb { \theta } ~ = ~ [ \theta _ { 0 } , \widetilde { \theta } _ { 1 } , \cdot \cdot \cdot , \theta _ { M } ] \mathrm { { \ t o } } \widetilde { \pmb { \theta } } ~ = ~ [ \widetilde { \theta } _ { 0 } , \widetilde { \theta } _ { 1 } , \cdot \cdot \cdot , \widetilde { \theta } _ { M } ] .$ where θ and θe denote the sets of parameters that characterize the LLM f(·, θ) before and after the fine-tuning process, respectively. Typically, the entire set of parameters θ is adjusted during finetuning to enhance the model’s ability to represent and learn the new tasks more effectively. 

The Fine-Tuning Lottery Ticket Hypothesis. nA pre-trained neural network contains a small subset of parameters $\begin{array} { r l r } { ( { \widetilde { \pmb { \theta } } } ^ { D } } & { { } = } & { [ { \widetilde { \theta } } _ { 0 } , { \widetilde { \theta } } _ { 1 } } \end{array}$ $\dots , \widetilde { \theta } _ { D } , \theta _ { D + 1 } , \dots , \theta _ { M } ]$ , where $D ~ \ll ~ M )$ that is initialized such that—when fine-tuned in isolation—it can match the performance of full tuning. 

In this work, we examine the fine-tuning lottery ticket hypothesis on LLMs in a multilingual transfer scenario, and explore the following inquiries: 

• Existence of Winning Tickets: Is it certain that every LLM in multilingual transfer encompasses a compact subset of winning tickets? And how to quickly identify the winning tickets? 

• Efficiency of Winning Tickets: How minimal can this subset be in terms of size? 

• Interpretability of Winning Tickets: Do these winning tickets reflect the unique architectural characteristics of the multilingual LLM? 

Identifying Winning Tickets. We propose KS-Lottery, a method to identify a winning ticket by merely fine-tuning the embedding layer of an LLM. And then fine-tune these identified tickets, keeping the remaining parameters frozen. The whole KS-Lottery consists of three steps: 

1. Fine-tuning the embedding layer of an LLM $f ( \cdot , \pmb \theta )$ to obtain $f ( \cdot , \widetilde { \pmb { \theta } } ^ { D } )$ 

2. Run Kolmogorov-Smirnov Test between θ and $\widetilde { \pmb { \theta } } ^ { D }$ to select the winning ticket $ { { \widetilde { \theta } } } _ { a } ^ { D }$ . 

3. Tuning the $ { { \widetilde { \theta } } } _ { a } ^ { D }$ within θ on downstream tasks. 

The core idea of our method is to heuristically identify parameters with large distribution changes before and after fine-tuning. Here we use Kolmogorov-Smirnov Test to determine whether two sample variables stem from the same underlying distribution. Also, we simplify and accelerate the Kolmogorov-Smirnov Test process by focusing on the embedding layer, which constitutes the major change in parameters, due to the inductive bias of multilingual tasks. To this end, our approach, KS-Lottery, is a surprisingly straightforward yet effective technique for pinpointing winning tickets in LLMs. The Kolmogorov-Smirnov Test has the advantage of not assume the distribution, which is particularly useful when the data dows not conform to a normal distribution. Furthermore, a theoretical framework is developed to certify the effectiveness of our method. Inspired by randomized smoothing techniques (Zhao et al., 2021, 2022), we illustrate that parameters with Kolmogorov-Smirnov distance bounded by a small value before and after fine-tuning do not impact prediction. This provides a way for giving a certified lower bound for the performance of partial tuning on winning tickets. Our analysis also proves that KS-Lottery can find a small set of winning tickets when the original prediction model shows little uncertainty. This gives us a strong foundation for asserting that KS-Lottery can be an effective tool for finding winning tickets. 

• We propose KS-Lottery, a method to identify winning tickets – an ultra-small subset of parameters that are sufficient to fine-tune on to achieve that of Full Tuning. 

• Theoretically, we prove that KS-Lottery finds certified winning tickets. 

• Empirically, we demonstrate that fine-tuning as few as 18 identified winning tickets (the token embedding) of LLaMA-7B using en→ca data achieves surprisingly good performance in translation tasks. This will result in a new standard in multilingual transfer of LLMs. 

## 2 Related Work

Lottey Tickets Hypothesis. The Lottery Tickets Hypothesis suggests the presence of ‘winning tickets’ or beneficial subnetworks within a model(Frankle and Carbin, 2018; Malach et al., 2020). These subnetworks, discovered during pruning, are believed to be specifically suited to the learning task (Frankle and Carbin, 2018). In relation to this, Zheng et al. (2022) fins that these ‘winning tickets’ are more sparse in the later layers of the BERT model on GLUE tasks (Wang et al., 2018). While much research focuses on model pruning, there’s also work on efficient parameter tuning (Ding et al., 2023). For example, it’s shown that the performance of fine-tuned parameters can indicate both the task’s inductive biases and the model’s inherent structure (Ding et al., 2023). 

Certified Methods for Transfer Learning. Certification is crucial in transfer learning, aiming to measure a model’s generalization and capabilities (Raghunathan et al., 2018; Jia et al., 2019). Research has introduced certified robustness accuracy as a defense against adversarial attack (Raghunathan et al., 2018; Jia et al., 2019; Muravev and Petiushko, 2021; Zhao et al., 2022; Lecuyer et al., 2019). Randomized smoothing, a modelindependent certification technique, assesses how input changes affect predictions (Lecuyer et al., 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/622691a6-a1f2-40b5-bee7-743c94c4d403/d5ebb2d28ee68b987d3666dd56832e36827a151bc4390cc5072c604a6415103c.jpg)



(a) The embedding distribu tion of token $\ddot { \mathbf { \Omega } } \mathbf { a } ^ { \mathbf { \eta } } .$


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/622691a6-a1f2-40b5-bee7-743c94c4d403/6cd851b3111b7a2c99ba05be9ee1f283c2107b92e8efd8117f70f86056cecefe.jpg)



(b) The embedding distribution of token “Do”.



Figure 2: Not all token embeddings change during finetuning. For example, the embedding for “Do” shows only slight changes before and after tuning, while the embedding for “_a” changes significantly.


2019; Muravev and Petiushko, 2021). Our approach focuses on model parameter variations. Other studies have certified fairness in models (Ruoss et al., 2020; Peychev et al., 2021) and robustness against data selection (Wang et al., 2023). 

Multilingual Large Language Model. Large Language Models (LLMs; OpenAI, 2023; Zhang et al., 2022; Brown et al., 2020; Chowdhery et al., 2022; Touvron et al., 2023a,b, inter alia) excel in English but underperform in other language. Studies have fine-tuned LLMs using monolingual or multilingual data to enhance their multilingual capabilities (Zhu et al., 2023; Li et al., 2023; Jiao et al., 2023; Cui et al., 2023; Yang et al., 2023). Embedding Tuning can activate multilingual abilities in certain languages, suggesting that the intrinsic dimension of these abilities may lie within the embedding layer (Li et al., 2018; Yuan et al., 2023b). The intrinsic dimension, the minimum parameters needed for a specific objective function, is estimated using heuristic methods and random subspace training due to computational constraints (Li et al., 2018; Aghajanyan et al., 2021). 

## 3 Observation

In this section, we first treat each token embedding values as a distribution, then use Kolmogorov-Smirnov Test to test the embedding distribution before and after tuning. Finally, we discuss some phenomena observed before and after fine-tuning. 

## 3.1 Observation 1: Not all token embeddings change during fine-tuning

First, we conceptualize the embedding values corresponding to each token as a distribution that represents the range of contextual meanings that the token can convey within the model. Each token’s embedding captures its relationships with other to-

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/622691a6-a1f2-40b5-bee7-743c94c4d403/def54b82f601f74b207f832166465249ce0da2d7e5f764ea0ac3e244993fd591.jpg)



(a) The embedding distribution gap of token $" \_ \overrightarrow { \mathbfit { a } } ^ { , * }$


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/622691a6-a1f2-40b5-bee7-743c94c4d403/a144be99d85e4691b178cb2fde26df463af7409e8273f0a8675967efa30e891f.jpg)



(b) The embedding distribution gap of token $\mathrm { \Delta \stackrel { \sim } { D o } } \mathrm { \Delta \stackrel { \sim } { \delta } }$


Figure 3: Evaluating the embedding distribution gap using the cumulative distribution function (CDF) of the tokens “_a” and “Do” before and after fine-tuning, respectively. By comparing “_a” with “Do”, the CDF quantifies the gap, making it more comparable. 

kens in various contexts, forming a multidimensional space where similar meanings cluster together. In our experiments, we fine-tune the embeddings of the LLaMA model using a dataset of 10,000 English-Canadian translation pairs. This focused tuning aims to enhance the model’s performance on translation tasks. Interestingly, our analysis shows that not all token embeddings exhibited significant parameter changes before and after the fine-tuning process. For example, as shown in Figure 2, while some tokens, like “_a”, demonstrated substantial changes in their embeddings, others, such as “Do”, remained relatively stable. 

## 3.2 Observation 2: The extent of the changes can be assessed by the shift in distribution

We observe that fine-tuning the model leads to noticeable distribution shifts in the token embeddings. To evaluate these changes systematically, we utilize cumulative distribution function (CDF) for the embeddings before and after fine-tuning. By plotting the CDF for each token, we visualize how the embedding values are distributed across both phases. The results reveal that some tokens exhibit substantial shifts in their embeddings, indicating a significant alteration in their contextual representations. For example, as shown in Figure 3, certain tokens show a pronounced rightward shift in the CDF. In contrast, other tokens display minimal changes, with their CDFs remaining stable. 

## 4 Certified Winning Tickets via KS-Lottery

In this section, we first introduce the Kolmogorov-Smirnov Test (Fasano and Franceschini, 1987) and propose KS-Lottery (§ 4.1). Then we apply KS-Lottery to find certifiable winning tickets in a multilingual transfer scenario (§ 4.2). Finally, the theoretical result of § 4.2, as well as the experimental result (§ 4.3), validate the effectiveness of KS-Lottery. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/622691a6-a1f2-40b5-bee7-743c94c4d403/f3511669b48f83741f6839c29cc381d1ac17ce09566d1b454f1cd80b131cbadd.jpg)



Figure 4: Illustration of $f ( \cdot , \pmb { \theta } _ { a } ^ { E } , \pmb { \theta } _ { b } ^ { E } )$ in 2 dimensions. Left: The concentric circles are the density contours of embedding parameters before and after tuning, and the colored landscape is the decision boundaries of $f ( \cdot )$ . Right: the distribution $\mathbb { P } \left\lceil f ( x , \theta _ { a } ^ { E } , \widetilde { \theta _ { b } ^ { E } } ) \right\rceil$ and $\mathbb { P } \left\lceil f ( x , \widetilde { \theta _ { a } ^ { E } } , \theta _ { b } ^ { E } ) \right\rceil$ $\underline { { p _ { A } } }$ is the probability $\mathbb { P } \left\lceil f ( x , \widetilde { \theta } _ { a } ^ { E } , \widetilde { \theta } _ { b } ^ { E } ) \right\rceil$ predicts x to be token $c _ { A }$ (color blue), and p as the probability of second most likely prediction (color red). $D _ { k s }$ denotes the Kolmogrov-Smirnov distance between distributions before and after tuning. We choose the set of token embeddings for tuning as those with little distribution overlap before and after fine-tuning, which may be critical to prediction.


## 4.1 KS-Lottery

Guided by the hypothesis that parameters undergoing substantial changes during fine-tuning are crucial for making predictions (Levin et al., 2022), we exam the distribution $p _ { i }$ of each token parameter (the embedding values) $\pmb \theta _ { i }$ before and after the fine-tuning. Although various metrics exist for pinpointing essential parameters (Li et al., 2016; Dalvi et al., 2019; Meng et al., 2022), these often depend on specific cutoff values, and determining the necessary number of parameters in advance is challenging. We advocate for a new approach: actively seeking out "lottery tickets" that are guaranteed to achieve similarfine-tuning outcomes with a high level of confidence. This calls for a more principled approach and the Kolmogorov-Smirnov Test stands out. In this section, we introduce the test and then theoretically explain its effectiveness in the following section. 

We propose a probing strategy that employs the Kolmogorov-Smirnov Test, which measures the maximum distance between the two CDFs. This statistical approach allows us to determine whether the observed shifts are statistically significant, providing deeper insights into how fine-tuning influences the embeddings of different tokens. The Kolmogorov-Smirnov Test is an exact test, meaning that the distribution does not depend on the underlying cumulative distribution function being tested. Specifically, we view the embedding of each LLM token i as a distribution i.e. $\theta _ { i } ^ { E } \sim p _ { i }$ The cumulative distribution function (CDF) of $\theta _ { i } ^ { E }$ and $\widetilde { \theta } _ { i } ^ { E }$ could be denoted by $\Phi _ { i } ( \theta )$ and $\widetilde { \Phi _ { i } } ( \theta )$ . The Kolmogorov-Smirnov distance between the two CDFs is $D _ { i } = \operatorname* { s u p } _ { \boldsymbol { \theta } } | \widetilde { \Phi _ { i } } ( \boldsymbol { \theta } ) - \Phi _ { i } ( \boldsymbol { \theta } ) |$ 

Now we wish to determine whether a token embedding before and after fine-tuning comes from the same distribution. Formally, we state Kolmogorov-Smirnov Test as: 

Theorem 1. (Kolmogorov-Smirnov Test, Fasano and Franceschini (1987)) The test statisticfor this Kolmogorov-Smirnov Test can be defined in terms oftwo hypotheses: 

$H _ { 0 } \colon \theta _ { i }$ and $\dot { \theta _ { i } }$ comefrom the same distribution. 

$H _ { 1 } .$ two samples aren’tfrom the same distribution. 

If test T: $D _ { i } > \tau ( \alpha )$ is passed, then $H _ { 1 }$ holds with confidence $1 - \alpha ,$ , where $D _ { i } = \operatorname* { s u p } _ { \theta } | \widetilde { \Phi _ { i } } ( \theta ) -$ $\begin{array} { r } { \Phi _ { i } ( \theta ) | , \tau ( \alpha ) = c ( \alpha ) \sqrt { \frac { 2 } { d } } , } \end{array}$ the value $o f c ( \alpha )$ is given in the reference table (Karson, 1968), and d is the parameter dimension. 

Based on the Kolmogorov-Smirnov Test, we came to propose our method, KS-Lottery. 

KS-Lottery. A parameter is designated as a “winning ticket" if it meets the criterion of rejecting the null hypothesis (no difference in the distribution of the embedding before and after fine-tuning) and the alternative hypothesis $H _ { 1 }$ (indicating a significant distributional change) is accepted. Kolmogorov-Smirnov Test ensures that if the distribution of parameter θ does not change after fine-tuning, then $\mathbb { P } \left[ D _ { i } > \tau ( \alpha ) \right] < \alpha ,$ , ensuring the majority of crucial token embeddings would be chosen by the test. 

## 4.2 Finding 1: KS-Lottery finds certifiable winning tickets in multilingual transfer.

There exists a set of winning tickets $\theta _ { a } ^ { E }$ within the token embeddings $\pmb { \theta } ^ { E } = [ \theta _ { 0 } , \theta _ { 1 } , \dots , \theta _ { | V | } ]$ ], where $\pmb { \theta } ^ { E } = [ \pmb { \theta } _ { a } ^ { E } , \pmb { \theta } _ { b } ^ { E } ] , \widetilde { \pmb { \theta } } ^ { E } = [ \widetilde { \pmb { \theta } } _ { a } ^ { E } , \pmb { \theta } _ { b } ^ { E } ]$ is the parameters of embedding layer before and after tuning. If tuning on parameters $\theta _ { a } ^ { E }$ would achieve similar performance to full tuning on $\theta ^ { E }$ downstream tasks, then we refer to $\pmb { \theta } ^ { E }$ as the winning tickets. For convenience, we discuss the downstream task as a next-token prediction problem, which is central to text generation, such that for an LLM $f ( \cdot , \pmb \theta ^ { E } )$ given input context x, generate the most probable token in vocabulary set Y. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/622691a6-a1f2-40b5-bee7-743c94c4d403/99d4c2361638c1c86ad8f1af95ebc1c2f17740fe5a1dacff58c1ba87972ac0a2.jpg)



Figure 5: Overview of Partial Training, one application of KS-Lottery. It consists of two steps: (1) finding the winning tickets in the embedding layer by Kolmogorov-Smirnov Test; (2) one way to use these winning tickets is partial tuning these tokens ensuring other parameters keep frozen.


$$
\begin{array}{l} g (x, \widetilde {\boldsymbol {\theta}} _ {a} ^ {E}, \boldsymbol {\theta} _ {b} ^ {E}) = g (x, \widetilde {\boldsymbol {\theta}} _ {a} ^ {E}, \widetilde {\boldsymbol {\theta}} _ {b} ^ {E}), \\ \text {where} g (x, \boldsymbol {\theta} _ {a} ^ {E}, \boldsymbol {\theta} _ {b} ^ {E}) = a r g \max _ {c \in \mathcal {Y}} \mathbb {P} \left[ f (x, \boldsymbol {\theta} _ {a} ^ {E}, \boldsymbol {\theta} _ {b} ^ {E}) = c \right] \end{array}\tag{1}
$$

Suppose that when the LLM $f ( \cdot , \pmb { \theta } _ { a } ^ { E } , \pmb { \theta } _ { b } ^ { E } )$ predicts any given input x, the most probable token $c _ { A }$ is returned with probability $p _ { A }$ . Given that small changes to parameters in a smooth subspace do not affect decision boundary (Muravev and Petiushko, 2021; Zhao et al., 2022), we could probably guarantee that training on KS-Lottery selected token embeddings could achieve the same performance as full layer Embed Tuning at high confidence. The intuition of the theory is illustrated in Figure 4. Our main theoretical results are as follows, the proofs can be found in Appendix B: 

Theorem 2. (Certified Winning Tickets) Let $f ( \cdot , \pmb { \theta } _ { a } , \pmb { \theta } _ { b } ) : \mathbb { R } ^ { d }  \mathcal { V }$ be any random or deterministicfunction, and let g be defined as in Equation 1. For any input x, suppose $c _ { A } \in \mathcal { V } _ { : }$ , the bounds of prediction based on random variable parameters $\widetilde { \pmb { \theta } } _ { a } , \widetilde { \pmb { \theta } } _ { b } , p _ { A } , \overline { { p _ { B } } } \in [ 0 , 1 ]$ satisfies 

$$
\begin{array}{l} \mathbb {P} \left(f (x, \widetilde {\boldsymbol {\theta}} _ {a}, \widetilde {\boldsymbol {\theta}} _ {b}) = c _ {A}\right) \\ \geq \underline {{p _ {A}}} \geq \overline {{p _ {B}}} \geq \max _ {c \neq c _ {A}} \mathbb {P} \left(f (x, \widetilde {\boldsymbol {\theta}} _ {a}, \widetilde {\boldsymbol {\theta}} _ {b} = c)\right) \end{array}\tag{2}
$$

If the set of parameters $\theta _ { b }$ satisfies 

$$
D (\widetilde {\theta} _ {b} ^ {i}, \theta_ {b} ^ {i}) <   \tau (\alpha) <   \frac {p _ {A} - \overline {{p _ {B}}}}{2},\tag{3}
$$

for all $i \in V _ { b }$ , the generator partial-tuned on parameters $\pmb { \theta } _ { a }$ always return token $c _ { A }$ , i.e. $g ( x , \widetilde { \pmb { \theta } } _ { a } ^ { E } , \pmb { \theta } _ { b } ^ { E } ) = c _ { A }$ 

Remark 3. (About $\theta _ { a } )$ Theorem 2 immediately holds for Partial Transfer setting as specified in § 4.3. As for Partial Tuning setting, we need to use the hypothesis that the value of $\pmb { \theta } _ { a }$ in the Embed Tuning setting and partial-tuning setting are the same for Theorem 2 to hold. This is due to $\pmb { \theta } _ { a }$ taking major effect during Embed Tuning despite small changes in $\pmb { \theta } _ { b }$ . We also show with empirical analysis in § 5.2 that $\dot { \pmb { \theta } } _ { a }$ in two tuning settings share the same distribution. 

Remark 4. (About α) Practically, $\tau ( \alpha )$ is small. In the case of fully embedding tuned LLM that has optimal performance and produces confident prediction, $1 > p _ { A } \gg p _ { B } > 0$ , we can choose small $\begin{array} { r } { \alpha = \tau ^ { - 1 } \big ( \frac { p _ { A } - p _ { B } } { 2 } \big ) } \end{array}$ while maintaining performance. In this case, the Kolmogorov-Smirnov Test samples fewer token embeddings for fine-tuning. 

Remark 5. (About Applicability) This theorem could be used for certifying fine-tuning lottery tickets for any black-box model and input. Note that KS-Lottery could be applied to the entire model, though we only perform testing on the embedding layer for convenience and training stability. 

## 4.3 Finding 2: 18 identified winning tickets (18 tokens) achieves remarkable performance.

There are two different ways to verify the effectiveness of winning tickets: Partial Tuning (sufficient condition) and Partial Transfer (necessary condition). Partial Tuning. As shown in Figure 5, only train winning tickets, keeping the remaining parameters frozen. Partial Transfer. Given a model trained by Embed Tuning, we select the winning tickets from the embedding layer and use them to replace the corresponding parameters in the original model, thus curating a new model. 


Table 1: Empirical verification of winning tickets with Partial Tuning and Partial Transfer with bilingual translation data. * denotes that the random results from three distinct seeds.


<table><tr><td rowspan="2">Method</td><td colspan="6">LLaMA-7B</td><td colspan="6">Mistral-7B</td></tr><tr><td># Token</td><td>en→ro</td><td>en→es</td><td>en→de</td><td>en→ca</td><td>Avg.</td><td># Token</td><td>en→ro</td><td>en→es</td><td>en→de</td><td>en→ca</td><td>Avg.</td></tr><tr><td>Original Model</td><td></td><td>3.5</td><td>4.8</td><td>4.8</td><td>5.7</td><td>4.6</td><td></td><td>13.4</td><td>11.7</td><td>14.2</td><td>16.8</td><td>14.0</td></tr><tr><td>Full Tuning</td><td>-</td><td>28.3</td><td>23.5</td><td>22.5</td><td>34.9</td><td>27.3</td><td>-</td><td>33.4</td><td>25.1</td><td>29.1</td><td>37.8</td><td>31.4</td></tr><tr><td>Embed Tuning</td><td></td><td>28.7</td><td>25.5</td><td>25.8</td><td>36.1</td><td>29.0</td><td></td><td>34.2</td><td>28.3</td><td>33.2</td><td>40.2</td><td>34.0</td></tr><tr><td>Random Tuning*</td><td>&lt;18</td><td>0.1</td><td>0.1</td><td>0.1</td><td>0.1</td><td>0.1</td><td>&lt;169</td><td>16.2</td><td>13.9</td><td>15.3</td><td>20.3</td><td>16.4</td></tr><tr><td>Partial Tuning</td><td></td><td>20.7</td><td>26.7</td><td>26.4</td><td>37.7</td><td>27.9</td><td></td><td>23.7</td><td>26.3</td><td>27.2</td><td>34.2</td><td>27.9</td></tr><tr><td>Partial Transfer</td><td>(p-value &lt; 0.05)</td><td>23.4</td><td>22.6</td><td>16.1</td><td>31.1</td><td>23.4</td><td>(p-value &lt; 0.05)</td><td>26.7</td><td>27.3</td><td>30.5</td><td>37.5</td><td>30.5</td></tr><tr><td>Random Tuning*</td><td>&lt;100</td><td>0.1</td><td>0.1</td><td>0.1</td><td>0.1</td><td>0.1</td><td>&lt;170</td><td>17.3</td><td>13.9</td><td>15.3</td><td>20.4</td><td>16.7</td></tr><tr><td>Partial Tuning</td><td></td><td>26.9</td><td>27.3</td><td>27.2</td><td>37.4</td><td>29.7</td><td></td><td>23.7</td><td>26.3</td><td>27.2</td><td>34.2</td><td>27.9</td></tr><tr><td>Partial Transfer</td><td>(p-value &lt; 0.25)</td><td>25.1</td><td>26.5</td><td>30.4</td><td>34.4</td><td>29.1</td><td>(p-value &lt; 0.25)</td><td>26.7</td><td>27.3</td><td>30.5</td><td>37.5</td><td>30.5</td></tr><tr><td>Random Tuning*</td><td></td><td>5.8</td><td>8.7</td><td>5.2</td><td>11.8</td><td>7.9</td><td></td><td>16.3</td><td>13.9</td><td>15.4</td><td>20.3</td><td>16.5</td></tr><tr><td>Partial Tuning</td><td>≤800</td><td>29.4</td><td>27.3</td><td>30.1</td><td>37.7</td><td>31.1</td><td>≤180</td><td>30.7</td><td>28.2</td><td>29.5</td><td>39.2</td><td>31.9</td></tr><tr><td>Partial Transfer</td><td></td><td>30.2</td><td>26.7</td><td>30.4</td><td>37.6</td><td>31.2</td><td></td><td>33.7</td><td>27.9</td><td>32.6</td><td>40.3</td><td>33.6</td></tr></table>

Algorithm 1: Next Token Prediction Certification
Input: Sequence x, ground truth output token is y.
LLM $f(x, \boldsymbol{\theta})$ that maps sequence to the probability of the next token class, with pre-trained parameters $\theta$ .
Multilingual training set D. KS-Lottery parameter $\alpha$ .
Output: Whether $g(x, \widetilde{\boldsymbol{\theta}}_{a}^{E}, \boldsymbol{\theta}_{b}^{E}) = y$ fine-tune LLM $f(x, \boldsymbol{\theta})$ on D with parameters $\widetilde{\boldsymbol{\theta}}$ . $\boldsymbol{\theta}_{a}^{E}, \boldsymbol{\theta}_{b}^{E} = \text{KS-Lottery}(\boldsymbol{\theta}, \widetilde{\boldsymbol{\theta}}, \tau(\alpha))$ $p_{A}, \overline{p_{B}} = \text{Top-2}(f(x, \boldsymbol{\theta}_{a}^{E}, \boldsymbol{\theta}_{b}^{E}))$ if $g(x, \widetilde{\boldsymbol{\theta}}_{a}^{E}, \widetilde{\boldsymbol{\theta}}_{b}^{E}) = y$ and $\frac{p_{A}-\overline{p_{B}}}{2} > \tau(\alpha)$ then
| return True // Certification can be provided.
else
| return False // Certification cannot be provided.
end 

As shown in Table 1, just tuning winning tickets can achieve results on par with Embed Tuning. Meanwhile, by only modifying the winning tickets, it manages to retain 86.9%(31.3/36.1) of the performance. The experimental result demonstrates that a low-dimensional subspace exists, guided by the winning tickets, that can achieve comparable performance as optimizing all parameters in the embedding-tuned model. The result empirically verifies the effectiveness of winning tickets. 

## 5 Analysis

KS-Lottery . There are three different ways to use the winning tickets. Partial Tuning, Partial Transfer and Frequency Tuning, which only train the high-frequency tokens in the corpus. 

Other Baselines. Original Model Directly using the LLaMA-7B (Touvron et al., 2023a)/Mistral-7B (Jiang et al., 2023) weight on test data, without any tuning. Random Tuning Only the tokens randomly selected in the embedding layer are fine-tuned, while the remaining parameters are kept frozen. Full Tuning is an approach to transfer learning where the weights of a pretrained entire model are trained on new data. Embed Tuning Merely fine-tuning the embedding layer of a model keeping the remaining parameters frozen. LoRA (Hu et al., 2022) utilizes low-rank matrices for approximating parameter updates. QLoRA (Dettmers et al., 2023) reduces memory usage enough to efficiently finetune. Prefix-Tuning (Li and Liang, 2021) introduce a lightweight prefix module into the input layer and each transformer layer, enabling efficient training over these modules. 

Training and Evaluation. To ensure a fair comparison, we apply various parameter-efficient settings on LLaMA-7B using Lego-MT (Yuan et al., 2023a) 10k data. For full tuning, training with LoRA, and Embed Tuning, we set the learning rate to 2e − 5 and the number of epochs to 3. For partial tuning, prefix-tuning, and KS-Lottery, we set the learning rate to 1e − 2 and the number of epochs to 5. All other parameters are kept consistent across all settings. We test each model on the Flores-101 (Goyal et al., 2022) devtest, which offers human-written translation pairs across 101 languages. In alignment with Flores-101, we employ the same evaluation metric, sentence piece BLEU (spBLEU) on beam size= 4, to assess multilingual capabilities. 

## 5.1 KS-Lottery Certification

Certified Accuracy, when using certified winning tickets Embed Tuning, is measured as the proportion of correct predictions from an embedding tuned model (reference model) that is certified to be correct at a significance level of α. The certification process follows Theorem 2 and is stated in Algorithm 1, where for each prediction based on input sequence x, we compare the probability gap between two most-likely tokens by the original LLaMA, i.e. $\frac { p _ { A } - \overline { { p _ { B } } } } { 2 }$ and $\tau ( \alpha ) . \ \tau ( \alpha )$ is a static value that could be obtained through the equation, though we use the value estimated with Scipy Kolmogorov-Smirnov test to be precise. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/622691a6-a1f2-40b5-bee7-743c94c4d403/089d168e07e8657ed8ccd37eb92dadba6d8cb09ae77c21f2a8f93ae8f0d3a260.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/622691a6-a1f2-40b5-bee7-743c94c4d403/39a54387f08c5b9031b41206980774055fc8f4c771bcee1c31762e67f9baaf4b.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/622691a6-a1f2-40b5-bee7-743c94c4d403/6eb3051a11d046ba7cb8e3963b127c0845b2a0ec85cf39151afb0dc58ecb2d8b.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/622691a6-a1f2-40b5-bee7-743c94c4d403/85cc4a1409800a086d68985275caef42363edffd0feb464ae0efea1d381e1929.jpg)



Figure 6: Certified experiment under Partial Tuning setting. A: Estimation of τ(α) w.r.t different α by running Kolmogorov-Smirnov Test between the distribution of LLaMA-7B embedding and fine-tuned embedding on different datasets. B C D: Comparison between Certified Accuracy and Empirical Prediction Accuracy w.r.t. different α on 3 datasets. More results are shown in Appendix C.



Table 2: Partial Tuning of the winning ticket is a parameter-efficient method. Its performance is demonstrated by comparing it with other parameter-efficient methods on the Flores-101 devtest.


<table><tr><td>Method</td><td># Param</td><td>en→ro</td><td>en→es</td><td>en→ca</td><td>en→no</td><td>Avg.</td></tr><tr><td>LLaMA w/o Tuning</td><td>7B</td><td>3.5</td><td>4.8</td><td>5.7</td><td>3.2</td><td>4.3</td></tr><tr><td>Full Fine-Tuning</td><td>7B</td><td>28.3</td><td>23.5</td><td>34.9</td><td>21.2</td><td>27.0</td></tr><tr><td>LoRA</td><td>4.2M</td><td>29.8</td><td>26.4</td><td>37.3</td><td>19.6</td><td>28.3</td></tr><tr><td>Prefix Tuning</td><td>0.04M</td><td>17.6</td><td>20.4</td><td>20.8</td><td>11.0</td><td>17.4</td></tr><tr><td>Embed Tuning</td><td>131M</td><td>28.7</td><td>25.5</td><td>36.1</td><td>19.5</td><td>27.5</td></tr><tr><td>Partial Tuning</td><td>0.29/1M</td><td>27.9</td><td>26.7</td><td>37.7</td><td>19.4</td><td>27.9</td></tr></table>

To check the validity of our certification method and test the empirical tightness of the certification bound, we experiment on Flores-101 devtest. A model, developed using partial tuning, and Embed Tuning, utilizes both the instruction and a partial sequence of reference tokens as input. Temporarily both settings of certification only use the first twenty token prediction results for calculation for a fair comparison and don’t let overlong sentences dominate the results. It subsequently predicts the next token in the partial reference sentence. The “prediction accuracy” is ascertained by comparing the predicted token, generated by the partially tuned model, with the reference token. 

Our theoretical discovery provides certification for the lottery tickets hypothesis. In Figure 6 that plots certified accuracy as a function of α, the certified accuracy always increases gradually until reaching the same value as the prediction accuracy. This is due to the decrease in $\tau ( \alpha )$ as α increases and reaches 0 when α = 1. At any α, the empirical estimation of certified accuracy is a lower bound of prediction accuracy, providing a performance guarantee for tuning on the lottery tickets. 


Table 3: The efficiency and interpretability of KS-Lottery. Given an en→ca Embed Tuning bilingual model, KS-Lottery is denoted by the p-value, whereas the metrics from other evaluation methods are normalized for comparability by calculating the importance # rank/32000.


<table><tr><td>Idx</td><td>Token String</td><td>Frequency Rank</td><td>KS-Lottery</td><td>Cos</td><td>Absolute</td><td>Relative</td><td>Ratio</td><td>KL</td></tr><tr><td>13</td><td>\n</td><td>5</td><td>0.0000</td><td>0.0001</td><td>0.9972</td><td>0.0004</td><td>0.0004</td><td>0.0154</td></tr><tr><td>263</td><td>a</td><td>10</td><td>0.0000</td><td>0.0016</td><td>0.9964</td><td>0.1376</td><td>0.1376</td><td>0.0071</td></tr><tr><td>278</td><td>the</td><td>570</td><td>0.0000</td><td>0.0003</td><td>0.9508</td><td>0.0546</td><td>0.0546</td><td>0.0018</td></tr><tr><td>297</td><td>in</td><td>286</td><td>0.0000</td><td>0.0034</td><td>0.8807</td><td>0.9885</td><td>0.9885</td><td>0.0353</td></tr><tr><td>304</td><td>to</td><td>146</td><td>0.0000</td><td>0.0016</td><td>0.8860</td><td>0.9032</td><td>0.9032</td><td>0.0155</td></tr><tr><td>310</td><td>of</td><td>264</td><td>0.0000</td><td>0.0009</td><td>0.1571</td><td>0.9955</td><td>0.9955</td><td>0.0171</td></tr><tr><td>322</td><td>and</td><td>680</td><td>0.0000</td><td>0.0004</td><td>0.9913</td><td>0.0060</td><td>0.0060</td><td>0.0077</td></tr><tr><td>338</td><td>is</td><td>1356</td><td>0.0214</td><td>0.0005</td><td>0.1557</td><td>0.0201</td><td>0.0201</td><td>0.0023</td></tr><tr><td>376</td><td>&quot;</td><td>174</td><td>0.0163</td><td>0.0000</td><td>0.0375</td><td>0.9796</td><td>0.9796</td><td>0.0001</td></tr><tr><td>393</td><td>that</td><td>2965</td><td>0.0175</td><td>0.0007</td><td>0.9922</td><td>0.0086</td><td>0.0086</td><td>0.0035</td></tr><tr><td>411</td><td>with</td><td>2102</td><td>0.0459</td><td>0.0013</td><td>0.1553</td><td>0.1493</td><td>0.1493</td><td>0.0054</td></tr><tr><td>29871</td><td>~</td><td>6</td><td>0.0000</td><td>0.0005</td><td>0.9936</td><td>0.9908</td><td>0.9908</td><td>0.0026</td></tr><tr><td>29889</td><td>.</td><td>4</td><td>0.0000</td><td>0.0002</td><td>0.9962</td><td>0.8387</td><td>0.8387</td><td>0.0068</td></tr><tr><td>29892</td><td>,</td><td>3</td><td>0.0000</td><td>0.0002</td><td>0.9976</td><td>0.9950</td><td>0.9950</td><td>0.0048</td></tr><tr><td>29896</td><td>1</td><td>9</td><td>0.0000</td><td>0.0023</td><td>0.0440</td><td>0.0500</td><td>0.0500</td><td>0.0139</td></tr><tr><td>29900</td><td>0</td><td>8</td><td>0.0008</td><td>0.0045</td><td>0.0728</td><td>0.9743</td><td>0.9743</td><td>0.0254</td></tr><tr><td>29901</td><td>:</td><td>30</td><td>0.0069</td><td>0.0003</td><td>0.9791</td><td>0.9950</td><td>0.9950</td><td>0.0008</td></tr><tr><td>29949</td><td>O</td><td>294</td><td>0.0406</td><td>0.0028</td><td>0.9336</td><td>0.0359</td><td>0.0359</td><td>0.0201</td></tr></table>

Certification lower bound is tighter for larger α . When $\underline { { p _ { A } } } \gg \overline { { p _ { B } } } ,$ , there must exist a α that satisfies the certification constraint. Since ${ \underline { { p _ { A } } } }$ and $\overline { { p _ { B } } }$ are input and model dependent, we empirically assessed the tightness of our bound by comparing the estimated value of certified accuracy with prediction accuracy. As shown in Figure 6, the bound is tighter when $\alpha  1$ , and the gap is larger when $\alpha  0$ . This is due to when fewer token embeddings are chosen, the certification expects the performance to be worse, though in actual tuning even zero-shot performance is quite good due to pre-training. The gap is quite small in practical cases, the gap is about 2% on average for $\alpha = 0 . 0 5$ and around 1.5% on average for $\alpha = 0 . 5$ . Therefore, we recommend using $\alpha \ge 0 . 0 5$ significance rate for guaranteed. 

## 5.2 KS-Lottery Efficiency and Interpretability

Partial Tuning can serve as a parameter-efficient tuning method. We proceed to evaluate it as a method that optimizes parameter usage, in comparison with another similar method. As illustrated in Table 2, in contrast to other methods such as LoRA (Hu et al., 2022) and Prefix-tuning (Li and Liang, 2021), Partial Tuning eliminates the need for an additional model structure. Remarkably, our method not only matches but frequently surpasses the performance of these alternate approaches. 


Table 4: The best result from Frequency Tuning, Partial Tuning, and the result of Embed Tuning.


<table><tr><td>Setting</td><td>en→ca</td><td>en→da</td><td>en→de</td><td>en→es</td><td>en→no</td><td>Avg.</td></tr><tr><td>Embedding Tuning</td><td>36.1</td><td>32.7</td><td>25.8</td><td>25.5</td><td>19.5</td><td>27.9</td></tr><tr><td>Partial Tuning</td><td>37.7</td><td>33.3</td><td>30.1</td><td>27.3</td><td>19.8</td><td>29.6</td></tr><tr><td>Frequency Tuning</td><td>37.7</td><td>30.0</td><td>30.7</td><td>26.1</td><td>19.6</td><td>28.8</td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/622691a6-a1f2-40b5-bee7-743c94c4d403/4f2fc75cec4693bdf0bbf523c73600bf1a11750e2909126a4b79f82272329439.jpg)



Figure 7: When selective tokens are restricted from being updated, the model’s fine-tuning process for downstream tasks loses its effectiveness.


Comparing with other selection methods, KS-Lottery is parameter-efficient. There are other ways (more details in Appendix C) to select critical parameters, such as Cos, Absolute, Relative, Ratio, and KL. In Table 3, we selected only 18 selective tokens by KS-Lottery on en→ca bilingual data with extremely stringent requirements $\mathrm { ( p \mathrm { - } v a l u e < 0 . 0 5 ) }$ At this time, we used other evaluation methods to measure the importance of these tokens separately: the evaluation results of these tokens based on distribution changes (such as Cos, KL) are highly consistent with KS-Lottery; those evaluation methods based on value changes (Absolute, Relative, and Ratio) tend to give these tokens very low importance evaluation results. If we use different methods to select the top 18 important tokens for Partial Transfer experiments, the performance of different selective methods is 31.3 (KS-Lottery), 28.7 (Cos), 5.8 (Absolute), 5.8 (Relative), 5.8 (Ratio) and 1.6 (KL). Based on the experimatl results, we find that except for the results of Cos, which are close to the performance of KS-Lottery (but still -2.6 points), the translation performance of other methods is very poor. 


Table 5: Impact of Different α on the en→ca. For α values greater than 0.05, the verification percentage reaches 94.3%, indicating substantial but not complete verification. The empirical accuracy under these conditions is also satisfactory.


<table><tr><td>pvalue&lt;α</td><td>Verified Percentage (% Di &gt;τ(α))</td><td>Original Accuracy</td><td>Certified Accuracy</td><td>Empirical Accuracy</td></tr><tr><td>0.05</td><td>94.3</td><td>75.86</td><td>65.77</td><td>78.87</td></tr><tr><td>0.10</td><td>95.1</td><td>75.86</td><td>65.98</td><td>78.85</td></tr><tr><td>0.20</td><td>95.6</td><td>75.86</td><td>66.15</td><td>78.93</td></tr><tr><td>0.30</td><td>95.8</td><td>75.86</td><td>66.19</td><td>78.98</td></tr><tr><td>0.50</td><td>96.5</td><td>75.86</td><td>66.35</td><td>78.97</td></tr><tr><td>0.70</td><td>96.8</td><td>75.86</td><td>66.48</td><td>79.03</td></tr><tr><td>0.90</td><td>97.5</td><td>75.86</td><td>66.66</td><td>79.09</td></tr><tr><td>1.00</td><td>100.0</td><td>75.86</td><td>67.23</td><td>78.46</td></tr></table>

Winning tickets are high-frequency tokens in the corpus. In the embedding layer, each dimension is associated with a meaningful token index. By referencing the vocabulary, we can decode the text representation corresponding to this index, as illustrated in Table 3 (Str). To examine the distinct characteristics of these tokens, we retrieve 50k sentences in ca language from the MC4 (Raffel et al., 2019) dataset, then employ LLaMA’s tokenizer to segment all sentences, tallying the occurrence of each token within the corpus. Table 3 (Freq.) indicates winning tickets commonly associated with the most frequently occurring tokens in the corpus. Based on this finding, we start the training only with the high-frequency tokens in the corpus. The summary of different tuning setting, as shown in Table 4, suggests the training is sufficient with frequency token tuning. 

Sensitivity in the siginificance level Selection (α). Our theoretical framework provides a way for heuristically finding an effective $\alpha ,$ as mentioned in Remark5. Since the original fully-tuned prediction model is available, we could directly predict the range of $\frac { p _ { A } - p _ { B } } { 2 }$ as $[ s _ { m i n } , s _ { m a x } ]$ for all predictions in the dataset. An ideal α would satisfy $\tau ( \alpha ) < s _ { m i n }$ , under this condition, every data point in the dataset is guaranteed to make the same prediction as the original prediction model. 

In practice, we don’t need a full certification of the dataset to make the KS-Lottery work, as this would result in quite large α values. As we can see in the en→ca example below: for α larger than 0.05, the verified percentage is an acceptable 94.3%, though not fully verified, and the empirical accuracy proves acceptable. In practice, the acceptable range of α varies between datasets, though usually α within 0.05-0.4 guarantees a verified percentage 95%. 


Table 6: With varying values of α, the distribution of winning tickets remains largely consistent across both Partial Tuning and Partial Transfer.


<table><tr><td>α</td><td>ro</td><td>es</td><td>de</td><td>ca</td><td>pt</td><td>da</td><td>no</td><td>bs</td></tr><tr><td>0.05</td><td>0.56</td><td>0.67</td><td>0.81</td><td>0.78</td><td>0.82</td><td>0.33</td><td>0.64</td><td>0.82</td></tr><tr><td>0.1</td><td>0.75</td><td>0.78</td><td>0.72</td><td>0.80</td><td>0.85</td><td>0.71</td><td>0.79</td><td>0.81</td></tr><tr><td>0.2</td><td>0.78</td><td>0.75</td><td>0.68</td><td>0.73</td><td>0.75</td><td>0.79</td><td>0.83</td><td>0.81</td></tr><tr><td>0.3</td><td>0.69</td><td>0.82</td><td>0.85</td><td>0.77</td><td>0.86</td><td>0.76</td><td>0.81</td><td>0.80</td></tr><tr><td>0.4</td><td>0.75</td><td>0.92</td><td>0.89</td><td>0.83</td><td>0.83</td><td>0.78</td><td>0.82</td><td>0.84</td></tr><tr><td>0.5</td><td>0.83</td><td>0.87</td><td>0.82</td><td>0.80</td><td>0.85</td><td>0.86</td><td>0.84</td><td>0.90</td></tr><tr><td>0.6</td><td>0.83</td><td>0.90</td><td>0.88</td><td>0.85</td><td>0.86</td><td>0.86</td><td>0.85</td><td>0.92</td></tr><tr><td>0.7</td><td>0.83</td><td>0.91</td><td>0.88</td><td>0.86</td><td>0.90</td><td>0.86</td><td>0.89</td><td>0.88</td></tr><tr><td>0.8</td><td>0.82</td><td>0.90</td><td>0.87</td><td>0.87</td><td>0.90</td><td>0.87</td><td>0.88</td><td>0.88</td></tr><tr><td>0.9</td><td>0.82</td><td>0.86</td><td>0.84</td><td>0.85</td><td>0.88</td><td>0.84</td><td>0.86</td><td>0.85</td></tr></table>

The distribution of winning tickets under both Partial Tuning and Partial Transfer is largely identical. By conducting KS-Lottery with varying α values, we obtain different winning tickets (denoted as $\theta _ { a } )$ . Furthermore, we perform a Kolmogorov-Smirnov Test on the winning tickets tuned with partial tuning and Embed Tuning, and compute the number of tokens that exhibit a significant difference $( { \widetilde { \pmb { \theta } } } _ { a } ^ { \prime } )$ . The ratio of unchanged tokens is calculated using the formula $1 - \widetilde { \pmb { \theta } } _ { a } ^ { \prime } / \widetilde { \pmb { \theta } } _ { a }$ As shown in Table 6, the distribution of winning tickets before and after tuning (Partial Transfer vs Partial Tuning) remains largely consistent. 

Keeping the parameters of winning tickets frozen while tuning the remaining parameters could lead to the collapse of the Embed Tuning. Our previous experiments have highlighted the significance of winning tickets in training, while it’s intriguing to consider whether such an important function can be replaced by other tokens. To delve into this problem, we freeze the winning tickets, and fine-tune the reset parameters on bilingual data. As shown in Figure 7, a small amount of disabled winning tickets is acceptable for tuning. However, as the number of disabled tokens increases, the entire tuning process crashes. The process reveals a remarkable finding: for a vocabulary size of 32k, 1k is a very small, yet fewer than 1k winning tickets play a crucial role in Embed Tuning. 

Applying KS-Lottery on the whole model. After training all model parameters using en→ca bilingual sentence pair from Lego-MT (Yuan et al., 2023a), we utilize the KS-Lottery to identify a small parameter set for multilingual transfer by comparing the parameters before and after tuning. Interestingly, no parameters exhibite significant changes before and after tuning. Concurrently, the study by Yuan et al. (2023b) revealed that finetuning specific layers, including the embedding layer, can yield results comparable to those of full-tuning. Figure 8 reveals that following finetuning on the Lego-MT (Yuan et al., 2023a) en→ca bilingual dataset, a subset of token embeddings exhibit significant changes in their parameters. Additionally, a small number of parameters (fewer than two for each layer with a significance level of $\alpha < 0 . 0 5 )$ demonstrate substantial changes within LayerNorm. However, for multilingual transfer, the impact of LayerNorm varies and is not uniform across the lower and higher layers. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/622691a6-a1f2-40b5-bee7-743c94c4d403/d8191f2e13900e5e195cb7ce558bd43c686d3322170fcd7b109c620d1cba71a6.jpg)



Figure 8: Applying KS-Lottery on whole LLaMA-7B which is single-layer tuned on Lego-MT en→ca 10k data. Each layer is trained in isolation and is analyzed by KS-Lottery to identify the parameters with signifi cant changes (as indicated by scatter points above the red line). We find that within each Transformer layer, changes are primarily focused on LayerNorm, while other notable changes occur in the embedding layer.


## 6 Conclusion

This work presents a novel method, KS-Lottery, which applies the lottery ticket hypothesis to LLMs fine-tuning. By employing the Kolmogorov-Smirnov Test, KS-Lottery first analyzes the shift in the parameter distribution before and after finetuning, and then identifies a small but effective subset of LLM parameters. Theoretical evidence confirms that KS-Lottery can pinpoint certified winning tickets within the embedding layer, thereby ensuring performance equivalent to full tuning. Notably, KS-Lottery surpasses other parameterefficient tuning algorithms by identifying fewer parameters for fine-tuning while maintaining similar performance levels. 

## References



Armen Aghajanyan, Sonal Gupta, and Luke Zettlemoyer. 2021. Intrinsic dimensionality explains the effectiveness of language model fine-tuning. In Proceedings of the 59th Annual Meeting of the Associationfor Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 7319–7328, Online. Association for Computational Linguistics. 





Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901. 





Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. 2022. Palm: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311. 





Yiming Cui, Ziqing Yang, and Xin Yao. 2023. Efficient and effective text encoding for chinese llama and alpaca. arXiv preprint arXiv:2304.08177. 





Fahim Dalvi, Nadir Durrani, Hassan Sajjad, Yonatan Belinkov, Anthony Bau, and James Glass. 2019. What is one grain of sand in the desert? analyzing individual neurons in deep nlp models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pages 6309–6317. 





Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. 2023. Qlora: Efficient finetuning of quantized llms. In Advances in Neural Information Processing Systems, volume 36, pages 10088–10115. Curran Associates, Inc. 





Ning Ding, Yujia Qin, Guang Yang, Fuchao Wei, Zonghan Yang, Yusheng Su, Shengding Hu, Yulin Chen, Chi-Min Chan, Weize Chen, Jing Yi, Weilin Zhao, Xiaozhi Wang, Zhiyuan Liu, Hai-Tao Zheng, Jianfei Chen, Yang Liu, Jie Tang, Juanzi Li, and Maosong Sun. 2023. Parameter-efficient fine-tuning of largescale pre-trained language models. Nature Machine Intelligence, 5(3):220–235. 





G. Fasano and A. Franceschini. 1987. A multidimensional version of the Kolmogorov–Smirnov test. Monthly Notices ofthe Royal Astronomical Society, 225(1):155–170. 





Jonathan Frankle and Michael Carbin. 2018. The lottery ticket hypothesis: Finding sparse, trainable neural networks. arXiv preprint arXiv:1803.03635. 





Jonathan Frankle and Michael Carbin. 2019. The lottery ticket hypothesis: Finding sparse, trainable neural networks. In International Conference on Learning Representations. 





Naman Goyal, Cynthia Gao, Vishrav Chaudhary, Peng-Jen Chen, Guillaume Wenzek, Da Ju, Sanjana Krishnan, Marc’Aurelio Ranzato, Francisco Guzmán, and Angela Fan. 2022. The Flores-101 evaluation benchmark for low-resource and multilingual machine translation. Transactions ofthe Associationfor Computational Linguistics, 10:522–538. 





Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations. 





Robin Jia, Aditi Raghunathan, Kerem Göksel, and Percy Liang. 2019. Certified robustness to adversarial word substitutions. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 4129–4142, Hong Kong, China. Association for Computational Linguistics. 





Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de Las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2023. Mistral 7b. CoRR, abs/2310.06825. 





Wenxiang Jiao, Jen tse Huang, Wenxuan Wang, Zhiwei He, Tian Liang, Xing Wang, Shuming Shi, and Zhaopeng Tu. 2023. Parrot: Translating during chat using large language models tuned with human translation and feedback. In Findings ofEMNLP. 





Marvin Karson. 1968. Handbook of methods of applied statistics. volume i: Techniques of computation descriptive methods, and statistical inference. volume ii: Planning of surveys and experiments. i. m. chakravarti, r. g. laha, and j. roy, new york, john wiley; 1967, $9.00. Journal ofthe American Statistical Association, 63(323):1047–1049. 





Mathias Lecuyer, Vaggelis Atlidakis, Roxana Geambasu, Daniel Hsu, and Suman Jana. 2019. Certified robustness to adversarial examples with differential privacy. Preprint, arXiv:1802.03471. 





Roman Levin, Manli Shu, Eitan Borgnia, Furong Huang, Micah Goldblum, and Tom Goldstein. 2022. Where do models go wrong? parameter-space saliency maps for explainability. Advances in Neural Information Processing Systems, 35:15602–15615. 





Chunyuan Li, Heerad Farkhoor, Rosanne Liu, and Jason Yosinski. 2018. Measuring the intrinsic dimension of objective landscapes. In International Conference on Learning Representations. 





Jiahuan Li, Hao Zhou, Shujian Huang, Shanbo Cheng, and Jiajun Chen. 2023. Eliciting the translation ability of large language models via multilingual finetuning with translation instructions. Preprint, i 5 5 





Jiwei Li, Will Monroe, and Dan Jurafsky. 2016. Understanding neural networks through representation erasure. arXiv preprint arXiv:1612.08220. 





Xiang Lisa Li and Percy Liang. 2021. Prefix-tuning: Optimizing continuous prompts for generation. In Proceedings ofthe 59th Annual Meeting ofthe Associationfor Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 4582– 4597, Online. Association for Computational Linguistics. 





Eran Malach, Gilad Yehudai, Shai Shalev-Schwartz, and Ohad Shamir. 2020. Proving the lottery ticket hypothesis: Pruning is all you need. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 6682–6691. PMLR. 





Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. 2022. Locating and editing factual associations in GPT. Advances in Neural Information Processing Systems, 35. 





Nikita Muravev and Aleksandr Petiushko. 2021. Certified robustness via randomized smoothing over multiplicative parameters of input transformations. arXiv preprint arXiv:2106.14432. 





OpenAI. 2023. Gpt-4 technical report. Preprint, arXiv:2303.08774. 





Momchil Peychev, Anian Ruoss, Mislav Balunovic, Maximilian Baader, and Martin T. Vechev. 2021. Latent space smoothing for individually fair representations. CoRR, abs/2111.13650. 





Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2019. Exploring the limits of transfer learning with a unified text-to-text transformer. arXiv e-prints. 





Aditi Raghunathan, Jacob Steinhardt, and Percy Liang. 2018. Semidefinite relaxations for certifying robustness to adversarial examples. CoRR, abs/1811.01057. 





Anian Ruoss, Mislav Balunovic, Marc Fischer, and Martin T. Vechev. 2020. Learning certified individually fair representations. CoRR, abs/2002.10312. 





Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023a. Llama: Open and efficient foundation language models. Preprint, arXiv:2302.13971. 





Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023b. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288 





Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. 2018. GLUE: A multi-task benchmark and analysis platform for natural language understanding. In Proceedings ofthe 2018 EMNLP Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP, pages 353–355, Brussels, Belgium. Association for Computational Linguistics. 





Yudong Wang, Chang Ma, Qingxiu Dong, Lingpeng Kong, and Jingjing Xu. 2023. A challenging benchmark for low-resource learning. arXiv preprint arXiv:2303.03840. 





Wen Yang, Chong Li, Jiajun Zhang, and Chengqing Zong. 2023. Bigtrans: Augmenting large language models with multilingual translation capability over 100 languages. arXiv preprint arXiv:2305.18098. 





Fei Yuan, Yinquan Lu, Wenhao Zhu, Lingpeng Kong, Lei Li, Yu Qiao, and Jingjing Xu. 2023a. Lego-MT: Learning detachable models for massively multilingual machine translation. In Findings of the Association for Computational Linguistics: ACL 2023, pages 11518–11533, Toronto, Canada. Association for Computational Linguistics. 





Fei Yuan, Shuai Yuan, Zhiyong Wu, and Lei Li. 2023b. How multilingual is multilingual llm? arXiv preprint arXiv:2311.09071. 





Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, Todor Mihaylov, Myle Ott, Sam Shleifer, Kurt Shuster, Daniel Simig, Punit Singh Koura, Anjali Sridhar, Tianlu Wang, and Luke Zettlemoyer. 2022. Opt: Open pre-trained transformer language models. Preprint, arXiv:2205.01068. 





Haiteng Zhao, Chang Ma, Qinyu Chen, and Zhi-Hong Deng. 2021. Domain adaptation via maximizing surrogate mutual information. arXiv preprint arXiv:2110.12184 





Haiteng Zhao, Chang Ma, Xinshuai Dong, Anh Tuan Luu, Zhi-Hong Deng, and Hanwang Zhang. 2022. Certified robustness against natural language attacks by causal intervention. In International Conference on Machine Learning, pages 26958–26970. PMLR. 





Rui Zheng, Bao Rong, Yuhao Zhou, Di Liang, Sirui Wang, Wei Wu, Tao Gui, Qi Zhang, and Xuanjing Huang. 2022. Robust lottery tickets for pre-trained language models. In Proceedings ofthe 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2211–2224, Dublin, Ireland. Association for Computational Linguistics. 





Wenhao Zhu, Yunzhe Lv, Qingxiu Dong, Fei Yuan, Jingjing Xu, Shujian Huang, Lingpeng Kong, Jiajun Chen, and Lei Li. 2023. Extrapolating large language models to non-english by aligning languages. arXiv preprint arXiv:2308.04948. 



## Acknowledgments

Authors of this paper would like to thank Zixian Huang for giving many helpful comments on previous versions of this paper. We deeply express our gratitude to Shanghai AI Laboratory. This work is partially supported by the National Key R&D Program of China (NO.2022ZD01601022030). 

## A Limitations and Broader Impacts

Limitations. In the scope of this research, our proposed KS-Lottery mainly targets multilingual research and has been thoroughly validated in bilingual translation tasks. However, we believe that our certified theoretical framework is generalizable to other types of tasks, and we leave these explorations for future work. 

Broader Impacts. This paper presents work whose goal is to advance the field of machine learning and multilingual research. We do not anticipate it will cause negative societal impacts, such as potential malicious or unintended uses. 

## B Proofs

Theorem 3. (Certified Winning Tickets) Let $f ( \cdot , \pmb { \theta } ^ { a } , \pmb { \theta } ^ { b } ) : \mathbb { R } ^ { d }  \mathcal { V }$ be any random or deterministic function, and let g be defined as in Equation 1. For any input x, suppose $c _ { A } \in \mathcal { V } , \underline { { p _ { A } } } , \overline { { p _ { B } } } \in [ 0 , 1 ]$ satisfies 

$$
\mathbb {P} \left(f (x, \widetilde {\boldsymbol {\theta}} ^ {a}, \widetilde {\boldsymbol {\theta}} ^ {b}) = c _ {A}\right) \geq \underline {{p _ {A}}} \geq \overline {{p _ {B}}} \geq \max _ {c \neq c _ {A}} \mathbb {P} (f (x +
$$

If the set of parameters $\theta ^ { b }$ satisfies 

$$
D (\widetilde {\theta} _ {i} ^ {b}, \theta_ {i} ^ {b}) <   \tau (\alpha) <   \frac {\underline {{p _ {A}}} - \overline {{p _ {B}}}}{2},
$$

for all $i \in V _ { b }$ , the classifier partial-tuned on parameters $\pmb { \theta } ^ { a }$ always return class $c _ { A } .$ , i.e. $g ( x , \widetilde { \pmb { \theta } } _ { a } ^ { E ^ { - } } , \pmb { \theta } _ { b } ^ { E } ) =$ $c _ { A }$ 

Proof Sketch Fix an input x, we study the change of prediction w.r.t. change in the distribution of $\pmb { \theta } ^ { b }$ . Let $A = \{ \pmb \theta | g ( X , \widetilde { \pmb \theta } ^ { a } , \pmb \theta ) = c _ { A } \}$ $B \ = \ \{ \pmb \theta | g ( x , \widetilde { \pmb \theta } ^ { a } , \pmb \theta ) \ \neq \ c _ { A } \}$ . Using definition, $\mathbb { P } ( \widetilde { \pmb { \theta } } ^ { b } \in \mathcal { A } ) \ \geq \ p _ { A } , \mathbb { P } ( \widetilde { \pmb { \theta } } ^ { b } \in \mathcal { B } ) \ \leq \ \overline { { p _ { B } } }$ . Since $D ( \widetilde { \pmb { \theta } } ^ { b } , \pmb { \theta } ^ { b } ) = \operatorname* { m a x } _ { i } D ( \widetilde { \pmb { \theta } } _ { i } ^ { b } , \pmb { \theta } _ { i } ^ { b } ) \leq \tau ( \alpha )$ , the minimum overlapping cumulative probability $\mathbb { P } ( \widetilde { \pmb { \theta } } ^ { b } \in$ $S \cap \theta ^ { b } \in S ) \geq 1 - \tau ( \alpha )$ , where S is any contour set $S = \{ \pmb \theta | | \pmb \theta - \widetilde { \mu } ^ { b } | \leq s , s \geq 0 \}$ 

Now we compare the probability that $f ( x , \widetilde { \pmb { \theta } } ^ { a } , \pmb { \theta } ^ { b } )$ predicts $c _ { A }$ or other classes. The probability that the partial-tuned f predicts $c _ { A }$ is 

$$
\begin{array}{l} \mathbb {P} (\boldsymbol {\theta} ^ {b} \in A) \geq \mathbb {P} (\boldsymbol {\theta} ^ {b} \in A \cap \boldsymbol {\theta} ^ {b} \in S) \\ \qquad \qquad \qquad \geq \mathbb {P} \left[ (\widetilde {\boldsymbol {\theta}} ^ {b} \in S \cap \boldsymbol {\theta} ^ {b} \in S) - (\widetilde {\boldsymbol {\theta}} ^ {b} \in S \cap \widetilde {\boldsymbol {\theta}} ^ {b} \not \in A) \right] \\ \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \geq (1 - \tau (\alpha)) - (1 - p _ {A}) = p _ {A} - \tau (\alpha) \end{array}
$$

Similarly, we can prove that $\mathbb { P } ( \pmb { \theta } ^ { b } \in B ) \le \overline { { p _ { B } } } +$ $\tau ( \alpha )$ . Thus when $\begin{array} { r } { \tau ( \alpha ) < \frac { p _ { A } - \overline { { p _ { B } } } } { 2 } } \end{array}$ and we partialtune all parameters $\pmb { \theta } ^ { a }$ that fails to pass the KStest, then the partial-tuned model $g ( x , \widetilde { \pmb { \theta } } _ { a } ^ { E } , \pmb { \theta } _ { b } ^ { E } ) =$ $c _ { A } .$ , i.e. always give the same prediction as the embedding fully-tuned model. □ 

## C More Analysis


Table 7: Certified experiment under Partial Tuning setting w.r.t significance level α. When α = 1, all token embeddings are chosen during tuning and thus certified accuracy equals to prediction accuracy.


<table><tr><td><eq>\alpha</eq></td><td>Setting</td><td>en→ca</td><td>en→bs</td><td>en→da</td><td>en→no</td><td>en→pt</td><td>en→ro</td><td>en→es</td><td>en→de</td></tr><tr><td rowspan="2">0</td><td>Certified</td><td>0.5649</td><td>0.4592</td><td>0.5384</td><td>0.4083</td><td>0.5959</td><td>0.5416</td><td>0.5108</td><td>0.5505</td></tr><tr><td>Prediction</td><td>0.7765</td><td>0.7040</td><td>0.7544</td><td>0.6476</td><td>0.7912</td><td>0.7555</td><td>0.7191</td><td>0.7644</td></tr><tr><td rowspan="2">0.05</td><td>Certified</td><td>0.7722</td><td>0.7312</td><td>0.7679</td><td>0.6484</td><td>0.7999</td><td>0.7714</td><td>0.7177</td><td>0.7313</td></tr><tr><td>Prediction</td><td>0.7887</td><td>0.7494</td><td>0.7822</td><td>0.6700</td><td>0.8156</td><td>0.7852</td><td>0.7372</td><td>0.7537</td></tr><tr><td rowspan="2">0.1</td><td>Certified</td><td>0.7752</td><td>0.7325</td><td>0.7700</td><td>0.6504</td><td>0.8006</td><td>0.7736</td><td>0.7197</td><td>0.7347</td></tr><tr><td>Prediction</td><td>0.7885</td><td>0.7494</td><td>0.7836</td><td>0.6693</td><td>0.8143</td><td>0.7855</td><td>0.7375</td><td>0.7551</td></tr><tr><td rowspan="2">0.5</td><td>Certified</td><td>0.7797</td><td>0.7375</td><td>0.7770</td><td>0.6579</td><td>0.8042</td><td>0.7765</td><td>0.7240</td><td>0.7482</td></tr><tr><td>Prediction</td><td>0.7897</td><td>0.7492</td><td>0.7864</td><td>0.6715</td><td>0.8143</td><td>0.7847</td><td>0.7356</td><td>0.7619</td></tr><tr><td rowspan="2">1</td><td>Certified</td><td>0.7846</td><td>0.7491</td><td>0.7817</td><td>0.6685</td><td>0.8097</td><td>0.7795</td><td>0.7349</td><td>0.7525</td></tr><tr><td>Prediction</td><td>0.7846</td><td>0.7491</td><td>0.7817</td><td>0.6685</td><td>0.8097</td><td>0.7795</td><td>0.7349</td><td>0.7525</td></tr></table>

The observed trend in the value of α suggests the presence of an optimal size for the set of selective tokens. In Figure 10, we examined the translation performance on the Flores-101 dataset under varying thresholds. It’s important to note that by default, the threshold creates an interval that is closed on the left and open on the right. Our observations indicate that even a minimal threshold is capable of preserving approximately 80% of the performance that is achieved through comprehensive embedding fine-tuning, as highlighted in green. Interestingly, each plot exhibits the same pattern characterized by an initial increase in performance, followed by a subsequent decline. This pattern suggests the existence of an optimal point that maximizes performance. 

Another Selective Methods There are five commonly used methods for determining parameters: Cos-Similarity (cos), Absolute Value (absolute), Related Value (relative), Related Ratio (ratio), and KL Divergence (KL). Each method emphasizes different aspects and requires a heuristic value p for selection. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/622691a6-a1f2-40b5-bee7-743c94c4d403/be0f4e08620291d1771fe6c6b5c3f24196611cb3478d655d9d1b5320c999c55a.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/622691a6-a1f2-40b5-bee7-743c94c4d403/0e6054e2fb5901fbeb416b7f8129ead9911b5bc90740047f620a0992e9a2b1f5.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/622691a6-a1f2-40b5-bee7-743c94c4d403/8f6a5e29a0cd8b97fe9a5ff508bd0fe7cccd0228d5cee6b8237d0197bf28442f.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/622691a6-a1f2-40b5-bee7-743c94c4d403/a9d3c3a1517927b53e04affcc689dc3353b7737c18b519ce062b84d224242ec1.jpg)



Figure 9: Certified experiment under Partial Transfer setting. A: Estimation of τ(α) w.r.t different α values by running KS-Test between the distribution of LLaMA-7b embedding and fine-tuned embedding on different datasets. B C D: Comparison between Certified Accuracy and Empirical Prediction Accuracy w.r.t. different α values on three datasets.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/622691a6-a1f2-40b5-bee7-743c94c4d403/73bc755a37e0d4bdce1ac54abce0c9660bd3c816ea655c1702dd22000947fb06.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/622691a6-a1f2-40b5-bee7-743c94c4d403/aa58b9adf638474df686a6f2b89dc37730edde500834c7ea8d02a8092571688b.jpg)



Figure 10: The translation performance of updated models with replacement method on Flores-101 with different token numbers. We observe the minimal threshold is sufficient to maintain approximately 80% of the performance achieved through embedding tuning (marked in orange). Meanwhile, for each plot, there exists a trend of initial increase followed by a decline, with the presence of an optimum point.


Cos-Similarity: This method determines the similarity between $\theta _ { i } ^ { E }$ and $\widetilde { \theta } _ { i } ^ { E }$ from the perspective of vector dot product. A lower cos-similarity suggests that it is the target of fine-tuning adjustment. 

Absolute Value: This is calculate as $| \widetilde { \theta } _ { i } ^ { E } - \theta _ { i } ^ { E } |$ which is concerned with the magnitude of the change. A larger value indicates that it is the target of fine-tuning. 

Related Value: This is calculated as $\widetilde { \theta } _ { i } ^ { E } / \theta _ { i } ^ { E }$ , which focuses on the rate of change before and after finetuning. A larger ratio suggested that it is the target. 

Related Ratio: This method considers the initial value and calculates the result as $( \widetilde { \theta } _ { i } ^ { E } - \theta _ { i } ^ { E } ) / \theta _ { i } ^ { E }$ . A larger ratio indicates that it is the key adjustment in fine-tuning. 

KL Divergence: This is a statistical distance, measuring the distance between $\widetilde { \theta } _ { i } ^ { E }$ and $\theta _ { i } ^ { E }$ , with a focus on the distribution itself. 

<table><tr><td>en→ca</td><td>en→es</td><td>en→ro</td><td>en→da</td><td>en→de</td><td>en→pt</td></tr><tr><td>13</td><td>13</td><td>13</td><td>13</td><td>13</td><td>13</td></tr><tr><td>263</td><td>262</td><td>278</td><td>263</td><td>263</td><td>263</td></tr><tr><td>278</td><td>263</td><td>297</td><td>278</td><td>278</td><td>278</td></tr><tr><td>297</td><td>278</td><td>304</td><td>297</td><td>297</td><td>297</td></tr><tr><td>304</td><td>297</td><td>310</td><td>304</td><td>304</td><td>304</td></tr><tr><td>310</td><td>304</td><td>322</td><td>310</td><td>310</td><td>310</td></tr><tr><td>322</td><td>310</td><td>338</td><td>322</td><td>322</td><td>322</td></tr><tr><td>338</td><td>313</td><td>366</td><td>29871</td><td>338</td><td>338</td></tr><tr><td>376</td><td>322</td><td>393</td><td>29889</td><td>363</td><td>363</td></tr><tr><td>393</td><td>363</td><td>29871</td><td>29892</td><td>29871</td><td>411</td></tr><tr><td>411</td><td>393</td><td>29889</td><td>29896</td><td>29889</td><td>29871</td></tr><tr><td>29871</td><td>411</td><td>29892</td><td>29900</td><td>29892</td><td>29889</td></tr><tr><td>29889</td><td>29871</td><td>29896</td><td></td><td>29896</td><td>29892</td></tr><tr><td>29892</td><td>29889</td><td>29900</td><td></td><td>29901</td><td>29896</td></tr><tr><td>29896</td><td>29892</td><td>29901</td><td></td><td>29915</td><td>29900</td></tr><tr><td>29900</td><td>29896</td><td></td><td></td><td></td><td>29901</td></tr><tr><td>29901</td><td>29897</td><td></td><td></td><td></td><td>29915</td></tr><tr><td>29949</td><td>29901</td><td></td><td></td><td></td><td></td></tr></table>


Table 8: Across various languages, the winning tickets share some common pattern.


## C.1 When Fine-tuning on different languages, the winning tickets contain the same tokens.

The winning tickets across various languages overlap with a subset of tokens, as shown in Table 8, {index $\mathrm { i d } = 1 3 \ ( \mathrm { t o k e n } { = } " \backslash \mathrm { n " } )$ , index id=278 (token=’the’), index id=29871 (token=’∼’), and so on}, which are frequently used and common across various languages. 

## D AI Assistants

We just use AI assistants for our sentence-level polishing. 

## E Used Scientific Artifacts

Below lists scientific artifacts that are used in our work. For the sake of ethic, our use of these artifacts is consistent with their intended use. 

• Stanford Alpaca (Apache-2.0 license), a project that aims to build and share an instruction-following LLaMA model. 

• Lego-MT (MIT license), a dataset for machine translation. 

• Transformers (Apache-2.0 license), a framework that provides thousands of pretrained models to perform tasks on different modalities such as text, vision, and audio. 