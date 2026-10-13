# BEYOND RAW DETECTION SCORES: MARKOV-INFORMED CALIBRATION FOR BOOSTING MACHINE-GENERATED TEXT DETECTION

Chenwang Wu $^{1}$ Yiu-ming Cheung $^{1*}$ Shuhai Zhang $^{2}$ Bo Han $^{1}$ Defu Lian $^{3}$ 

$^{1}$ Department of Computer Science, Hong Kong Baptist University, Hong Kong, China $^{2}$ School of Software Engineering, South China University of Technology, Guangzhou, China $^{3}$ School of Computer Science, University of Science and Technology of China, Hefei, China  
{cscwwu, ymc, bhanml}@comp.hkbu.edu.hk, shuhaizhangshz@gmail.com, liandefu@ustc.edu.cn 

## ABSTRACT

While machine-generated texts (MGTs) offer great convenience, they also pose risks such as disinformation and phishing, highlighting the need for reliable detection. Metric-based methods, which extract statistically distinguishable features of MGTs, are often more practical than complex model-based methods that are prone to overfitting. Given their diverse designs, we first place representative metric-based methods within a unified framework, enabling a clear assessment of their advantages and limitations. Our analysis identifies a core challenge across these methods: the token-level detection score is easily biased by the inherent randomness of the MGTs generation process. To address this, we theoretically and empirically reveal two relationships of context detection scores that may aid calibration: Neighbor Similarity and Initial Instability. We then propose a Markov-informed score calibration strategy that models these relationships using Markov random fields, and implements it as a lightweight component via a mean-field approximation, allowing our method to be seamlessly integrated into existing detectors. Extensive experiments in various real-world scenarios, such as cross-LLM and paraphrasing attacks, demonstrate significant gains over baselines with negligible computational overhead. The code is available at https://github.com/tmlr-group/MRF_Calibration. 

## 1 INTRODUCTION

In recent years, generative AI, represented by large language models (LLMs) (Achiam et al., 2023; Radford et al., 2019), has advanced rapidly, and the machine-generated texts (MGTs) they produce often match human writing in fluency, coherence, and diversity. While this technological breakthrough offers immense opportunities, it has also triggered widespread societal concerns, such as the spread of disinformation (Vykopal et al., 2024), the violation of intellectual property rights (Yu et al., 2023b), and phishing attacks (Hong, 2012). Therefore, the research and development of MGT detection technologies hold significant theoretical and practical value in uncovering the distinct patterns of generated text and ensuring a trustworthy AI environment. 

An effective detection method is to identify LLM's watermarks (Hou et al., 2024), but this requires injecting watermarks into the LLM, which is often impractical due to high access permissions. Therefore, passive detection methods, including model-based and metric-based methods, have garnered significant attention. Model-based methods, which use a set of human- and machine-generated texts to train a binary classifier, such as OpenAI detector (Solaiman et al., 2019), ChatGPT detector (Guo et al., 2023), SeqXGPT (Wang et al., 2023), and CoCo (Liu et al., 2022). However, such models are often too complex, leading to overfitting to the training data. Instead, metric-based methods exploit the inherent statistical biases of LLM to discriminate MGTs, which is model-agnostic and has better generalization properties. These methods use metrics such as log-likelihood, log-rank, and entropy. Furthermore, methods such as DetectGPT (Mitchell et al., 2023), DNA-GPT (Yang et al., 2024), and SimLLM (Nguyen-Son et al., 2024) detect MGTs by comparing the differences between a given text and a perturbed, regenerated, or continued text from an alternative model. 

Obviously, metric-based methods exhibit diverse designs. Therefore, this paper first systematically examines several representative approaches, including Log-Likelihood (Solaiman et al., 2019), Log-Rank (Mitchell et al., 2023), Entropy (Gehrmann et al., 2019), DetectGPT (Mitchell et al., 2023), Fast-DetectGPT (Bao et al., 2024), and DNA-GPT (Yang et al., 2024), and situates them within a unified framework (Section 2). Our analysis reveals that they share a threshold-based detection criterion, with some commonalities, such as the inclusion of auxiliary data (e.g., perturbed texts). This offers a theoretical basis for understanding their core mechanisms, strengths, and limitations. 

Based on the unified framework, we summarize the core challenge of metric-based methods (Section 2): the imprecision of token-level detection scores. Specifically, since these methods make decisions based on a threshold, their effectiveness is directly tied to score precision. However, randomness introduced by the LLM sampling mechanism can violate their underlying assumptions, leading to biased scores with low discrimination, as shown in "w/o refine" in Fig. 1 (more results can be found in Appendix E.1). Moreover, they tend to derive an overall detection score by naively aggregating token-level scores, failing to correct the underlying imprecision. Therefore, calibrating the token-level detection score is essential for improving overall detection performance. 

Given that detection scores are tied to tokens and the LLM generation process induces dependencies among tokens (Achiam et al., 2023), context tokens' detection scores may have relationships that are easy to overlook. Revealing and modeling these relationships may help calibrate these scores. Accordingly, we theoretically and empirically reveal two relationships among detection scores of context tokens (Section 3): Neighbor Similarity, where adjacent tokens exhibit similar detection scores, 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/9742cefc153c7ffb1dc3979193ba53ff2517542a586858fd17e7fdc09f8385f4.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/687d920436f8df14d527876a4de920fb50ceba61fefc109b157b9eb592351d37.jpg)



Figure 1: Distribution of token scores obtained by the DetectGPT method without and with score calibration in the Essay dataset. The proposed calibration method enhances the discriminative nature of the token scores.


and Initial Instability, where the detection scores of initial tokens are unstable. 

Finally, building on these two relationships, we propose a Markov-informed score calibration method to enhance MGT detection (Section 4). Our method models the identified relationships through Markov random fields and, via a mean-field approximation, implements it as a lightweight iterative neural network. As shown by the more discriminative detection scores in the "w refine" method in Fig. 1, the proposed method boosts the discriminative nature of the scores. Notably, our method can be seamlessly stacked on top of existing detectors without architectural changes, providing flexibility. Compared with complex model-based approaches, our method introduces only a negligible $2 \times 2$ parameterization, making its computational delay negligible and less prone to overfitting. Extensive experiments demonstrate our method's enhanced effectiveness. Our contributions can be summarized as follows: 

- We view existing metric-based detection methods through a unified lens, which facilitates precise comparison and enables potential improvements. 

- We theoretically and empirically demonstrate that token-level detection scores exhibit neighbor similarity and initial instability, offering avenues for improved detection. 

- We propose a Markov-informed score calibration method and, via a mean-field approximation, implement it as a lightweight component that can be seamlessly integrated into existing detectors to further unlock their potential. 

- We conduct extensive experiments across three datasets to demonstrate superior performance in diverse scenarios, including cross-LLM, cross-domain, mixed-text, and paraphrase attacks. 


Table 1: Comparing existing metric-based methods from a unified view. Here, s is the text to be detected containing N tokens, $s'$ is the perturbed text generated by DetectGPT, $\tilde{s}$ and $\hat{s}$ are the regenerated texts of Fast-DetectGPT and DNA-GPT, respectively. Function $\mu(\cdot)$ and $\sigma(\cdot)$ represent the mean and standard deviation of the given set, respectively.


<table><tr><td>Method</td><td>Data</td><td>Token-level Score</td><td>Score Aggregation</td><td>Detection</td></tr><tr><td>Log-likelihood</td><td>s</td><td><eq>\log p(s_t|_{s&lt;t})</eq></td><td><eq>\frac{1}{N-1}\sum_{t=2}^{N}\log p(s_t|_{s&lt;t})</eq></td><td>score &gt; <eq>\epsilon</eq></td></tr><tr><td>Log-Rank</td><td>s</td><td><eq>\text{rank}(p(s_t|_{s&lt;t}))</eq></td><td><eq>\frac{1}{N-1}\sum_{t=2}^{N}\text{rank}(p(s_t|_{s&lt;t}))</eq></td><td>score &gt; <eq>\epsilon</eq></td></tr><tr><td>Entropy</td><td>s</td><td><eq>\sum_{v\in V}p(v|_{s&lt;t})\log p(v|_{s&lt;t})</eq></td><td><eq>-\frac{1}{N-1}\sum_{t=2}^{N}\sum_{v\in V}p(v|_{s&lt;t})\log p(v|_{s&lt;t})</eq></td><td>score &gt; <eq>\epsilon</eq></td></tr><tr><td>DetectGPT</td><td><eq>\{s,s_1&#x27;,s_2&#x27;,...,s_n&#x27;\}</eq></td><td><eq>p(s_t|_{s&lt;t})</eq></td><td><eq>\frac{\frac{1}{N-1}\sum_{t=2}^{N}p(s_t|_{s&lt;t})-\mu(\{\frac{1}{N-1}\sum_{t=2}^{N}p(s_i,t&#x27;s_i,t&#x27;)\}_{i})}{\sigma(\{\frac{1}{N-1}\sum_{t=2}^{N}p(s_i,t&#x27;s_i,t&#x27;)\}_{i})}</eq></td><td>score &gt; <eq>\epsilon</eq></td></tr><tr><td>Fast-DetectGPT</td><td><eq>\{s,\tilde{s}_1,...,\tilde{s}_n\}</eq></td><td><eq>p(s_t|_{s&lt;t})</eq></td><td><eq>\frac{\frac{1}{N-1}\sum_{t=2}^{N}p(s_t|_{s&lt;t})-\mu(\{\frac{1}{N-1}\sum_{t=2}^{N}p(\tilde{s}_i,t|s,\tilde{s}_i,t&#x27;)\}_{i})}{\sigma(\{\frac{1}{N-1}\sum_{t=2}^{N}p(\tilde{s}_i,t|s,\tilde{s}_i,t&#x27;)\}_{i})}</eq></td><td>score &gt; <eq>\epsilon</eq></td></tr><tr><td>DNA-GPT</td><td><eq>\{s,\tilde{s}_1,...,\tilde{s}_n\}</eq></td><td><eq>p(s_t|_{s&lt;t})</eq></td><td><eq>\frac{1}{N-1}\sum_{t=2}^{N}\log p(s_t|_{s_{1:t-1}})-\frac{1}{n}\sum_{i=1}^{n}\frac{1}{N-t}\sum_{t=2}^{N}\log p(\tilde{s}_i,t|\tilde{s}_{i,1:t-1})</eq></td><td>score &gt; <eq>\epsilon</eq></td></tr></table>

## 2 A UNIFIED PERSPECTIVE ON METRIC-BASED DETECTION

Although model-based methods have shown competitive potential in specific domains, they are often too complex, leading to a tendency to overfit their training data. This limitation requires them to retrain or fine-tune for newly released LLMs, which hinders their generalizability. In contrast, metric-based methods extract discriminative features from MGT, and their model-agnostic nature provides superior generalization potential. Given the diverse implementations of representative metric-based methods such as Log-Likelihood (Solaiman et al., 2019), Log-Rank (Gehrmann et al., 2019), Entropy (Gehrmann et al., 2019), DetectGPT (Mitchell et al., 2023), Fast-DetectGPT (Bao et al., 2024), and DNA-GPT (Yang et al., 2024), we first provide a systematic examination of them from a unified perspective. This facilitates a deeper understanding of their mechanisms and allows for a fair comparison of their strengths and weaknesses. As illustrated in Table 1, we compare these methods across data, score aggregation, and detection dimensions. Note that their diverse core metric designs are not discussed here. This is because we aim to design a general enhancement framework decoupled from specific detectors, requiring us to start from possible commonalities rather than diverse designs. 

- Data. Log-likelihood, Log-Rank, and Entropy are computationally efficient as they rely solely on the original input text $s$ . However, the detection error based on a single text may be large, because the randomness inherent in the LLM sampling mechanism may cause the MGT to deviate from these methods' underlying assumptions, e.g., Log-Rank assumes that the generated tokens have high rankings. In contrast, DetectGPT, Fast-DetectGPT, and DNA-GPT incorporate multiple perturbed (i.e., $s'$ ) or regenerated (i.e., $\tilde{s}$ and $\hat{s}$ ) samples, which mitigates the errors caused by randomness. However, this comes at the cost of increased computational overhead compared to single-text-based methods. 

- Score Aggregation. Although these methods appear to calculate scores differently, they all tend to directly aggregate token scores to obtain the final text score, typically through summation. As discussed, the randomness introduced by the LLM generation process may cause token-level scores to be biased. Therefore, the direct aggregation of these potentially imprecise token scores may result in an inaccurate final detection score. 

- Detection. These methods employ threshold-based detection mechanisms, whose effectiveness relies heavily on the accuracy of their calculated scores. Including uncalibrated, high-noise scores in threshold-based decision-making may lead to poor performance. 

In summary, to enhance detection, existing methods incorporate more textual information (e.g., regenerated texts in DetectGPT and Fast-DetectGPT) and use different score calculation strategies (e.g., the Likelihood difference between the detected and regenerated text in DNAGPT). However, as we discussed, they fail to address the underlying token-level errors caused by inherent randomness, limiting their detection potential. Considering that detection scores are tied to tokens and the LLMs' generative mechanism induces dependencies among tokens, revealing and modeling the relationships between tokens may help correct score errors and thus improve detection effectiveness. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/2116e232dbe7d6ef82e6f70ba5fac38114273cc31443d0c3a7f880b7901826ac.jpg)



(a) Log-Likelihood Score


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/0066bfcbad40c983057e819d1932d269f524b946d8f0e7c64fc1b0cd35c82a58.jpg)



(b) Log-Rank Score


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/1369e2477c87e10ca7471c21eabeccc9cb22b848d01cabb615bfc9cde327e2a2.jpg)



Figure 2: The detection score distances (Mean Absolute Difference) of neighbors at different hops in the Essay dataset. Log-likelihood and Log-Rank score are used.



(a) Log-Likelihood Score


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/dcc2b4e716fa371f332ba853934bb344226457706b56a2f7539f14c38989812d.jpg)



(b) Log-Rank Score



Figure 3: The detection score distances (Mean Absolute Difference) of 1-hop neighbors at different normalized relative positions in Essay. Log-likelihood and Log-Rank score are used.


## 3 RELATION BETWEEN CONTEXTUAL TOKEN-LEVEL DETECTION SCORES

To understand the relationship between context tokens' detection scores, following existing work (Liu et al., 2023), we consider the token generation process of a simplified model: a single-layer transformer model with single-head attention: 

$$
x _ {t + 1} = \mathcal {F} \left(a _ {t}\right), \text {where} a _ {t} = \operatorname{softmax} \left(1 / t \cdot x _ {t} W _ {Q} W _ {K} ^ {\top} X _ {t - 1} ^ {\top}\right) X _ {t - 1} W _ {V} W _ {O}.\tag{1}
$$

$x_{t} \in R^{1 \times d}$ is the embedding of token $s_{t}$ , and $d$ denotes the embedding dimension. The matrix $X_{t-1} \in \mathbb{R}^{(t-1) \times d}$ is stacked by the embeddings $x_{1}, \ldots, x_{t-1}$ , where the $j$ -th row is $x_{j}$ . $W_{Q}, W_{K}, W_{V} \in \mathbb{R}^{d \times d}$ and $W_{O} \in \mathbb{R}^{d \times d}$ are the attention weights. Following the attention block, an MLP block, denoted as $\mathcal{F}: \mathbb{R}^{1 \times d} \to \mathbb{R}^{1 \times d}$ , is applied, and it is a two-layer network with skip connections: 

$$
\mathcal {F} (x) = x + W _ {2} \operatorname{relu} \left(W _ {1} x\right).
$$

As shown in Table 1, the detection score of token $s_t$ is usually the function of $s_t$ (i.e., $x_i$ ), and $x_i$ is related to the attention scores $\alpha_{t-1} = \text{softmax}\left(1/t \cdot x_t W_Q W_K^\top X_{t-1}^\top\right)$ in Eq. (1). The following theorem will reveal the relationship between attention scores, which in turn help us understand the relationship between detection scores of context tokens. 

Theorem 1. Let $\lambda_K, \lambda_Q, \lambda_V, \lambda_O$ be the largest singular values of parameters $W_K, W_Q, W_V, W_O$ , respectively, and let $W = W_V W_O W_Q W_K^\top$ . For the transformer defined in Eq. (1), assuming normalized inputs ( $\|x_t\|_2 = 1$ for all $t$ ) and constants $c, \epsilon > 0$ , consider $a_t x_{t+1}^\top \geq (1 - \delta) \|a_t\|_2$ with $\delta \leq \left(\frac{c\epsilon}{\lambda_Q \lambda_K \lambda_V \lambda_O}\right)^2$ . If $x_\ell$ satisfies $x_\ell W x_\ell^\top \geq c$ and $x_\ell W x_\ell \geq \epsilon^{-1} \max_{j \in [\ell], j \neq \ell} x_j W x_\ell^\top$ , then 

$$
\alpha_ {t + 1, l} \leq \frac {\exp (C _ {l} \cdot \alpha_ {t , l} + \eta)}{\exp (C _ {l} \cdot \alpha_ {t , l} + \eta) + \sum_ {j \neq l} \exp (C _ {j} \cdot \alpha_ {t , j} - \eta)},
$$

$$
\alpha_ {t + 1, l} \geq \frac {\exp (C _ {l} \cdot \alpha_ {t , l} - \eta)}{\exp (C _ {l} \cdot \alpha_ {t , l} - \eta) + \sum_ {j \neq l} \exp (C _ {j} \cdot \alpha_ {t , j} + \eta)},
$$

where 

$$
C _ {j} = \frac {x _ {j} W x _ {j} ^ {T}}{t | a _ {t} | _ {2}}, a n d \eta = \frac {(1 + \sqrt {2}) \epsilon x _ {j} W x _ {j} ^ {T}}{(t + 1) | a _ {t} | _ {2}}.
$$

The proof can be found in Appendix C. This theorem establishes the upper and lower bounds for the attention score at step $t + 1$ , which are determined by the attention scores at step t. In Transformer, the function mapping $a_{t} \to \log p(x_{t+1})$ is continuous, constraints on the dynamics of $a_{t}$ naturally propagate to the detection scores. Therefore, Theorem 1 can reveal two relationships in token-level detection scores: 

- Neighbor Similarity, where the detection scores of adjacent tokens in a sequence exhibit statistically lower variance compared to tokens that are far apart. This is because Theorem 1 creates a positive feedback loop analogous to simulated annealing (Kirkpatrick et al., 1983), where high scores at the current step lead to high scores at the next, and vice versa. That is, the attention (also token-level detection scores) cannot shift fast between steps. 

- Initial Instability, where the detection scores of initial tokens in a sequence are statistically unstable compared to the subsequent tokens (i.e., fluctuate greatly). This is because these bounds are closely tied to the current step $t$ . When $t$ is small (early position), $\eta$ and $C$ are large, allowing for dramatic fluctuations in $a_t$ . That is, the attention score (also token-level detection scores) is unstable in the initial generation process. 

Given that analyzing the dynamics of full-scale LLMs is mathematically intractable, we follow a practical setup Liu et al. (2023) and start with a simplified model, which provides theoretical motivation for the neighbor similarity and initial instability phenomena. We then empirically verify these two findings. 

First, to empirically validate the neighbor similarity property, we evaluated the distance (mean absolute difference) in detection scores across k hops (i.e., $\frac{1}{|S|}\frac{1}{N-K}\sum_{s\in S}\sum_{t=0}^{N-K}|score(s_{t})-score(s_{t+k})|$ , where S is the text set, and $score(s_{t})$ is provided in Table 1). As illustrated in Fig. 2 (more results can be found in Appendix E.2), there is a clear positive correlation between the detection score distance and the hop, and adjacent tokens have the highest detection score similarity; thereby providing empirical evidence for our theoretical finding. 

Second, to validate the initial instability property, we analyzed the distance in detection score between adjacent tokens at different percentage positions (i.e., $\frac{1}{S}\sum_{s\in S}|score(s_{t})-score(s_{t+1})|$ ). Fig. 3 illustrates that the score difference is substantially larger for tokens at the beginning of the text and progressively decreases, eventually stabilizing. More results can be found in Appendix E.2. Considering our established finding on neighbor similarity (i.e., adjacent scores should be highly similar), a high detection score difference at the sequence beginning indicates a significant instability in detection scores of initial tokens. 

## 4 MARKOV-INFORMED DETECTION SCORE CALIBRATION

Based on the revealed relationships, this section uses an MRF to capture them (Section 4.1) and adopts the mean field approximation to model the MRF model as a lightweight component stacked on existing detectors to calibrate detection scores (Section 4.2), thereby enhancing detection. 

## 4.1 MARKOV RANDOM FIELD FOR MGT DETECTION

We capture these two types of relationships by modeling the joint probability distribution of text's token detection scores through pairwise Markov Random Fields (pMRF). Specifically, for each token $s_t$ in text $s$ , we assign a binary random variable $y_{s_t}$ , where $y_{s_t} = 0$ and $y_{s_t} = 1$ indicate a human- or machine-generated token $^1$ , respectively, as measured by the detection score of the token. Let $y_s$ denote the label set for all tokens in text $s$ , the pMRF over these tokens can be formalized as a Gibbs distribution: $P(y_s) = \frac{1}{Z}\exp (-E(s,y_s))$ , where $Z$ is a normalizing constant and $E(s,y_s)$ is the energy function. Our objective is to maximize the posterior probability of the token labels $y_s$ by minimizing the global energy function $E(s,y_s)$ . The energy function typically consists of two components: the unary potential $\Psi_U$ and the pairwise potential $\Psi_P$ : 

$$
E (s, y _ {s}) = \sum_ {t = 1} ^ {N} \Psi_ {U} (s _ {t}, y _ {s _ {t}}) + \sum_ {t = 1} ^ {N} \sum_ {s _ {j} \in \mathcal {N} (s _ {t})} \Psi_ {P} (y _ {s _ {t}}, y _ {s _ {j}}),
$$

where $\mathcal{N}(s_t)$ denotes the adjacent tokens of token $s_t$ , i.e., $\mathcal{N}(s_t) = \{s_{t-1}, s_{t+1}\}$ (if existing). 

Unary potential $\Psi_{U}(s_{t},y_{s_{t}})$ quantifies the cost of assigning label $y_{s_{t}}$ to token $s_{t}$ . We let $\Psi_{U}(s_{t},y_{s_{t}}) = -\log p(s_{t})$ , where $p(s_{t})$ is the output probability from the original detector. For detectors without probability output, it is measured by the 0-1 normalized score of token $s_{t}$ . 

Pairwise potential $\Psi_{P}\left(y_{s_{t}},y_{s_{j}}\right)$ models the similarity in detection scores between adjacent tokens. A penalty is applied if two adjacent tokens are assigned different labels; otherwise, a reward is given. This enforces label smoothness and captures the neighbor similarity property: 

$$
\Psi_ {P} \left(y _ {s _ {t}}, y _ {s _ {j}}\right) = w \cdot (2 \cdot I (y _ {s _ {t}} \neq y _ {s _ {j}}) - 1),\tag{2}
$$

where $I(\cdot)$ is the indicator function, and the reward and penalty factor $w \geq 0$ . This implies an energy penalty of w when adjacent tokens have different labels; otherwise, the reward is -w. 

To model the initial instability property, we introduce a positional weighting function $\beta(t)$ in the binary potential. This function assigns lower weights to binary potentials at earlier positions, thereby mitigating the amplification of energy errors caused by unstable initial neighbor tokens. In this paper, we define the positional weighting function $\beta(t)$ as a Sigmoid function to ensure a smooth transition of weights, and the revised binary potential is then given by: 

$$
\Psi_ {P} \left(y _ {s _ {i}}, y _ {s _ {j}}\right) = \beta (j) \cdot w \cdot (2 \cdot I (y _ {s _ {i}} \neq y _ {s _ {j}}) - 1), \text {with} \beta (t) = \frac {1}{1 + e x p (- (t - t _ {0}))},\tag{3}
$$

where $t_{0}$ is the weighted center, effectively suppressing the pairwise potential of tokens before $t_{0}$ . 

## 4.2 MEAN FIELD APPROXIMATE IN MGT DETECTION

Given the MRF model, this subsection details how to model it as a lightweight component stacked on the original detector through mean field approximation theory, thereby enhancing detection. 

In the MRF posterior probability $P(y_{s}) = \frac{1}{Z} \exp(-E(s, y_{s}))$ , the partition function $Z = \sum_{y_{s}} \exp(-E(s, y_{s}))$ , which is obtained by adding over all possible combinations of $y_{s}$ . For a text with N tokens, there are $2^{N}$ combinations, making exact computation of $P(y_{s}|s)$ infeasible. Inspired by existing work (Deng et al., 2022), we employ mean-field theory for approximate inference. Its core idea is to use a simpler, factorized distribution $Q(y_{s}) = \prod_{t=1}^{N} Q_{s_{t}}(y_{s_{t}})$ to approximate the true joint distribution $P(y_{s})$ , achieved by minimizing the KL divergence between these two distributions: 

$$
\begin{array}{r l} & D (Q \| P) = \mathbb {E} _ {y _ {s} \sim Q} \left[ \log Q (y _ {s}) \right] - \mathbb {E} _ {y _ {s} \sim Q} \left[ \log P (y _ {s}) \right] \\ & \qquad = \sum_ {t = 1} ^ {N} \mathbb {E} _ {y _ {s _ {t}} \sim Q _ {s _ {t}}} \left[ \log Q _ {s _ {t}} (y _ {s _ {t}}) \right] + \mathbb {E} _ {y _ {s} \sim Q} \left[ E (s, y _ {s}) \right] + \log Z. \end{array}\tag{4}
$$

The first equation is the definition of KL divergence, and the second is obtained by substituting $P(y_{s})$ and $Q(y_{s})$ . Then, we define a Lagrangian composed of all terms involving $Q_{s_t}(y_{s_t})$ in $D(Q\| P)$ : 

$$
L _ {s _ {t}} (Q) = Q _ {s _ {t}} \left(y _ {s _ {t}}\right) \log Q _ {s _ {t}} \left(y _ {s _ {t}}\right) + \mathbb {E} _ {y _ {s} \sim Q} \left[ E \left(s, y _ {s}\right) \right] + \lambda (\sum_ {y _ {s _ {t}}} Q _ {s _ {t}} \left(y _ {s _ {t}}\right) - 1).\tag{5}
$$

Here, the term involving Lagrange multiplier $\lambda$ assures that $Q_{s_{i}}$ is a proper probability distribution. Now we take derivatives of Eq. (5) with respect to $Q_{s_{t}}(y_{s_{t}})$ and set the corresponding derivative to 0, then we get the optimal $Q_{s_{t}}(y_{s_{t}})$ : 

$$
Q _ {s _ {t}} (y _ {s _ {t}}) = \frac {1}{z} e x p (\Psi_ {U} (s _ {t}, y _ {s _ {t}}) - \sum_ {s _ {j} \in \mathcal {N} (s _ {t})} \mathbb {E} _ {y _ {s _ {j}} \sim Q _ {s _ {j}}} \left[ \Psi_ {P} (y _ {s _ {t}}, y _ {s _ {j}}) \right]).\tag{6}
$$

We can express the single token calculation in Eq. (6) as a matrix form of multiple token calculations, which involves three steps: 

- Unary potential calculation. Corresponding to the unary potential $\Psi_U(s_t, y_{s_t})$ of token $s_t$ being $-\log p(s_i)$ , we can express the unary potential of text $s$ in matrix form $-\log H$ , where the i-th row of $H$ corresponds to token $s_i$ 's prior probability and the two columns correspond to identity labels (HGT and MGT), that is, $H = [1 - p(s), p(s)]$ , where $p(s) = [p(s_1), ..., p(s_N)]^\top$ . 

- Pairwise potential calculation. For the revised pairwise potential (i.e., Eq. (3)), we can define the weighted adjacency matrix as $A^{corr}$ , where $A_{t,t+1}^{corr} = \beta(t+1)$ for all $t = 0, \ldots, N-2$ , $A_{t-1,t}^{corr} = \beta(t-1)$ for all $t = 1, \ldots, N-1$ , and 0 otherwise. Then the matrix form of the weighted pairwise potential is $A^{corr}Q\left[ \begin{array}{cc} -w & w \\ w & -w \end{array} \right]$ . This update strategy enforces that all relationships receive the same reward or penalty. However, the influence of MGT neighbors and HGT neighbors is intuitively different, so this kind of reward and punishment mechanism with the same weight will limit the expressive ability of MRF. To this end, we relax the weights for different relationships and get the pairwise potential as $A^{corr}Q(W_{mrf} \odot \left[ \begin{array}{cc} -1 & 1 \\ 1 & -1 \end{array} \right])$ , where $W_{mrf} \in \mathbb{R}_+^{2 \times 2}$ . 

- Normalization. The operator $\frac{1}{z} exp(\cdot)$ in Eq. (6) can be naturally modeled as Softmax function. 

In summary, we get the following update rule for tokens' detection scores: 

$$
Q = \text { softmax } \left(\log H - A ^ {c o r r} Q \left(W _ {m r f} \odot \left[ \begin{array}{c c} - 1 & 1 \\ 1 & - 1 \end{array} \right]\right)\right).\tag{7}
$$

Notably, Eq. (7) shows that the computation of $Q$ relies on $Q$ itself; hence, iterative computation is required. Initializing the initial $Q$ to $H$ , we can get the iterative version as follows: 

$$
Q ^ {t} = \text { softmax } \left(\log Q ^ {t - 1} - A ^ {c o r r} Q ^ {t - 1} \left(W _ {m r f} \odot \left[ \begin{array}{c c} - 1 & 1 \\ 1 & - 1 \end{array} \right]\right)\right), \text {   where   } Q ^ {0} = H.\tag{8}
$$

In addition to using position weight function $\beta(t)$ in pairwise potential to reduce the impact of initial unstable scores, we also use this position weight function on the final calibrated scores $Q^{T}$ of T iterations to reduce their impact on detection: 

$$
Q _ {f i n a l} = [ \beta (1),..., \beta (N) ] \odot Q ^ {T}.\tag{9}
$$

The MRF-informed calibration component is then directly stacked on the original detector to calibrate the token detection scores. Specifically, if the de-

Algorithm 1 Markov-informed Enhancement Framework
1: Input: Text s to be detected, the original detector $f_{1} \circ f_{2}$ , iteration steps T, MRF weights W.
2: Construct $A^{corr}$ based on s.
3: Get each token $s_{i}$ 's detection score from the detection score calculation module $f_{1}$ , and set $Q^{0} = H$ .
4: for t = 0 to T - 1 do
5: Update Q according to Eq. (8).
6: end for
7: Calculate $Q_{final}$ according to Eq. (9).
8: Return detection score $f_{2}(Q_{final})$ of text s. 

tector is formalized as a combination of the detection score calculation module $f_{1}$ and the detection module $f_{2}$ , i.e., $f(s) = f_{1} \circ f_{2}(s)$ , the MRF-informed component of Eq. (9) is defined as $f_{mrf}$ , then the enhanced detector is $f_{enh.(s)} = f_{1} \circ f_{mrf} \circ f_{2}(s)$ . The complete inference process is shown in Alg. 1. To learn weights W, we use supervised training: $L = -\sum_{s \in D_{train}} (Y_s \log f_{enh.(s)} + (1 - Y_s) \log (1 - f_{enh.(s)}))$ . 

Computational Complexity. The MRF layer can be computed using sparse-dense matrix multiplication. For a text containing N tokens, the number of iterations is T, resulting in a computational complexity of $\mathcal{O}(NT)$ , which is negligible for calculating the detection score. We will also empirically verify the efficiency of our method in Appendix E.15. 

## 5 EXPERIMENTS

Dataset. We conducted experiments on three widely used public datasets: Essay (Verma et al., 2024), Reuters (Verma et al., 2024), DetectRL (Wu et al., 2024), and TruthfulQA (Lin et al., 2022). The Essay and Reuters datasets collect machine text generated by GPT4All, ChatGPT, ChatGPT-turbo, ChatGLM, Dolly, and Claude. The TruthfulQA dataset collects machine text from GPT4, ChatGPT-turbo, ChatGLM, Dolly, ChatGPT, and StableLM. The DetectRL dataset not only includes pure machine-generated text by Google-PaLM, ChatGPT, and Llama-2-70b, but its unique mixed, paraphrased, and cross-domain texts also allow us to more comprehensively evaluate the model's performance in complex real-world scenarios. For a complete description of the datasets, please refer to Appendix D.1. 

Baselines. We select the following metric-based methods for comparison and enhancement: Log-Likelihood (Likelihood) (Solaiman et al., 2019), Log-Rank (Gehrmann et al., 2019), Entropy (Gehrmann et al., 2019), DetectGPT (Mitchell et al., 2023), Fast-DetectGPT (FastGPT) (Bao et al., 2024), DNA-GPT (Yang et al., 2024), Repreguard (Chen et al., 2025), Lastde (Xu et al., 2025), FourierGPT (Xu et al., 2024), and Binoculars Hans et al. (2024). The versions equipped with the proposed method are defined with 'M' suffix, e.g., Likelihood-M. Furthermore, although we focus on metric-based methods, we also compare with model-based methods ChatGPT-D (Guo et al., 2023) and MPU (Tian et al., 2024). Their details can be found in Appendix D.2. 

Metrics. First, as a binary classification problem, we use the area under the receiver operating characteristic curve (AUROC). Second, following (Tufts et al., 2024; Fraser et al., 2025; Hans et al., 2024), we recognize the negative impact of misclassifying human text as machine-generated text. Therefore, another important evaluation metric is the true positive rate (TPR) at a low false positive rate (FPR). Specifically, we measure the TPR at an FPR of 1%, denoting this as TPR@FPR-1%. 


Table 2: Performance concerning AUROC (%) on Essay (left) and DetectRL (right).


<table><tr><td rowspan="2">Method</td><td colspan="7">Essay (Training Text: GPT4All)</td><td colspan="4">DetectRL (Training Text: Llama-2-70b)</td></tr><tr><td>GPT4All</td><td>ChatGPT</td><td>Dolly</td><td>ChatGLM</td><td>Claude</td><td>ChatGPT-turbo</td><td>Avg.</td><td>Llama-2-70b</td><td>ChatGPT</td><td>Google-PaLM</td><td>Avg.</td></tr><tr><td>Likelihood</td><td><eq>96.16 \pm 0.30</eq></td><td><eq>98.79 \pm 0.19</eq></td><td><eq>90.90 \pm 1.33</eq></td><td><eq>99.29 \pm 0.25</eq></td><td><eq>92.76 \pm 0.23</eq></td><td><eq>99.13 \pm 0.19</eq></td><td>96.17</td><td><eq>78.58 \pm 0.41</eq></td><td><eq>66.61 \pm 0.99</eq></td><td><eq>71.42 \pm 0.49</eq></td><td>72.20</td></tr><tr><td>Likelihood-M</td><td><eq>98.58 \pm 0.14</eq></td><td><eq>99.47 \pm 0.11</eq></td><td><eq>94.59 \pm 0.96</eq></td><td><eq>99.54 \pm 0.18</eq></td><td><eq>94.82 \pm 0.36</eq></td><td><eq>99.72 \pm 0.13</eq></td><td>97.79</td><td><eq>87.61 \pm 0.48</eq></td><td><eq>73.70 \pm 0.58</eq></td><td><eq>81.21 \pm 1.21</eq></td><td>80.84</td></tr><tr><td>Log-Rank</td><td><eq>96.55 \pm 0.31</eq></td><td><eq>98.95 \pm 0.13</eq></td><td><eq>90.08 \pm 1.28</eq></td><td><eq>99.36 \pm 0.13</eq></td><td><eq>92.01 \pm 0.20</eq></td><td><eq>99.24 \pm 0.15</eq></td><td>96.03</td><td><eq>79.67 \pm 0.46</eq></td><td><eq>65.85 \pm 0.94</eq></td><td><eq>70.66 \pm 0.40</eq></td><td>72.06</td></tr><tr><td>Log-Rank-M</td><td><eq>98.57 \pm 0.06</eq></td><td><eq>99.41 \pm 0.09</eq></td><td><eq>93.82 \pm 0.91</eq></td><td><eq>99.55 \pm 0.08</eq></td><td><eq>92.91 \pm 0.30</eq></td><td><eq>99.64 \pm 0.10</eq></td><td>97.32</td><td><eq>90.20 \pm 0.44</eq></td><td><eq>75.32 \pm 0.91</eq></td><td><eq>84.34 \pm 0.61</eq></td><td>83.29</td></tr><tr><td>Entropy</td><td><eq>74.19 \pm 1.62</eq></td><td><eq>89.49 \pm 0.34</eq></td><td><eq>73.26 \pm 1.48</eq></td><td><eq>84.11 \pm 0.77</eq></td><td><eq>86.58 \pm 0.66</eq></td><td><eq>95.94 \pm 0.35</eq></td><td>83.93</td><td><eq>66.21 \pm 0.88</eq></td><td><eq>63.09 \pm 1.05</eq></td><td><eq>60.73 \pm 0.91</eq></td><td>63.34</td></tr><tr><td>Entropy-M</td><td><eq>83.52 \pm 0.73</eq></td><td><eq>93.28 \pm 0.15</eq></td><td><eq>81.11 \pm 0.91</eq></td><td><eq>91.44 \pm 0.35</eq></td><td><eq>87.96 \pm 0.48</eq></td><td><eq>96.75 \pm 0.17</eq></td><td>89.01</td><td><eq>69.97 \pm 0.90</eq></td><td><eq>65.26 \pm 1.36</eq></td><td><eq>65.82 \pm 0.96</eq></td><td>67.02</td></tr><tr><td>DetectGPT</td><td><eq>50.81 \pm 0.58</eq></td><td><eq>46.40 \pm 0.77</eq></td><td><eq>57.48 \pm 0.84</eq></td><td><eq>50.41 \pm 1.70</eq></td><td><eq>41.54 \pm 0.60</eq></td><td><eq>17.90 \pm 1.25</eq></td><td>44.09</td><td><eq>52.37 \pm 0.59</eq></td><td><eq>50.22 \pm 0.60</eq></td><td><eq>43.19 \pm 1.41</eq></td><td>48.60</td></tr><tr><td>DetectGPT-M</td><td><eq>95.37 \pm 2.42</eq></td><td><eq>96.20 \pm 2.48</eq></td><td><eq>85.39 \pm 7.02</eq></td><td><eq>95.97 \pm 2.93</eq></td><td><eq>80.29 \pm 10.43</eq></td><td><eq>98.49 \pm 0.78</eq></td><td>91.95</td><td><eq>78.81 \pm 3.68</eq></td><td><eq>64.15 \pm 4.48</eq></td><td><eq>75.90 \pm 4.53</eq></td><td>72.95</td></tr><tr><td>FastGPT</td><td><eq>64.63 \pm 1.53</eq></td><td><eq>67.68 \pm 1.70</eq></td><td><eq>47.17 \pm 1.53</eq></td><td><eq>71.08 \pm 1.51</eq></td><td><eq>75.31 \pm 0.90</eq></td><td><eq>88.62 \pm 0.67</eq></td><td>69.08</td><td><eq>67.72 \pm 1.02</eq></td><td><eq>58.50 \pm 1.22</eq></td><td><eq>56.68 \pm 1.21</eq></td><td>60.97</td></tr><tr><td>FastGPT-M</td><td><eq>87.22 \pm 3.40</eq></td><td><eq>91.56 \pm 3.33</eq></td><td><eq>82.61 \pm 17.98</eq></td><td><eq>95.36 \pm 0.48</eq></td><td><eq>59.29 \pm 1.89</eq></td><td><eq>63.48 \pm 18.13</eq></td><td>79.92</td><td><eq>66.55 \pm 3.36</eq></td><td><eq>55.19 \pm 3.26</eq></td><td><eq>63.30 \pm 4.37</eq></td><td>61.68</td></tr><tr><td>DNA-GPT</td><td><eq>98.08 \pm 0.23</eq></td><td><eq>96.56 \pm 0.28</eq></td><td><eq>92.78 \pm 0.68</eq></td><td><eq>98.00 \pm 0.13</eq></td><td><eq>89.92 \pm 0.29</eq></td><td><eq>96.22 \pm 0.27</eq></td><td>95.26</td><td><eq>71.91 \pm 0.91</eq></td><td><eq>64.14 \pm 0.91</eq></td><td><eq>67.32 \pm 0.94</eq></td><td>67.79</td></tr><tr><td>DNA-GPT-M</td><td><eq>99.68 \pm 0.07</eq></td><td><eq>98.88 \pm 0.06</eq></td><td><eq>97.04 \pm 0.49</eq></td><td><eq>99.26 \pm 0.05</eq></td><td><eq>94.73 \pm 0.28</eq></td><td><eq>98.85 \pm 0.09</eq></td><td>98.07</td><td><eq>74.39 \pm 1.02</eq></td><td><eq>64.36 \pm 1.08</eq></td><td><eq>70.50 \pm 1.03</eq></td><td>69.75</td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/a506598141b0158729692dad311378a3537a96988758f325e8149dc307592419.jpg)


## 5.1 PERFORMANCE COMPARISON


Figure 4: The performance improvement of the proposed method on Likelihood and Log-Rank. Values greater than 0 indicate an enhanced effect.


In this section, we evaluate the enhancement effectiveness of the proposed method in various real-world scenarios, including cross-LLM, cross-domain, detecting mixed machine text, and resisting paraphrase attacks. Details of these scenarios are provided in Appendix D.3. 

Performance across Different LLMs. Table 2 reports AUROC for detectors trained on GPT4All (Essay) and Llama-2-70b (DetectRL) and evaluated on texts from various LLMs. TPR@FPR-1% results are provided in Table 3 in the Appendix. Besides, results of detectors trained on other LLM texts and on the Reuters dataset appear in Tables 4–15 of the Appendix. The proposed method yields significant gains for nearly all baselines. For example, on Essay, it raises Likelihood from 52.4% to 77.86% (+25.46%). Encouragingly, our method also benefits weak detectors: DetectGPT significantly improves from 0.15% to 37.18% (+37.03%), suggesting that the assumptions underlying its scoring are reasonable and that underperformance mainly stemmed from score estimation error. Finally, detection performance on texts from the same LLM is not always superior to cross-LLM. For example, on Essay, performance on ChatGLM is particularly strong and even exceeds the intra-LLM case of GPT4All, possibly because ChatGLM texts are more discriminative. 

Performance across Different Domains. We evaluate cross-domain performance in four high-risk domains: arXiv, Writing Prompts, XSum, and Yelp Reviews. Results are summarized in Fig. 4; additional enhanced results for more detectors under cross-domain settings are provided in Fig. 21 and Fig. 22 of the Appendix. In most settings, detectors equipped with our strategy show substantially stronger cross-domain generalization. We attribute this to calibrating the detection score with only a few carefully designed parameters (a 2x2 reward–penalty coefficient matrix), which helps prevent overfitting and thereby improves out-of-domain detection. 

Performance against Mixed Texts. In practice, human–AI collaboration is pervasive, leading to widespread mixed human–machine text. We therefore evaluate the proposed strategy for mixed-text detection. Two training strategies are considered: training on the original text and training on the mixed text. Results concerning AUROC are shown in Fig. 5, with TPR@FPR-1% shown in Fig. 23 of the Appendix. In most settings, the proposed strategy enhances the detector's ability to recognize mixed-text. Moreover, comparing training on original versus mixed text shows that simply training on mixed text does not improve the detection ability of mixed text, highlighting the focus of mixed text detection research. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/546462017ebf976bddee4e7485bd5058240106b9b96c7cfa714435131397734c.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/e11632f54739863813d59ded97ed11e9594599ada1f3b1a4263cc0337dff1a0b.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/8d66fcd20af82909a1555da05641b519d155b779d6e992cf16b384594b59d39b.jpg)



Figure 5: Detection performance concerning AUROC under different LLM mixed texts. All detectors are trained on Llama-2-70b texts.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/06693abc39ea7e0ec6a633936b464f757e4ecf5785bc3f516d2fa0f3d6d73408.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/0d93ae3af6785f0984213888f27ee4bf3fccf379e9f900e839954c861fb1aa5d.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/0644e0025e2f135378554bfa878c039c720ca9c9f8fdc7360ef90d34bdb41f87.jpg)



Figure 6: Detection performance concerning AUROC under different paraphrasing texts (Dipper and Polish). All detectors are trained on Llama-2-70b texts.


Performance against Paraphrasing Attacks. Prior work (Sadasivan et al., 2023) shows that MGT detection is typically vulnerable to paraphrasing attacks, where an adversary rewrites a passage without altering its semantics to evade detection. We therefore assess the robustness gains of our framework on the Dipper and Polish paraphrase attacks provided by DetectRL. The results in Fig. 6 and Fig. 24 clearly indicate that, even in adversarial settings, our strategy yields encouraging improvements in robustness against paraphrasing attacks. 

## 5.2 ABLATION STUDY

We introduce an MRF layer and a positional weighting function to model neighbor similarity and initial instability, respectively. In this section, we verify their effectiveness through ablation studies, denoted as "w/o MRF" and "w/o Pos". Results on the Essay dataset are shown in Fig. 7, with additional results in Fig. 25 and Fig. 26 of the Appendix. We can find that removing either component significantly drops performance, while retaining only one still outperforms the baseline detector, underscoring their designs' rationality. Besides, the contribution of each component varies by detector type. For single-text detectors like Likelihood and Log-Rank, the positional weighting function provides the most improvement by addressing instability in initial scores. However, for methods that aggregate multiple text scores, such as DetectGPT and FastGPT, this instability is already partially mitigated, making the MRF-based score calibration the primary source of gains. 

## 5.3 MRF vs. NEURAL NETWORK

This paper proposes a Markov-informed calibration method to model the relationship of detection scores of context tokens. To highlight the rationale of this design, we compare it with methods that directly use neural networks to calibrate scores. A simple three-layer neural network is used here, defined with the "nn" suffix. The comparison results on Likelihood are shown in Fig. 8, and more results can be found in Fig. 31 and Fig. 32 of the Appendix. Although NN-based methods exhibit competitive performance in some settings for intra-LLMs, their generalization ability on cross-LLMs drops significantly. This suggests that it does not truly learn the general ability to correct scores, but rather overfits the training data. In contrast, our strategy shows good generalization. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/02e2b5db59a92233df42ecd887ac210c4ec7d4334ca2d7678b7181a4e674f7b0.jpg)



Figure 7: Ablation results on the Essay dataset. The y-axis represents the LLM text on which the detector was trained, and the x-axis represents the average performance across LLMs.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/6535ebfc2cfd0a196a259a84daaf9b6988b20fd35f1bd45ed4570bd74f6bdebc.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/104620296159a82dca31b435be6ac361cd10127b050b29b5a63b8f1c6f3b72e4.jpg)



Essay Dataset


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/ede37c8666999208b541d90b883a1ed5eb94cca3e4f14bade04b3ff3b9ac4feb.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/bad3c10d8fc121ade9efbde0dec8748dc1e10a2d2a54ea68e03952f97f556c7c.jpg)



DetectRL Dataset



Figure 8: Comparison with NN-based calibration methods. The detector used is Likelihood.


## 6 CONCLUSION

This paper has systematically examined representative metric-based detectors within a unified framework, revealing a core challenge: the inherent randomness of the LLM generation process leads to inaccurate detection scores, and naive aggregation of existing methods fails to fix this. Therefore, we have theoretically and empirically established two key properties of these scores: neighbor similarity and initial instability. Building on these insights, we have proposed a Markov-informed score calibration method that captures token relationships and corrects the biased scores produced by base detectors. Extensive experiments show substantial and consistent performance gains of the proposed method. Admittedly, the proposed method relies on modeling the relationships between context tokens to calibrate detection scores. Consequently, our method is not directly applicable to detection methods that do not provide this fine-grained, token-level output. 

## ACKNOWLEDGMENT

This work was supported by the RGC Senior Research Fellow Scheme under the grant: SRFS2324-2S02, RGC Young Collaborative Research Grant No. C2005-24Y, and HKBU CSD Departmental Incentive Scheme. 

## ETHICS STATEMENT

This paper proposes a Markov-informed calibration strategy to enhance machine-generated text detection and mitigate the potential risks posed by machine-generated text, including disinformation and phishing. Our work does not involve ethical issues such as dataset releases, potentially harmful insights, potential conflicts of interest and sponsorship, discrimination/bias/fairness concerns, privacy and security issues, legal compliance, and research integrity issues. 

## REPRODUCIBILITY STATEMENT

Our code is available at https://github.com/tmlr-group/MRF_Calibration, and all datasets used in our experiments (Essay, Reuters, and DetectRL) are publicly available for download. In addition, we provide detailed implementation details in the Appendix, including data partitioning, the fixed seeds, the learning rate, the training batch size, and the two parameters $t_{0}$ and T introduced by the proposed strategy, to ensure reproducibility of our work. 

## REFERENCES



Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 





Guangsheng Bao, Yanbin Zhao, Zhiyang Teng, Linyi Yang, and Yue Zhang. Fast-detectgpt: Efficient zero-shot detection of machine-generated text via conditional probability curvature. In ICLR, 2024. 





Xin Chen, Junchao Wu, Shu Yang, Runzhe Zhan, Zeyu Wu, Ziyang Luo, Di Wang, Min Yang, Lidia S Chao, and Derek F Wong. Repreguard: Detecting llm-generated text by revealing hidden representation patterns. arXiv preprint arXiv:2508.13152, 2025. 





Leyan Deng, Chenwang Wu, Defu Lian, Yongji Wu, and Enhong Chen. Markov-driven graph convolutional networks for social spammer detection. IEEE Transactions on Knowledge and Data Engineering, 35(12):12310–12322, 2022. 





Kathleen C Fraser, Hillary Dawkins, and Svetlana Kiritchenko. Detecting ai-generated text: Factors influencing detectability with current methods. Journal of Artificial Intelligence Research, 82:2233–2278, 2025. 





Sebastian Gehrmann, Hendrik Strobelt, and Alexander M Rush. Gltr: Statistical detection and visualization of generated text. arXiv preprint arXiv:1906.04043, 2019. 





GPTZero. Gptzero official website. [Online], 2023. https://gptzero.me. 





Chenchen Gu, Xiang Lisa Li, Percy Liang, and Tatsunori Hashimoto. On the learnability of watermarks for language models. arXiv preprint arXiv:2312.04469, 2023. 





Biyang Guo, Xin Zhang, Ziyuan Wang, Minqi Jiang, Jinran Nie, Yuxuan Ding, Jianwei Yue, and Yu-peng Wu. How close is chatgpt to human experts? comparison corpus, evaluation, and detection. arXiv preprint arXiv:2301.07597, 2023. 





Abhimanyu Hans, Avi Schwarzschild, Valeriia Cherepanova, Hamid Kazemi, Aniruddha Saha, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Spotting llms with binoculars: Zero-shot detection of machine-generated text. In International Conference on Machine Learning, pp. 17519–17537. PMLR, 2024. 





Jason Hong. The state of phishing attacks. Communications of the ACM, 55(1):74–81, 2012. 





Abe Hou, Jingyu Zhang, Tianxing He, Yichen Wang, Yung-Sung Chuang, Hongwei Wang, Lingfeng Shen, Benjamin Van Durme, Daniel Khashabi, and Yulia Tsvetkov. Semstamp: A semantic watermark with paraphrastic robustness for text generation. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 4067–4082, 2024. 





Xiaomeng Hu, Pin-Yu Chen, and Tsung-Yi Ho. Radar: Robust ai-text detection via adversarial learning. Advances in neural information processing systems, 36:15077–15095, 2023. 





John Kirchenbauer, Jonas Geiping, Yuxin Wen, Jonathan Katz, Ian Miers, and Tom Goldstein. A watermark for large language models. In International Conference on Machine Learning, pp. 17061–17084. PMLR, 2023. 





Scott Kirkpatrick, C Daniel Gelatt Jr, and Mario P Vecchi. Optimization by simulated annealing. science, 220(4598):671–680, 1983. 





Kalpesh Krishna, Yixiao Song, Marzena Karpinska, John Wieting, and Mohit Iyyer. Paraphrasing evades detectors of ai-generated text, but retrieval is an effective defense. Advances in Neural Information Processing Systems, 36:27469–27500, 2023. 





Linyang Li, Pengyu Wang, Ke Ren, Tianxiang Sun, and Xipeng Qiu. Origin tracing and detecting of llms. arXiv preprint arXiv:2304.14072, 2023. 





Stephanie Lin, Jacob Hilton, and Owain Evans. Truthfulqa: Measuring how models mimic human falsehoods. In Proceedings of the 60th annual meeting of the association for computational linguistics (volume 1: long papers), pp. 3214–3252, 2022. 





Xiaoming Liu, Zhaohan Zhang, Yichen Wang, Hang Pu, Yu Lan, and Chao Shen. Coco: Coherence-enhanced machine-generated text detection under data limitation with contrastive learning. arXiv preprint arXiv:2212.10341, 2022. 





Yepeng Liu and Yuheng Bu. Adaptive text watermark for large language models. In Forty-first International Conference on Machine Learning. 





Zichang Liu, Aditya Desai, Fangshuo Liao, Weitao Wang, Victor Xie, Zhaozhuo Xu, Anastasios Kyrillidis, and Anshumali Shrivastava. Scissorhands: Exploiting the persistence of importance hypothesis for llm kv cache compression at test time. Advances in Neural Information Processing Systems, 36:52342–52364, 2023. 





Shixuan Ma and Quan Wang. Zero-shot detection of llm-generated text using token cohesiveness. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 17538–17553, 2024. 





Eric Mitchell, Yoonho Lee, Alexander Khazatsky, Christopher D Manning, and Chelsea Finn. Detectgpt: Zero-shot machine-generated text detection using probability curvature. In International Conference on Machine Learning, pp. 24950–24962. PMLR, 2023. 





Hoang-Quoc Nguyen-Son, Minh-Son Dao, and Koji Zettsu. Simllm: Detecting sentences generated by large language models using similarity between the generation and its re-generation. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 22340–22352, 2024. 





Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019. 





Vinu Sankar Sadasivan, Aounon Kumar, Sriram Balasubramanian, Wenxiao Wang, and Soheil Feizi. Can ai-generated text be reliably detected? arXiv preprint arXiv:2303.11156, 2023. 





Irene Solaiman, Miles Brundage, Jack Clark, Amanda Askell, Ariel Herbert-Voss, Jeff Wu, Alec Radford, Gretchen Krueger, Jong Wook Kim, Sarah Kreps, et al. Release strategies and the social impacts of language models. arXiv preprint arXiv:1908.09203, 2019. 





Jinyan Su, Terry Yue Zhuo, Di Wang, and Preslav Nakov. Detectllm: Leveraging log rank information for zero-shot detection of machine-generated text. In The 2023 Conference on Empirical Methods in Natural Language Processing. 





Yuchuan Tian, Hanting Chen, Xutao Wang, Zheyuan Bai, QINGHUA ZHANG, Ruifeng Li, Chao Xu, and Yunhe Wang. Multiscale positive-unlabeled detection of ai-generated texts. In The Twelfth International Conference on Learning Representations, 2024. 





Brian Tufts, Xuandong Zhao, and Lei Li. An examination of ai-generated text detectors across multiple domains and models. In Neurips Safe Generative AI Workshop 2024, 2024. 





Eduard Tulchinskii, Kristian Kuznetsov, Laida Kushnareva, Daniil Cherniavskii, Sergey Nikolenko, Evgeny Burnaev, Serguei Barannikov, and Irina Piontkovskaya. Intrinsic dimension estimation for robust detection of ai-generated texts. Advances in Neural Information Processing Systems, 36, 2024. 





Vivek Verma, Eve Fleisig, Nicholas Tomlin, and Dan Klein. Ghostbuster: Detecting text ghostwritten by large language models. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 1702–1717, 2024. 





Ivan Vykopal, Matúš Pikuliak, Ivan Srba, Robert Moro, Dominik Macko, and Mária Bieliková. Disinformation capabilities of large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 14830–14847, 2024. 





Pengyu Wang, Linyang Li, Ke Ren, Botian Jiang, Dong Zhang, and Xipeng Qiu. Seqxgpt: Sentence-level ai-generated text detection. In The 2023 Conference on Empirical Methods in Natural Language Processing, 2023. 





Junchao Wu, Runzhe Zhan, Derek Wong, Shu Yang, Xinyi Yang, Yulin Yuan, and Lidia Chao. Detectrl: Benchmarking llm-generated text detection in real-world scenarios. Advances in Neural Information Processing Systems, 37:100369–100401, 2024. 





Junxi Wu, Jinpeng Wang, Zheng Liu, Bin Chen, Dongjian Hu, Hao Wu, and Shu-Tao Xia. Moses: Uncertainty-aware ai-generated text detection via mixture of stylistics experts with conditional thresholds. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 5797–5816, 2025. 





Kangxi Wu, Liang Pang, Huawei Shen, Xueqi Cheng, and Tat-Seng Chua. Llmdet: A large language models detection tool. CoRR, 2023a. 





Yihan Wu, Zhengmian Hu, Hongyang Zhang, and Heng Huang. Dipmark: A stealthy, efficient and resilient watermark for large language models. 2023b. 





Yang Xu, Yu Wang, Hao An, Zhichen Liu, and Yongyuan Li. Detecting subtle differences between human and model languages using spectrum of relative likelihood. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 10108–10121, 2024. 





Yihuai Xu, Yongwei Wang, Yifei Bi, Huangsen Cao, Zhouhan Lin, Yu Zhao, and Fei Wu. Training-free llm-generated text detection by mining token probability sequences. In The Thirteenth International Conference on Learning Representations, 2025. 





Xianjun Yang, Kexun Zhang, Haifeng Chen, Linda Petzold, William Yang Wang, and Wei Cheng. Zero-shot detection of machine-generated codes. arXiv preprint arXiv:2310.05103, 2023. 





Xianjun Yang, Wei Cheng, Yue Wu, Linda Ruth Petzold, William Yang Wang, and Haifeng Chen. Dna-gpt: Divergent n-gram analysis for training-free detection of gpt-generated text. In The Twelfth International Conference on Learning Representations, 2024. 





KiYoon Yoo, Wonhyuk Ahn, and Nojun Kwak. Advancing beyond identification: Multi-bit watermark for large language models. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 4031–4055, 2024. 





Xiao Yu, Yuang Qi, Kejiang Chen, Guoqiang Chen, Xi Yang, Pengyuan Zhu, Weiming Zhang, and Nenghai Yu. Gpt paternity test: Gpt generated text detection with gpt genetic inheritance. CoRR, 2023a. 





Xiao Yu, Kejiang Chen, Qi Yang, Weiming Zhang, and Nenghai Yu. Text fluoroscopy: Detecting llm-generated text through intrinsic features. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 15838–15846, 2024. 





Zhiyuan Yu, Yuhao Wu, Ning Zhang, Chenguang Wang, Yevgeniy Vorobeychik, and Chaowei Xiao. Codeipprompt: intellectual property infringement assessment of code language models. In International conference on machine learning, pp. 40373–40389. PMLR, 2023b. 





Haolan Zhan, Xuanli He, Qiongkai Xu, Yuxiang Wu, and Pontus Stenetorp. G3detector: General gpt-generated text detector. arXiv preprint arXiv:2305.12680, 2023. 





Ruisi Zhang, Shehzeen Samarah Hussain, Paarth Neekhara, and Farinaz Koushanfar. {REMARK-LLM}: A robust and efficient watermarking framework for generative large language models. In 33rd USENIX Security Symposium (USENIX Security 24), pp. 1813–1830, 2024. 



## APPENDIX

A More Discussion of the Proposed Method 15
A.1 Contribution . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
A.2 Differences from Existing Methods . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
B Related Work 15
B.1 Watermark-based Detection . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . B.2 Model-based Detection 16
B.3 Metric-based Detection 16
C Proof of Theorem 1 17
D Experimental Details 19
D.1 Datasets 19
D.2 Baselines 19
D.3 Experimental Scenario 20
D.4 Threat Model and Proxy Settings 20
D.5 Parameter settings 21
E More Experimental Results 21
E.1 More Results of Token Score Distribution before and after Enhancement 21
E.2 More Results of Context Token Relationships 22
E.3 More Performance Comparison 22
E.4 More Performance Comparison on the QA Dataset 23
E.5 More Performance Comparison on Short-text 23
E.6 Enhancement Experiments on More Detectors 23
E.7 Performance Comparison against Perturbation Texts 24
E.8 More Results of Ablation Study 24
E.9 Sensitivity Analysis 24
E.10 More Results Comparing with Neural Network Calibration 25
E.11 MRF vs. HMM 25
E.12 Comparison with Enhanced Method 26
E.13 Analysis of Cross-domain Failure Cases 26
E.14 Comparison with Model-based Detectors 27
E.15 Running Time 27
F The Use of Large Language Models 27 

## A MORE DISCUSSION OF THE PROPOSED METHOD

## A.1 CONTRIBUTION

The contributions of this paper are multifaceted. 

- We provide a unified perspective for understanding metric-based detection methods. The diversity of these methods (e.g., using metrics based on log-likelihood or log-rank, and introducing perturbed or regenerated text) makes comparison difficult. To this end, we re-examine these methods from three perspectives: data, score calculation, and detection. This analysis provides a precise definition for each method, facilitating comparison and potential improvements. Our analysis shows that these methods employ more reasonable evaluation metrics and incorporate additional contextual information to enhance detection. However, they all fail to address the underlying token-level errors caused by inherent randomness, which limits their detection potential. Therefore, our unified analysis encourages a more nuanced characterization of token-level detection scores, which will provide guidance for future improvements. 

- We reveal the relationships between contextual token detection scores. While the generative mechanisms of LLMs introduce dependencies between tokens, these relationships remain unclear. To this end, we theoretically reveal two relationships between contextual token scores: neighboring similarity and initial instability, which are further validated through empirical experiments. Constraining these relationships during detection has the potential to mitigate the imprecision in score calculation caused by the inherent randomness of MGTs, which is crucial for the field of MGT detection. 

- We propose a Markov-informed score calibration method to enhance MGT detection. This involves using Markov random fields to capture the revealed relationships and, through mean-field approximation, modeling the MRF model as a lightweight component that can be stacked on existing detectors to further unlock detection potential. It is worth noting that our main technical contribution and innovation lies not only in the specific implementation but also in the conceptual token-level score calibration. While the current implementation is based on Markov random fields, this is only one possible approach; alternatives include using sequence models or graph neural networks. This conceptual insight can inspire improving MGT detection. 

- Extensive experiments consistently demonstrate the enhanced effectiveness of the proposed strategy. We empirically verify that it not only excels on a single task but also demonstrates strong capabilities in multiple complex and challenging real-world scenarios, including generalization across LLMs and domains, and robustness to mixed text and paraphrased text. Furthermore, the proposed enhancement component incurs negligible computational overhead compared to the original detector. This combination of effectiveness and efficiency provides a solid foundation for developing practically deployable enhancement solutions for AI-generated text detection. 

## A.2 DIFFERENCES FROM EXISTING METHODS

Although some works, such as FourierGPT Xu et al. (2024) and Lastde Xu et al. (2025), also learn local patterns, our approach is quite different: 

- Methodology. Unlike FourierGPT or Lastde, which propose new standalone metrics, our method is a universal plug-in. We do not replace the metric; we calibrate the intermediate token-level scores of any existing metric (Likelihood, DetectGPT, etc.) using a Markov Random Field. 

- Theory vs. Empiricism. While many existing local-pattern methods are empirically motivated (e.g., observing spectral power differences and token probability fluctuation), our approach is grounded in the theoretical derivation of attention dynamics (Theorem 1), which formally reveals the Neighbor Similarity and Initial Instability properties we model. 

## B RELATED WORK

Existing detection methods can be categorized into active watermark-based methods and passive model-based and metric-based methods. 

## B.1 WATERMARK-BASED DETECTION

Watermarking is a proactive defense technique that embeds verifiable information during the text generation stage, thereby enabling simple and reliable detection. RedList (Kirchenbauer et al., 2023) is a model-agnostic watermarking method that dynamically partitions the vocabulary into a “greenlist” and “redlist” based on preceding context, slightly increasing the probability of sampling tokens from the greenlist. Subsequent works have made various improvements to this approach. For instance, SemStamp (Hou et al., 2024) introduces a sentence-level semantic hashing watermark to enhance robustness against paraphrasing attacks; DiPmark (Wu et al., 2023b) designs an unbiased watermark that does not alter the original output distribution. REMARK-LLM (Zhang et al., 2024) is a training-based watermarking method that employs a message encoding module to generate an encrypted token distribution for watermark embedding prior to inference. Beyond manually designed watermarks, directly leveraging language models to learn to generate watermarked text is also promising, including training student models (Gu et al., 2023) and semantically invariant watermarking models (Liu & Bu). In addition to the standard binary (0/1 bit) detection of AI-generated content, researchers have also explored multi-bit watermarks (Yoo et al., 2024) for embedding more information. 

## B.2 MODEL-BASED DETECTION

Model-based methods represent a classical paradigm in detection, training a binary classifier on a dataset containing both human- and machine-generated texts. A series of works, such as OpenAI Detector (Solaiman et al., 2019), ChatGPT Detector (Guo et al., 2023), GPTZero (GPTZero, 2023), and G3 Detector (Zhan et al., 2023), collect texts generated by various LLMs to train a unified classifier. GPT-Pat (Yu et al., 2023a) finds that detectors trained solely on a single decoding strategy generalize poorly and enhances performance by utilizing mixed decoding strategies. In addition to original data, GLTR (Gehrmann et al., 2019) trains a simple logistic regression classifier by analyzing the predicted ranking of each word within its context. SeqXGPT (Wang et al., 2023) treats the sequence of logits as waveform signals for detection, while Sniffer (Li et al., 2023) uses the difference in logits from different models on the same text as features for detection and attribution. Beyond the data level, recent works have explored more advanced training strategies. For example, CoCo (Liu et al., 2022) introduces graph structures and contrastive learning; LLMDet (Wu et al., 2023a) leverages the perplexity of surrogate models as additional features; MPU (Tian et al., 2024) adopts a positive-unlabeled learning paradigm; and RADAR (Hu et al., 2023) incorporates adversarial training to enhance model robustness. The above methods generally assume a known text source, but when it is unknown, Ghostbuster (Verma et al., 2024) proposes training classifiers directly using texts generated from known surrogate models. 

## B.3 METRIC-BASED DETECTION

Metric-based methods do not require training on specific datasets; instead, they directly leverage the inherent statistical biases or intrinsic properties of language model-generated text for distinction. The main advantage of such methods lies in their stronger generalization to new models and domains. Classic approaches in this category include the use of Log-Likelihood (Solaiman et al., 2019), Log-Rank (Mitchell et al., 2023), and Entropy (Gehrmann et al., 2019). DetectGPT (Mitchell et al., 2023) finds that AI-generated text typically lies in regions of negative curvature with respect to the model's log-probability function. By perturbing the text and observing changes in log-probability, it can effectively distinguish AI-generated text. Inspired by DetectGPT, Fast-DetectGPT (Bao et al., 2024) replaces log-probability with conditional probability curvature, significantly improving detection efficiency while maintaining performance. DetectLLM-LRR (Su et al.) proposes using the ratio of log-likelihood to log-rank for detection. Some works, such as DNA-GPT (Yang et al., 2024) and DetectGPT4Code (Yang et al., 2023), detect AI-generated text by comparing discrepancies between the original text and continuations generated by a surrogate model. PHD (Tulchinskii et al., 2024) observes that genuine human-written text possesses higher intrinsic dimensionality after encoder mapping. SimLLM (Nguyen-Son et al., 2024) is based on the observation that the similarity between the original text and its generated continuation is significantly higher than that between generated text and its re-generated version; thus, it estimates the similarity between an input sentence and its generated counterpart for detection. Given that existing methods struggle with out-of-distribution data, token coherence (Ma & Wang, 2024) has been shown to be a reliable metric since 

LLM-generated text usually exhibits higher token coherence than human-written text. Yu et al. (Yu et al., 2024) capture the intrinsic features of text by identifying layers with the greatest distributional differences when projecting into the vocabulary space, and using intrinsic rather than semantic features for detection has been demonstrated to yield better results. RepreGuard (Chen et al., 2025) extracted unique activation features from the MGT using a surrogate model, and then the projection scores along the direction of these features are more discriminative. Lastde (Xu et al., 2025) introduced time series analysis to LLM-generated text detection, capturing the temporal dynamics of token probability sequences. FourierGPT (Xu et al., 2024) offered a new perspective on human and model text detection tasks by using relative rather than absolute probability values and by extracting useful features from the probability spectrum. MoSEs (Wu et al., 2025) achieved the quantification of uncertainty in style perception through conditional threshold estimation. 

## C PROOF OF THEOREM 1

Theorem. 1 Let $\lambda_K, \lambda_Q, \lambda_V, \lambda_O$ be the largest singular values of parameters $W_K, W_Q, W_V, W_O$ , respectively, and let $W = W_V W_O W_Q W_K^\top$ . For the transformer defined in Eq. (1), assuming normalized inputs ( $\|x_t\|_2 = 1$ for all $t$ ) and constants $c, \epsilon > 0$ , consider $a_t x_{t+1}^\top \geq (1 - \delta) \|a_t\|_2$ with $\delta \leq \left(\frac{c\epsilon}{\lambda_Q \lambda_K \lambda_V \lambda_O}\right)^2$ . If $x_\ell$ satisfies $x_\ell W x_\ell^\top \geq c$ and $x_\ell W x_\ell \geq \epsilon^{-1} \max_{j \in [\ell], j \neq \ell} x_j W x_\ell^\top$ , then 

$$
\alpha_ {t + 1, l} \leq \frac {\exp (C _ {l} \cdot \alpha_ {t , l} + \eta)}{\exp (C _ {l} \cdot \alpha_ {t , l} + \eta) + \sum_ {j \neq l} \exp (C _ {j} \cdot \alpha_ {t , j} - \eta)},
$$

$$
\alpha_ {t + 1, l} \geq \frac {\exp (C _ {l} \cdot \alpha_ {t , l} - \eta)}{\exp (C _ {l} \cdot \alpha_ {t , l} - \eta) + \sum_ {j \neq l} \exp (C _ {j} \cdot \alpha_ {t , j} + \eta)},
$$

where 

$$
C _ {j} = \frac {x _ {j} W x _ {j} ^ {T}}{t | a _ {t} | _ {2}}, a n d \eta = \frac {(1 + \sqrt {2}) \epsilon x _ {j} W x _ {j} ^ {T}}{(t + 1) | a _ {t} | _ {2}}.
$$

Proof. Our proof follows the proof of existing work (Liu et al., 2023). First, we introduce two necessary lemmas to help our proof. 

Lemma 2 ((Liu et al., 2023)). Let $x_{1}, x_{2} \in \mathbb{R}^{1 \times m}$ satisfies $\| x_{1} \|_{2} = \| x_{2} \|_{2} = 1$ and $x_{1}x_{2}^{\top} \geq 1 - \delta$ for some $\delta \in (0,1)$ . Then for all $y \in \mathbb{R}^{1 \times m}$ we have 

$$
\left| x _ {1} y ^ {\top} - x _ {2} y ^ {\top} \right| \leq \sqrt {2 \delta} \| y \| _ {2}
$$

Lemma 3 ((Liu et al., 2023)). Let $\ell \in [t]$ be given. Suppose that $x_{\ell}Ax_{\ell}^{\top} > \epsilon^{-1}\left|x_{j}Ax_{\ell}^{\top}\right|$ for all $j \neq \ell$ . Then we have 

$$
(\mathcal {S} (t) _ {\ell} - \epsilon) x _ {\ell} ^ {\top} a x _ {\ell} \leq x _ {\ell} ^ {\top} W _ {K} ^ {\top} W _ {Q} a _ {t} \leq (\mathcal {S} (t) _ {\ell} + \epsilon) x _ {\ell} ^ {\top} a x _ {\ell}
$$

Based on these two lemmas, we can formally prove the theorem. Let $x_{1} = \frac{a_{t}}{\|a_{t}\|}$ , and $x_{2} = x_{t + 1}$ . If $\frac{a_t x_{t + 1}^\top}{\|a_t\|} \geq 1 - \delta$ , using the conclusion of Lemma 2, we have 

$$
\left| \frac {a _ {t}}{(t + 1) \| a _ {t} \| _ {2}} W _ {Q} W _ {K} ^ {T} x _ {\ell} ^ {T} - \frac {1}{t + 1} x _ {t + 1} W _ {Q} W _ {K} ^ {T} x _ {\ell} ^ {T} \right| \leq \frac {\sqrt {2 \delta}}{t + 1} \left\| W _ {Q} W _ {K} ^ {T} x _ {\ell} ^ {T} \right\| _ {2}
$$

Since $\lambda_{Q}$ , $\lambda_{K}$ are the maximum singular values, respectively. Then we have $\left\| W_{Q}W_{K}^{\top}x_{\ell}^{\top}\right\|_{2} \leq \lambda_{Q}\lambda_{K}\left\| x_{\ell}\right\|_{2} = \lambda_{Q}\lambda_{K}$ . This leads to: 

$$
\left| \frac {a _ {t}}{(t + 1) \| a _ {t} \| _ {2}} W _ {Q} W _ {K} ^ {T} x _ {\ell} ^ {T} - \frac {1}{t + 1} x _ {t + 1} W _ {Q} W _ {K} ^ {T} x _ {\ell} ^ {T} \right| \leq \frac {\sqrt {2 \delta}}{t + 1} \lambda_ {Q} \lambda_ {K}\tag{10}
$$

Since 

$$
\left\| a _ {t} \right\| _ {2} = \left\| \left(\sum_ {j = 1} ^ {t - 1} \alpha_ {t, j} x _ {j}\right) W _ {V} W _ {O} \right\| \leq \lambda_ {O} \lambda_ {V} \left\| \sum_ {j = 1} ^ {t - 1} \alpha_ {t, j} x _ {j} \right\| _ {2} \leq \lambda_ {O} \lambda_ {V} \sum_ {j = 1} ^ {t - 1} \alpha_ {t, j} \| x _ {j} \| _ {2} = \lambda_ {O} \lambda_ {V},
$$

and the theorem assumes $\delta \leq \left(\frac{c\epsilon}{\lambda_Q\lambda_K\lambda_V\lambda_O}\right)^2$ , substituting these into Eq. (10), we can obtain: 

$$
\frac {\sqrt {2 \delta}}{t + 1} \lambda_ {Q} \lambda_ {K} \leq \frac {\sqrt {2} c \epsilon}{(t + 1) \lambda_ {V} \lambda_ {O}} \leq \frac {\sqrt {2} c \epsilon}{(t + 1) \| a _ {t} \| _ {2}} \leq \frac {\sqrt {2} \epsilon}{(t + 1) \| a _ {t} \| _ {2}} x _ {\ell} a x _ {\ell} ^ {T}\tag{11}
$$

The last inequality is obtained from $x_{\ell}ax_{\ell}^{T} \geq c$ . Then combining Formula (10) and Formula (11), we have 

$$
\left| \frac {a _ {t}}{(t + 1) \| a _ {t} \| _ {2}} W _ {Q} W _ {K} ^ {T} x _ {\ell} ^ {T} - \frac {1}{t + 1} x _ {t + 1} W _ {Q} W _ {K} ^ {T} x _ {\ell} ^ {T} \right| \leq \frac {\sqrt {2} \epsilon}{(t + 1) \| a _ {t} \| _ {2}} x _ {\ell} a x _ {\ell} ^ {T}
$$

From Lemma 3, we have: 

$$
\left| \frac {a _ {t} W _ {Q} W _ {K} ^ {\top} x _ {\ell} ^ {\top}}{(t + 1) \| a _ {t} \| ^ {2}} - \frac {\alpha_ {t , \ell} x _ {\ell} W x _ {\ell} ^ {\top}}{(t + 1) \| a _ {t} \| ^ {2}} \right| \leq \frac {\epsilon}{(t + 1) \| a _ {t} \| ^ {2}} x _ {\ell} ^ {\top} a x _ {\ell}
$$

By the triangle inequality, we can combine the upper bounds: 

$$
\begin{array}{r l} \left| \frac {x _ {t + 1} W _ {Q} W _ {K} ^ {T} x _ {\ell} ^ {T}}{t + 1} - \frac {\alpha_ {t , \ell} x _ {\ell} W x _ {\ell} ^ {T}}{(t + 1) \| a _ {t} \| _ {2}} \right| & \leq \left| \frac {x _ {t + 1} W _ {Q} W _ {K} ^ {T} x _ {\ell} ^ {T}}{t + 1} - \frac {a _ {t} W _ {Q} W _ {K} ^ {T} x _ {\ell} ^ {T}}{(t + 1) \| a _ {t} \| _ {2}} \right| \\ & \quad + \left| \frac {a _ {t} W _ {Q} W _ {K} ^ {T} x _ {\ell} ^ {T}}{(t + 1) \| a _ {t} \| _ {2}} - \frac {\alpha_ {t , \ell} x _ {\ell} W x _ {\ell} ^ {T}}{(t + 1) \| a _ {t} \| _ {2}} \right| \\ & \leq \frac {(1 + \sqrt {2}) \epsilon}{(t + 1) \| a _ {t} \| ^ {2}} x _ {\ell} ^ {\top} a x _ {\ell} \end{array}\tag{12}
$$

Then, we rearrange the inequality, and we can obtain: 

$$
\frac {x _ {\ell} W x _ {\ell} ^ {\top}}{(t + 1) \| a _ {t} \| _ {2}} \left(\alpha_ {t, \ell} - (1 + \sqrt {2}) \epsilon\right) \leq \frac {1}{t + 1} x _ {t + 1} W _ {Q} W _ {K} ^ {\top} x _ {\ell} ^ {\top} \leq \frac {x _ {\ell} W x _ {\ell} ^ {\top}}{(t + 1) \| a _ {t} \| _ {2}} \left(\alpha_ {t, \ell} + (1 + \sqrt {2} \epsilon\right)
$$

Now we give the lower and upper bounds of $\alpha_{t+1,\ell} = \text{softmax}(1/(t+1) \cdot x_{t+1}W_Q W_K^\top X_t^\top)_l$ . Upper bound. Let $\gamma_{t+1,\ell} = 1/(t+1) \cdot x_{t+1}W_Q W_K^\top x_\ell^\top$ . For the softmax function $\alpha_{t+1,l} = \frac{\exp(\gamma_{t+1,\ell})}{\exp(\gamma_{t+1,\ell}) + \sum_{k \neq \ell} \exp(\gamma_{t+1,k})}$ , to get the maximum value of $\alpha_{t+1,\ell}$ , we need to (1) make the numerator as big as possible, which means $\gamma_{t+1,\ell}$ should take its maximum value $\frac{x_\ell W x_\ell^\top}{(t+1)\|a_t\|_2} (\alpha_{t,\ell} + (1 + \sqrt{2})\epsilon)$ , (2) make the denominator as small as possible, which means that all other values $\gamma_{t+1,k}$ (when $k \neq \ell$ ) should be minimized $\frac{x_k W x_k^\top}{(t+1)\|a_t\|_2} (\alpha_{t,k} - (1 + \sqrt{2})\epsilon)$ . Therefore, 

$$
\alpha_ {t + 1, l} \leq \frac {\frac {x _ {\ell} W x _ {\ell} ^ {\top}}{(t + 1) \| a _ {t} \| _ {2}} \left(\alpha_ {t , \ell} + (1 + \sqrt {2}) \epsilon\right)}{\frac {x _ {\ell} W x _ {\ell} ^ {\top}}{(t + 1) \| a _ {t} \| _ {2}} \left(\alpha_ {t , \ell} + (1 + \sqrt {2}) \epsilon\right) + \sum_ {k \neq \ell} \frac {x _ {k} W x _ {k} ^ {\top}}{(t + 1) \| a _ {t} \| _ {2}} \left(\alpha_ {t , k} - (1 + \sqrt {2}) \epsilon\right)}
$$

Lower bound. Similarly, for the lower bound, we should (1) make the numerator as small as possible, which means $\gamma_{t+1,\ell}$ should take its minimum value $\frac{x_{\ell}Wx_{\ell}^{\top}}{(t+1)\|a_{t}\|_{2}}\left(\alpha_{t,\ell}-(1+\sqrt{2})\epsilon\right)$ , (2) make the denominator as large as possible, which means that all other values $\gamma_{t+1,k}$ (when $k\neq\ell$ ) should be maximized $\frac{x_{k}Wx_{k}^{\top}}{(t+1)\|a_{t}\|_{2}}\left(\alpha_{t,k}+(1+\sqrt{2})\epsilon\right)$ . Therefore, 

$$
\alpha_ {t + 1, l} \geq \frac {\frac {x _ {\ell} W x _ {\ell} ^ {\top}}{(t + 1) \| a _ {t} \| _ {2}} \left(\alpha_ {t , \ell} - (1 + \sqrt {2}) \epsilon\right)}{\frac {x _ {\ell} W x _ {\ell} ^ {\top}}{(t + 1) \| a _ {t} \| _ {2}} \left(\alpha_ {t , \ell} - (1 + \sqrt {2}) \epsilon\right) + \sum_ {k \neq \ell} \frac {x _ {k} W x _ {k} ^ {\top}}{(t + 1) \| a _ {t} \| _ {2}} \left(\alpha_ {t , k} + (1 + \sqrt {2}) \epsilon\right)}
$$

The proof is completed. 

## D EXPERIMENTAL DETAILS

## D.1 DATASETS

The details of the dataset used in the paper are as follows: 

- Essay (Verma et al., 2024). Each source of this dataset (human-written texts, various LLM-generated texts) contains 1,000 samples. The HGT portion comprises original IvyPanda essays that cover a wide array of subjects and academic levels, from high school through university. For the MGT portion, a tailored prompt was first crafted for each source essay using ChatGPT-turbo, and that prompt was then submitted to several LLMs, including GPT4All, ChatGPT, ChatGPT-turbo, ChatGLM, Dolly, and Claude, to generate machine-written essays. This workflow produced a diverse set of model-generated texts that remained aligned with the topics of their corresponding source documents. 

- Reuters (Verma et al., 2024). Built on the Reuters 50–50 authorship benchmark, this dataset contains 1,000 articles from 50 writers, with each author contributing 20 pieces. Replicating the pipeline used for the essay corpus, the team first asked ChatGPT-turbo to invent a headline for every article. Those auto-generated headlines were then embedded into prompts and submitted to multiple LLMs, including ChatGPT, GPT-4, ChatGPT-turbo, ChatGLM, Dolly, and Claude, to create the machine-generated texts. 

- TruthfulQA (Lin et al., 2022). It contains 817 questions covering 38 categories, including health, law, finance, and politics. The generated answers were produced by several large language models, including GPT4, ChatGPT-turbo, ChatGLM, Dolly, ChatGPT, and StableLM. 

- DetectRL (Wu et al., 2024). In this dataset, the human-authored portion is drawn from four sources: arXiv abstracts dated 2002–2017, XSum news reports, Writing Prompts stories, and Yelp reviews. These types were chosen because they are especially vulnerable to producing convincing but misleading content when LLMs are misapplied. From each source, 2,800 human texts are selected as HGTs. The machine-generated texts are created using four widely used LLMs—GPT-3.5-turbo (ChatGPT), PaLM-2-bison (Google-PaLM), and Llama-2-70b. The dataset further models practical adversarial settings: (1) a paraphrasing attack that rewrites MGTs with the Dipper paraphraser (Krishna et al., 2023) and Polish paraphraser, and (2) a mixed-text condition where 1/4 of machine-generated sentences is randomly replaced with human-written content while the label remains “machine-generated.” 

## D.2 BASELINES

A detailed description of the baselines used is shown below: 

- Likelihood (Solaiman et al., 2019). It uses an LLM to calculate the log probability of each token in a text. The average of these probabilities gives a detection score. A higher score indicates a greater chance that the text was generated by LLMs. 

- Log-Rank (Gehrmann et al., 2019). Its detection score is created by first using an LLM to rank each token in a text based on its predicted order within a given context. The logarithm of each word's predicted rank is then calculated. The final score is an average of these values, and a lower score is a strong indicator of machine-generated text. 

- Entropy (Gehrmann et al., 2019). Similar to Log-Rank, it calculates a score for a text by taking the average of each token's conditional entropy within its given context. A lower score suggests a higher likelihood that the text was generated by LLMs. 

- DetectGPT (Mitchell et al., 2023). It determines if a text is machine-generated by measuring how small changes affect its log probability. The underlying idea is that text created by LLMs is already a high-probability output. So, when it is slightly altered, the new version is likely to have a lower log probability. In contrast, making similar small changes to human-written text does not consistently lower the log probability; it can just as easily stay the same or increase. 

- Fast-DetectGPT (FastGPT) (Bao et al., 2024). To overcome the major computational expense of DetectGPT, this approach replaces DetectGPT's resource-intensive perturbation step with a more efficient sampling process. It identifies differences in token selection between humans and LLMs using a conditional probability curvature metric. 

- DNA-GPT (Yang et al., 2024). This method involves a two-step process. First, it cuts a text in half and uses the first part to prompt an LLM to generate a new continuation. Next, it examines the differences between the newly created segment and the original one. This comparison, done via N-gram analysis for black-box models or probability divergence for white-box models, reveals a clear distinction between how humans and machines generate text. 

- Repreguard (Chen et al., 2025). It utilizes key feature directions determined by a proxy model to project and score the representation of the text under test, comparing the result against a threshold. This approach demonstrates extremely high detection accuracy and robustness when dealing with out-of-distribution data and various types of attacks. 

- Lastde (Xu et al., 2025). It introduces time series analysis into machine-generated text detection, overcoming the limitations of traditional methods that ignore local discriminative information by collaboratively modeling the local dynamic features and global statistical indicators of token probability sequences. 

- FourierGPT (Xu et al., 2024). It proposes a detection method based on a likelihood spectrum perspective, which captures subtle differences between human and machine language by analyzing the relative changes in text likelihood values rather than their absolute values. 

- Binoculars Hans et al. (2024). It is a detection algorithm that requires no training data, accurately distinguishing between human and machine-generated text by comparing the score differences of a pair of pre-trained LLMs. 

## D.3 EXPERIMENTAL SCENARIO

To extensively evaluate the effectiveness of the proposed enhancement model, we conduct experiments in the following real-world scenarios: 

- Cross-LLM. To assess how well the proposed model works across different LLMs, we trained detectors on a single LLM's text and then tested it on a variety of LLMS' texts. The main body of the paper presents the results from training detectors on the GPT4All texts (Essay dataset) and Llama-2-70b texts (DetectRL), and then testing them on various LLMs, as shown in Table 2. Complete results for every training and testing combination can be found in Appendix E.3 (Tables 3-15). 

- Cross-Domain. The DetectRL dataset includes texts from four distinct domains: arXiv academic abstracts, XSum news articles, Writing Prompts stories, and Yelp Reviews. We utilized this dataset to evaluate the model's performance across various domains. To achieve this, we trained the detector on one domain and then tested it on the other domains. For this evaluation, all machine-generated texts were created using the default PaLM model. The heatmaps in Figs. 4, 21, and 22 illustrate the performance improvement achieved by our enhancement model compared to various baseline detection models. 

- Paraphrasing Attack. Studies have shown that MGT detection is vulnerable to paraphrase attacks. Therefore, this scenario is used to evaluate the robustness of the MGT detection method. Using the DetectRL dataset, which includes data from the Polish and Dipper paraphraser, we trained our detector on clean, original texts and then evaluated its robustness on these paraphrased texts. Specifically, we trained the detector using clean texts from Llama-2-70b and then tested it on paraphrased texts from several different LLMs. The results can be found in Fig. 6 and 24. 

- Mixed Text. Because a blend of human and machine-generated text is common in the real world, we use the mixed texts provided by DetectRL for evaluation. It involved randomly swapping out 25% of the sentences in an LLM-generated text with human-written ones. We conducted two separate experiments on this dataset: (1) The detector was trained on pure, non-mixed text and then tested for its ability to detect the mixed texts. (2) The detector was both trained and tested on the mixed texts themselves. The performance of the detector in these mixed settings is shown in Figs. 5 and 23. In each sub-figure, the detectors trained on original text are shown on the left, and those trained on mixed text are shown on the right. 

## D.4 THREAT MODEL AND PROXY SETTINGS

We conducted the experiments in a black-box setting: 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/53f643d069d11997a1ed4a7523e35f6f1ba08f39fbdf42f5e9347a7346e1159f.jpg)



(a) Essay Dataset



(b) Reuters Dataset



(c) DetectRL Dataset



Figure 9: Distribution of token scores obtained by the Log-Likelihood method without and with enhancement. It can be observed that the proposed method enhances the discriminative nature of the token scores.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/ffa10fcb5315831ab0e7c85673d58b4ec0bc084decc82138f8d1552760dafa14.jpg)



(a) Essay Dataset



(b) Reuters Dataset



(c) DetectRL Dataset



Figure 10: Distribution of token scores obtained by the Log-Rank method without and with enhancement. It can be observed that the proposed method enhances the discriminative nature of the token scores.


- Threat Model. We operate under a strictly Black-Box setting regarding the target LLM. We assume that the detector is entirely unknown to the LLM that generates the text (e.g., ChatGPT, Claude). Therefore, we employ a Proxy Model to compute token-level metrics (e.g., Log-Likelihood and Rank) for the candidate text, which are reasonable due to the transferability of the extracted metrics. 

- Proxy Models. For all baselines, we use GPT2 as the proxy model. Additionally, we use GPT2-XL as the scoring model for Fast-DetectGPT. To learn the 2x2 parameter $W_{mrf}$ introduced by our strategy, we perform training based on various LLM texts, as explicitly stated in the "Training Text" of the table. For example, in Table 2 in the paper, we learn $W_{mrf}$ on GPT4All texts of the Essay dataset and Llama-2-70b texts of the DetectRL dataset, and then test all candidate texts (from ChatGPT, Claude, or Dolly, etc.) using the learned detector. 

## D.5 PARAMETER SETTINGS

We conducted five independent experiments to ensure the consistency of our results, using five fixed random seeds (1-5). For all datasets, we used 10% of the data for training, while the remaining 90% was split evenly between validation and testing. To ensure a fair comparison, the enhanced models shared the same hyperparameters as their base models. In our enhancement model, there are two hyperparameters: the transition center $t_{0}$ in the positional weighting function $\beta(t)$ and the number of iterations T in the MRF layer. By default, we set $t_{0} = 30$ and T = 10 for the enhanced versions of all detectors across the three datasets, highlighting the flexibility of our approach. For training the MRF layer, we use a learning rate of 0.05 and train for 10 epochs. Hyperparameter sensitivity analyses are provided in Appendix E.9. 

## E MORE EXPERIMENTAL RESULTS

## E.1 MORE RESULTS OF TOKEN SCORE DISTRIBUTION BEFORE AND AFTER ENHANCEMENT

In addition to the partial results on DetectGPT presented in the main text, we also present the complete results about token-level detection score distributions for Log-Likelihood, Log-Rank, Entropy, and DetectGPT in Figs. 9, 10, 11, and 12. The results are similar to those in the main text: the original detector's scores show substantial overlap between human- and machine-generated text. However, after calibration using our proposed augmentation strategy, the scores achieve significantly improved discriminability. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/d05922e5e283ab68625ba663f0a12237935c8151dcec25fc50720310dd111140.jpg)



(a) Essay Dataset



(b) Reuters Dataset



(c) DetectRL Dataset



Figure 11: Distribution of token scores obtained by the Entropy method without and with enhancement. It can be observed that the proposed method enhances the discriminative nature of the token scores.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/d02a0cca51f8e3fc98e148ffa19a7887de9c6d036127078918342d367c809f61.jpg)



(a) Essay Dataset



(b) Reuters Dataset



(c) DetectRL Dataset



Figure 12: Distribution of token scores obtained by the DetectGPT method without and with enhancement. It can be observed that the proposed method enhances the discriminative nature of the token scores.


## E.2 MORE RESULTS OF CONTEXT TOKEN RELATIONSHIPS

In the main text, we demonstrated the existence of neighbor similarity (Fig. 2) and initial instability (Fig. 2) in token-level detection scores through experiments on the Essay dataset. Here, we provide additional supplementary results to further validate these relationships. 

To verify neighbor similarity, we provide additional results using Entropy and DetectGPT detection scores on the Essay dataset (Fig. 13), as well as results on the Reuters and DetectRL datasets (Figs. 14 and 15). In addition, we provide Chinese texts on the CUDRT dataset (Fig. 16). These supplementary experiments consistently show that the closer the tokens are, the more similar their detection scores are. 

Similarly, to verify initial instability, we provide additional results on the Essay dataset (Fig. 17), as well as results on the Reuters and DetectRL datasets (Figs. 18 and 19). In addition, we provide Chinese texts on the CUDRT dataset (Fig. 20). These results again demonstrate that token scores at the beginning of a text fluctuate significantly before gradually stabilizing. 

## E.3 MORE PERFORMANCE COMPARISON

Performance across Different LLMs. In the main text, we evaluated the cross-LLM performance of detectors trained on GPT4All (Essay) and Llama-2-70b (DetectRL) on various LLM texts in terms of AUROC, as shown in Table 2. This section aims to provide more comprehensive supplementary experimental results, including: (1) cross-LLM performance performance under the same experimental settings in terms of TPR@FPR-1% (Table 3), and (2) cross-LLM performance comparisons of detectors trained on other LLMs on the Essay, DetectRL, and Reuters datasets (Tables 4 to 15). These extensive experimental results are consistent with the conclusions of the main paper. Among the 888 cross-LLM evaluation settings, our proposed enhanced model achieved better performance than the original detector in 91.4% of the cases, highlighting the generalization ability and application value of this method on different models and datasets. 

Performance across Different Domains. In addition to the cross-domain performance improvements for Log-Likelihood and Log-Rank demonstrated in the main text, this section provides additional results for other detectors. Specifically, it includes improvements to the cross-domain performance of Entropy and DetectGPT (Fig. 21), as well as improvements to FastGPT and DNA-GPT (Fig. 22). Combining all experimental results, we reach the same conclusion as in the main text: in most experimental settings, detectors applied with our strategy significantly improve their cross-domain generalization capabilities. 

Performance against Mixed Texts. In addition to the AUROC performance comparison for mixed texts presented in the main text, this section provides additional performance comparisons at TPR@FPR-1%, as shown in Fig. 23. We reach consistent conclusions with those in the main text: in most experimental settings, the detector equipped with the proposed strategy significantly improves its ability to detect mixed text. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/6f4d340782782f7108f77dc3d295a52de50f667ba271ad27e3756190bd39dfab.jpg)



(a) Entropy Score


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/8f9d8b9a7e0afcedba3f0556ad10361586202bbefb68ed90dd87e30946c1a9f7.jpg)



(b) DetectGPT Score



Figure 13: The detection score distances (Mean Absolute Difference) of neighbors at different hops in the Essay dataset. Entropy and DetectGPT score are used here.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/96f8ee13124e38a71b461684dcd69d6be88fafebdda6d203a527ff671bc729be.jpg)



(a) Log-Likelihood Score


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/f52a2631455bbf63f305f3ee5165e69f8cf0164caeafd1020596da1ad4dc7ca3.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/b7678b28a00eb0fed04c484d1b1a983c6473585e3dd41c545e5a11cbbdcf9c33.jpg)



(c) Entropy Score


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/8249075586e987966b97785d3c799e98a1a830718624f5e7acf1cd99e2ca55ec.jpg)



Figure 14: The detection score distances (Mean Absolute Difference) of neighbors at different hops in the Reuters dataset. Log-Likelihood, Log-Rank, Entropy, and DetectGPT score are used here.


Performance against Paraphrasing Attacks. In addition to the AUROC performance comparison for paraphrasing texts presented in the main text, this section also presents a performance comparison under the TPR@FPR-1% metric, as shown in Fig. 24. We find that in most experimental settings, the detector applying our proposed strategy significantly improves its robustness against paraphrasing attacks, which is consistent with the conclusions drawn in the main text. 

## E.4 MORE PERFORMANCE COMPARISON ON THE QA DATASET

In this section, we explore testing on a challenging QA dataset, specifically the TruthfulQA dataset (Lin et al., 2022), which covers Q&A tasks in 38 domains (e.g., health, law, and finance). The results are shown in Table 16. As can be seen, our method consistently boosts performance. For example, DNA-GPT-M achieves a significant leap from 80.17% to 88.32% (+8.15%) in average AUROC. 

## E.5 MORE PERFORMANCE COMPARISON ON SHORT-TEXT

In this section, we performed a stress test on the Essay dataset with a strict 50-word limit. In Table 17, our calibration remains highly effective even with limited context. For example, we improve FastGPT by 5.35% and DNA-GPT by 3.03% on average. This confirms our calibration still holds even in short sequences. 

## E.6 ENHANCEMENT EXPERIMENTS ON MORE DETECTORS

In this section, we integrated more state-of-the-art metric-based baselines: RepreGuard Chen et al. (2025), Binoculars Hans et al. (2024), Lastde Xu et al. (2025), and FourierGPT Xu et al. (2024), and applied our strategy to them as “E” suffix. As shown in Tables 18-20, our method functions as a universal plug-in, enhancing these detectors. For example, on the Reuters dataset, our method improves RepreGuard by 8.97% and Binoculars by 4.42%. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/71a138dec65583338e175c1a89cb26620f881e341ae79f4ccc1886aa13bca2bd.jpg)



(a) Log-Likelihood Score


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/ea103f1c93d22cfd5a8db1922383f759b0a37824a7bdff1ee191db8d1194c0ad.jpg)



(b) Log-Rank Score


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/a75408c293461c9aea98af9f2fa217fdd90e487414d5d6e95e275eb3919825ff.jpg)



(c) Entropy Score


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/768c49564096c081e3c90f9ac62a57c71fc1582964682eb48a071fe7815c3a43.jpg)



(d) DetectGPT Score



Figure 15: The detection score distances (Mean Absolute Difference) of neighbors at different hops in the DetectRL dataset. Log-Likelihood, Log-Rank, Entropy, and DetectGPT score are used here.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/e13452cf2e4cafef315661cfb33b85f2313b7fbf20680a6c61f4ff979c37c904.jpg)



(a) Log-Likelihood Score


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/04b23301114db70579285e9ce9a29f9c18519012fc34a400bcfba243aa1e7fa7.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/880e4573e1d620ad898944b04238f14c5ebd0e8d60a2b12d292a097884da8fe5.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/d073dc26c682e6e978d65d31535213a8f783ff7f66868ef22b5470e855193bdc.jpg)



Figure 16: The detection score distances (Mean Absolute Difference) of neighbors at different hops in the CUDRT dataset. Log-Likelihood, Log-Rank, Entropy, and DetectGPT score are used here.


## E.7 PERFORMANCE COMPARISON AGAINST PERTURBATION TEXTS

In this section, we evaluated the robustness against perturbed texts. Here, we chose two distinct attack methods in the DetectRL dataset: Back-Translation and Word Perturbation. The results are shown in Tables 21. Across both perturbation scenarios, detectors equipped with our Markov-informed calibration (e.g., Likelihood-M, Log-Rank-M) consistently outperformed the uncalibrated ones, typically by margins of 10-20%. These findings align perfectly with our existing evaluation on Paraphrasing Attacks, which shows that our method effectively mitigates the performance degradation typically caused by text perturbations. 

These encouraging gains are consistent with the fundamental theory of Markov random fields. The adversarial perturbation acts as token-level salt-and-pepper noise, while our Markov-informed calibration strategy (modeling Neighbor Similarity) leverages the remaining uncorrupted context to smooth out these local anomalies, recovering the latent detection signal. 

## E.8 MORE RESULTS OF ABLATION STUDY

In addition to the ablation experiments on the position weight function and MRF layer presented in the main text for the Essay dataset, we also provide ablation results for the Reuters and DetectRL datasets, as shown in Figs. 25 and 26. These experimental results are consistent with the conclusions of the main text: removing either component leads to a significant performance drop; even retaining only one of them outperforms the baseline detector, strongly demonstrating the effectiveness of both components. 

## E.9 SENSITIVITY ANALYSIS

Sensitivity w.r.t. transition center $t_{0}$ . In our experiments, we set the default transition center $t_{0}$ of the position weighting function to 30. This section examines the effect of varying $t_{0}$ values on detection performance. The AUROC and TPR@FPR-1% results are shown in Figs. 27 and 28, respectively. The experimental results show that detection performance steadily improves with increasing $t_{0}$ values, which is consistent with the conclusions of the ablation experiments and demonstrates the effectiveness of the position weighting function. However, performance improvement is not infinite. When $t_{0}$ values are too large, performance gradually saturates or even declines. This is likely because excessively large $t_{0}$ filters out useful token scores. Therefore, a trade-off is necessary. Through sensitivity analysis, we recommend setting $t_{0}$ values between 20 and 30. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/bf3ca9af7d3ea1a5e090af96382dd44685524890894c752d27f989bf83f86248.jpg)



(a) Entropy Score


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/68304947f2e837c2fdf12df5584642154b846c10cf2172e54357d9a45d2b587a.jpg)



(b) DetectGPT Score



Figure 17: The score distances (Mean Absolute Difference) of 1-hop neighbors at different positions in the Essay dataset. Entropy and DetectGPT score are used here.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/41e0b4c91b27095193eac005208b58ef67a17512325092a31810c15402ee561e.jpg)



(a) Log-Likelihood Score


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/bbb8fa29378d83c0c49753b7b82d41e9406e2e5db025e101e7915065a3ed459b.jpg)



(b) Log-Rank Score


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/f91cbfbc459b551304367d70a242620672f963e9b25fc528b3635fba21ea4153.jpg)



(c) Entropy Score


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/c976707fd8405f47d79d552d1a882b73c66edc619f472857c55a89db7b9e3a3b.jpg)



(d) DetectGPT Score



Figure 18: The score distances (Mean Absolute Difference) of 1-hop neighbors at different positions in the Reuters dataset. Log-Likelihood, Log-Rank, Entropy, and DetectGPT score are used here.


Sensitivity w.r.t. iterations T in MRF layer. We compute the posterior probability of the Markov random field using a multi-step iterative approach. To this end, we evaluate the impact of varying the number of iterations on detection performance, as shown in Figs. 29 and 30. The experimental results demonstrate that multi-step iterative computation significantly enhances detection performance compared to single-step computation, underscoring its importance in score calibration. However, performance may degrade with increasing the number of iterations, possibly due to oversmoothing of the detection scores. Based on our sensitivity analysis, we recommend setting the number of iterations to 10. 

## E.10 MORE RESULTS COMPARING WITH NEURAL NETWORK CALIBRATION

In the main text, we demonstrated, through experimental results using the Log-Likelihood score, that while neural network-based detection score correction performs well on the intra-LLM, its performance drops sharply on the cross-LLM. This section further presents comparative results using other detection scores, as shown in Figs. 31 and 32. Similar experimental findings further demonstrate that this method does not truly learn universal score correction capabilities, but rather overfits the training data. This strongly emphasizes the superiority and rationality of our proposed Markov-informed score calibration method. 

## E.11 MRF vs. HMM

In the paper, we chose the MRF over Markov chain for two key reasons: 

- Theoretical rationality. A Markov chain typically models unidirectional dependencies (current state depends on previous state). However, in MGT detection, we identified the Neighbor Similarity property, which is inherently bidirectional. An MRF naturally captures these bidirectional dependencies, whereas a standard Markov Chain struggles to utilize future context during inference without complex multi-pass algorithms. 

- Computational Efficiency. A raw MRF may be complicated. However, as described in Section 4.2, our Mean-Field Approximation transforms the problem into lightweight iterative matrix multiplications. This allows for massive parallelization on GPUs. In contrast, exact inference in HMMs (e.g., Forward-Backward algorithm) is inherently sequential, which can be slower on modern hardware despite having similar theoretical complexity $\mathcal{O}(N)$ . 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/4de193507c11b25e5e31d3ebbf5bdd52917981f3fbecec3f0666bf3e5675bd22.jpg)



(a) Log-Likelihood Score


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/cd2cf65a61e423dd3ce9c00739fff08b71d4e04885e9f71abbdac9681936c742.jpg)



(b) Log-Rank Score


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/b1fff2f0a61395ee08f04e6b2a70c7a7458dfd414c7543054c10a3b05fad53d1.jpg)



(c) Entropy Score


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/a27ec62ddb534110f9a40ce910f65fa601dc9eec813fc7922ef6289cd6cf3382.jpg)



(d) DetectGPT Score



Figure 19: The score distances (Mean Absolute Difference) of 1-hop neighbors at different positions in the DetectRL dataset. Log-Likelihood, Log-Rank, Entropy, and DetectGPT score are used here.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/154a3c9a04e39402d353997004a41b53bb90d71d518251ce7f804b701149a6f6.jpg)



(a) Log-Likelihood Score


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/f359cf7b08ecddb04c89db58e120e05639c3f3285cd71fbedeb871813fbc9f4e.jpg)



(b) Log-Rank Score


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/2d0acdc4eeb21fd244480e8d484ece0f0f3803c0ce914378efaef0c24834a304.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/97d6c95be029e33387a2a47a10eef9bbb782a2bb0c9782d71bf75845a549e570.jpg)



(d) DetectGPT Score



Figure 20: The score distances (Mean Absolute Difference) of 1-hop neighbors at different positions in the CUDRT dataset. Log-Likelihood, Log-Rank, Entropy, and DetectGPT score are used here.


To further verify, we implemented a Hidden Markov Model (HMM) based calibration strategy and applied it to existing detectors ("HMM" suffix). As shown in Table 22 (left part), while the HMM improves upon the original baseline, it consistently underperforms our MRF approach. This validates the necessity of modeling bidirectional dependencies. For computational efficiency (see the right part of Table 22), our MRF implementation is actually faster in practice due to GPU-friendly matrix operations. 

## E.12 COMPARISON WITH ENHANCED METHOD

TOCSIN is an enhanced metric-based approach which enhances the detector by dynamically adjusting the decision threshold. Clearly, it operates on a different dimension than our method. Therefore, in addition to supplementing TOCSIN-enhanced baselines (suffix "-T"), our Markov calibration can be further applied to the top of TOCSIN (suffix "-TM") to further explore its detection potential, as shown in Table 23. It can be seen that our method further improves the performance of TOCSIN-enhanced baselines in most cases. This significant gain suggests that TOCSIN (adjusting dynamic thresholds using token cohesiveness score) and our method (calibration token-level detection score) focus on different aspects, achieving complementarity. 

## E.13 ANALYSIS OF CROSS-DOMAIN FAILURE CASES

Aside from a few random performance fluctuations, most failures occur when xsum (training data) is applied to arXiv, writing, and yelp (test data). Given that our proposed calibration strategy hinges on learning a reward/penalty matrix $W_{mrf}$ to encourage neighbor similarity, we examined the distance of neighbor detection score across various domains, as summarized in Fig. 33. 

As observed, xsum exhibits considerably lower neighbor similarity (larger distance), whereas the other domains show higher similarity. This makes sense because, as a news summarization dataset, XSum is extremely information-dense, with a highly compressed syntactic structure and a lack of redundancy and transitional words seen in longer texts (such as arXiv). Consequently, adjacent tokens in XSum experience more drastic fluctuations in detection scores. Therefore, when trained on xsum, the MRF learns parameters $W_{mrf}$ that tolerate high local variance (treating it as a normal feature of the domain). When these parameters are applied to "smoother" domains like arXiv, the MRF fails to enforce the tighter consistency constraints required for those domains, leading to the observed performance drop. Encouragingly, compared to these occasional failure modes, the extensive enhancements highlight the practicality of our method. 


Table 3: Performance concerning TPR@FPR-1% (%) on Essay (left) and DetectRL (right).


<table><tr><td rowspan="2">Method</td><td colspan="7">Essay (Training Text: GPT4All)</td><td colspan="4">DetectRL (Training Text: Llama-2-70b)</td></tr><tr><td>GPT4All</td><td>ChatGPT</td><td>Dolly</td><td>ChatGLM</td><td>Claude</td><td>ChatGPT-turbo</td><td>Avg.</td><td>Llama-2-70b</td><td>ChatGPT</td><td>Google-PaLM</td><td>Avg.</td></tr><tr><td>Likelihood</td><td><eq>46.33 \pm 16.49</eq></td><td><eq>68.62 \pm 13.32</eq></td><td><eq>20.67 \pm 10.79</eq></td><td><eq>92.86 \pm 4.84</eq></td><td><eq>12.31 \pm 6.29</eq></td><td><eq>73.60 \pm 14.63</eq></td><td>52.4</td><td><eq>38.37 \pm 0.92</eq></td><td><eq>10.21 \pm 1.40</eq></td><td><eq>25.66 \pm 2.41</eq></td><td>24.75</td></tr><tr><td>Likelihood-M</td><td><eq>87.47 \pm 3.42</eq></td><td><eq>90.36 \pm 2.93</eq></td><td><eq>55.47 \pm 5.03</eq></td><td><eq>97.19 \pm 1.11</eq></td><td><eq>40.40 \pm 7.87</eq></td><td><eq>96.27 \pm 1.27</eq></td><td>77.86</td><td><eq>46.03 \pm 1.70</eq></td><td><eq>10.51 \pm 3.92</eq></td><td><eq>36.37 \pm 2.31</eq></td><td>30.97</td></tr><tr><td>Log-Rank</td><td><eq>62.69 \pm 13.36</eq></td><td><eq>79.07 \pm 7.89</eq></td><td><eq>25.11 \pm 8.35</eq></td><td><eq>95.71 \pm 2.05</eq></td><td><eq>19.20 \pm 8.26</eq></td><td><eq>80.89 \pm 10.73</eq></td><td>60.44</td><td><eq>42.05 \pm 0.84</eq></td><td><eq>12.44 \pm 1.71</eq></td><td><eq>22.84 \pm 1.28</eq></td><td>25.78</td></tr><tr><td>Log-Rank-M</td><td><eq>87.38 \pm 2.50</eq></td><td><eq>88.89 \pm 2.84</eq></td><td><eq>52.12 \pm 4.18</eq></td><td><eq>98.04 \pm 0.43</eq></td><td><eq>31.69 \pm 4.15</eq></td><td><eq>94.22 \pm 2.46</eq></td><td>75.39</td><td><eq>50.56 \pm 1.44</eq></td><td><eq>13.94 \pm 2.36</eq></td><td><eq>41.51 \pm 2.69</eq></td><td>35.34</td></tr><tr><td>Entropy</td><td><eq>2.73 \pm 0.69</eq></td><td><eq>7.16 \pm 3.17</eq></td><td><eq>2.29 \pm 1.36</eq></td><td><eq>6.88 \pm 3.05</eq></td><td><eq>3.91 \pm 2.24</eq></td><td><eq>13.24 \pm 7.60</eq></td><td>6.04</td><td><eq>2.03 \pm 0.73</eq></td><td><eq>0.25 \pm 0.16</eq></td><td><eq>6.95 \pm 0.78</eq></td><td>3.07</td></tr><tr><td>Entropy-M</td><td><eq>15.63 \pm 0.87</eq></td><td><eq>41.24 \pm 2.25</eq></td><td><eq>16.95 \pm 3.12</eq></td><td><eq>40.85 \pm 4.37</eq></td><td><eq>23.20 \pm 1.70</eq></td><td><eq>49.91 \pm 5.02</eq></td><td>31.30</td><td><eq>2.69 \pm 0.39</eq></td><td><eq>0.67 \pm 0.10</eq></td><td><eq>9.17 \pm 1.10</eq></td><td>4.18</td></tr><tr><td>DetectGPT</td><td><eq>0.00 \pm 0.00</eq></td><td><eq>0.00 \pm 0.00</eq></td><td><eq>0.00 \pm 0.00</eq></td><td><eq>0.00 \pm 0.00</eq></td><td><eq>0.89 \pm 0.24</eq></td><td><eq>0.00 \pm 0.00</eq></td><td>0.15</td><td><eq>4.15 \pm 0.40</eq></td><td><eq>2.94 \pm 0.62</eq></td><td><eq>1.71 \pm 0.47</eq></td><td>2.93</td></tr><tr><td>DetectGPT-M</td><td><eq>41.23 \pm 29.53</eq></td><td><eq>46.98 \pm 25.36</eq></td><td><eq>18.38 \pm 20.21</eq></td><td><eq>41.83 \pm 44.80</eq></td><td><eq>8.80 \pm 9.64</eq></td><td><eq>65.87 \pm 21.66</eq></td><td>37.18</td><td><eq>32.41 \pm 9.58</eq></td><td><eq>7.59 \pm 4.05</eq></td><td><eq>25.14 \pm 4.53</eq></td><td>21.71</td></tr><tr><td>FastGPT</td><td><eq>1.59 \pm 0.32</eq></td><td><eq>1.64 \pm 0.67</eq></td><td><eq>0.29 \pm 0.10</eq></td><td><eq>2.72 \pm 0.68</eq></td><td><eq>3.47 \pm 1.83</eq></td><td><eq>16.89 \pm 6.66</eq></td><td>4.43</td><td><eq>11.35 \pm 2.01</eq></td><td><eq>3.78 \pm 1.20</eq></td><td><eq>5.02 \pm 0.85</eq></td><td>6.72</td></tr><tr><td>FastGPT-M</td><td><eq>23.92 \pm 10.29</eq></td><td><eq>30.00 \pm 9.20</eq></td><td><eq>32.89 \pm 16.52</eq></td><td><eq>55.54 \pm 7.80</eq></td><td><eq>0.67 \pm 0.56</eq></td><td><eq>16.44 \pm 31.78</eq></td><td>26.58</td><td><eq>15.23 \pm 3.23</eq></td><td><eq>6.28 \pm 1.72</eq></td><td><eq>13.52 \pm 5.63</eq></td><td>11.68</td></tr><tr><td>DNA-GPT</td><td><eq>57.68 \pm 3.34</eq></td><td><eq>63.24 \pm 6.54</eq></td><td><eq>22.20 \pm 3.85</eq></td><td><eq>84.82 \pm 4.14</eq></td><td><eq>16.67 \pm 3.47</eq></td><td><eq>56.27 \pm 6.45</eq></td><td>50.15</td><td><eq>36.56 \pm 1.56</eq></td><td><eq>19.70 \pm 0.52</eq></td><td><eq>30.38 \pm 2.75</eq></td><td>28.88</td></tr><tr><td>DNA-GPT-M</td><td><eq>93.85 \pm 0.97</eq></td><td><eq>88.44 \pm 4.22</eq></td><td><eq>56.42 \pm 10.82</eq></td><td><eq>96.83 \pm 0.41</eq></td><td><eq>33.33 \pm 7.90</eq></td><td><eq>86.22 \pm 6.52</eq></td><td>75.85</td><td><eq>42.52 \pm 1.26</eq></td><td><eq>19.11 \pm 2.04</eq></td><td><eq>34.81 \pm 1.54</eq></td><td>32.15</td></tr></table>


Table 4: Performance on Essay dataset. The detection models are trained on text generated by ChatGPT.


<table><tr><td>Metric</td><td>Method</td><td>GPT4All</td><td>ChatGPT</td><td>Dolly</td><td>ChatGLM</td><td>Claude</td><td>ChatGPT-turbo</td><td>avg</td></tr><tr><td rowspan="12">TPR@FPR-1%</td><td>Likelihood</td><td><eq>46.24 \pm 16.60</eq></td><td><eq>68.62 \pm 13.32</eq></td><td><eq>20.67 \pm 10.79</eq></td><td><eq>92.81 \pm 4.82</eq></td><td><eq>12.31 \pm 6.29</eq></td><td><eq>73.60 \pm 14.63</eq></td><td>52.38</td></tr><tr><td>Likelihood-M</td><td><eq>89.38 \pm 4.05</eq></td><td><eq>92.00 \pm 2.84</eq></td><td><eq>59.19 \pm 5.98</eq></td><td><eq>97.10 \pm 0.69</eq></td><td><eq>42.22 \pm 6.32</eq></td><td><eq>96.80 \pm 1.23</eq></td><td>79.45</td></tr><tr><td>Log-Rank</td><td><eq>62.69 \pm 13.36</eq></td><td><eq>79.07 \pm 7.89</eq></td><td><eq>25.11 \pm 8.35</eq></td><td><eq>95.71 \pm 2.05</eq></td><td><eq>19.16 \pm 8.22</eq></td><td><eq>80.89 \pm 10.73</eq></td><td>60.44</td></tr><tr><td>Log-Rank-M</td><td><eq>87.15 \pm 2.62</eq></td><td><eq>88.80 \pm 2.81</eq></td><td><eq>52.12 \pm 4.32</eq></td><td><eq>98.04 \pm 0.43</eq></td><td><eq>31.33 \pm 4.10</eq></td><td><eq>94.22 \pm 2.46</eq></td><td>75.28</td></tr><tr><td>Entropy</td><td><eq>2.73 \pm 0.69</eq></td><td><eq>7.16 \pm 3.17</eq></td><td><eq>2.29 \pm 1.36</eq></td><td><eq>6.88 \pm 3.05</eq></td><td><eq>3.91 \pm 2.24</eq></td><td><eq>13.24 \pm 7.60</eq></td><td>6.04</td></tr><tr><td>Entropy-M</td><td><eq>12.76 \pm 2.18</eq></td><td><eq>34.84 \pm 5.15</eq></td><td><eq>14.27 \pm 2.63</eq></td><td><eq>36.96 \pm 2.81</eq></td><td><eq>19.78 \pm 3.76</eq></td><td><eq>43.20 \pm 6.33</eq></td><td>26.97</td></tr><tr><td>DetectGPT</td><td><eq>0.00 \pm 0.00</eq></td><td><eq>0.00 \pm 0.00</eq></td><td><eq>0.00 \pm 0.00</eq></td><td><eq>0.00 \pm 0.00</eq></td><td><eq>0.84 \pm 0.51</eq></td><td><eq>2.36 \pm 2.52</eq></td><td>0.53</td></tr><tr><td>DetectGPT-M</td><td><eq>41.09 \pm 30.67</eq></td><td><eq>42.04 \pm 25.95</eq></td><td><eq>9.55 \pm 9.01</eq></td><td><eq>39.02 \pm 38.26</eq></td><td><eq>10.13 \pm 10.28</eq></td><td><eq>68.49 \pm 23.15</eq></td><td>35.05</td></tr><tr><td>FastGPT</td><td><eq>1.59 \pm 0.32</eq></td><td><eq>1.64 \pm 0.67</eq></td><td><eq>0.29 \pm 0.10</eq></td><td><eq>2.72 \pm 0.68</eq></td><td><eq>3.47 \pm 1.83</eq></td><td><eq>16.89 \pm 6.66</eq></td><td>4.43</td></tr><tr><td>FastGPT-M</td><td><eq>19.73 \pm 11.16</eq></td><td><eq>26.18 \pm 12.42</eq></td><td><eq>24.82 \pm 20.23</eq></td><td><eq>45.76 \pm 21.65</eq></td><td><eq>3.38 \pm 5.34</eq></td><td><eq>26.04 \pm 32.58</eq></td><td>24.32</td></tr><tr><td>DNA-GPT</td><td><eq>40.77 \pm 8.33</eq></td><td><eq>73.69 \pm 3.78</eq></td><td><eq>19.95 \pm 7.42</eq></td><td><eq>82.59 \pm 11.71</eq></td><td><eq>18.04 \pm 3.53</eq></td><td><eq>63.82 \pm 6.62</eq></td><td>49.81</td></tr><tr><td>DNA-GPT-M</td><td><eq>89.02 \pm 4.14</eq></td><td><eq>95.07 \pm 1.33</eq></td><td><eq>65.97 \pm 4.57</eq></td><td><eq>99.24 \pm 0.36</eq></td><td><eq>39.56 \pm 4.01</eq></td><td><eq>94.62 \pm 1.43</eq></td><td>80.58</td></tr><tr><td rowspan="12">AUROC</td><td>Likelihood</td><td><eq>96.16 \pm 0.30</eq></td><td><eq>98.79 \pm 0.19</eq></td><td><eq>90.90 \pm 1.33</eq></td><td><eq>99.29 \pm 0.25</eq></td><td><eq>92.76 \pm 0.23</eq></td><td><eq>99.13 \pm 0.19</eq></td><td>96.17</td></tr><tr><td>Likelihood-M</td><td><eq>98.77 \pm 0.10</eq></td><td><eq>99.58 \pm 0.14</eq></td><td><eq>94.80 \pm 0.87</eq></td><td><eq>99.56 \pm 0.19</eq></td><td><eq>94.69 \pm 0.42</eq></td><td><eq>99.77 \pm 0.14</eq></td><td>97.86</td></tr><tr><td>Log-Rank</td><td><eq>96.55 \pm 0.31</eq></td><td><eq>98.95 \pm 0.13</eq></td><td><eq>90.08 \pm 1.28</eq></td><td><eq>99.36 \pm 0.13</eq></td><td><eq>92.01 \pm 0.20</eq></td><td><eq>99.24 \pm 0.15</eq></td><td>96.03</td></tr><tr><td>Log-Rank-M</td><td><eq>98.56 \pm 0.06</eq></td><td><eq>99.40 \pm 0.09</eq></td><td><eq>93.81 \pm 0.92</eq></td><td><eq>99.56 \pm 0.08</eq></td><td><eq>92.88 \pm 0.29</eq></td><td><eq>99.64 \pm 0.10</eq></td><td>97.31</td></tr><tr><td>Entropy</td><td><eq>74.19 \pm 1.62</eq></td><td><eq>89.49 \pm 0.33</eq></td><td><eq>73.26 \pm 1.48</eq></td><td><eq>84.11 \pm 0.77</eq></td><td><eq>86.58 \pm 0.66</eq></td><td><eq>95.94 \pm 0.35</eq></td><td>83.93</td></tr><tr><td>Entropy-M</td><td><eq>83.33 \pm 0.66</eq></td><td><eq>93.24 \pm 0.15</eq></td><td><eq>81.17 \pm 1.04</eq></td><td><eq>91.38 \pm 0.37</eq></td><td><eq>88.28 \pm 0.49</eq></td><td><eq>96.86 \pm 0.21</eq></td><td>89.04</td></tr><tr><td>DetectGPT</td><td><eq>50.15 \pm 0.99</eq></td><td><eq>50.53 \pm 3.65</eq></td><td><eq>48.04 \pm 7.27</eq></td><td><eq>49.25 \pm 1.58</eq></td><td><eq>51.21 \pm 8.39</eq></td><td><eq>55.34 \pm 31.67</eq></td><td>50.75</td></tr><tr><td>DetectGPT-M</td><td><eq>94.98 \pm 3.26</eq></td><td><eq>96.35 \pm 2.60</eq></td><td><eq>85.82 \pm 6.13</eq></td><td><eq>94.20 \pm 4.48</eq></td><td><eq>84.06 \pm 6.47</eq></td><td><eq>98.58 \pm 1.24</eq></td><td>92.33</td></tr><tr><td>FastGPT</td><td><eq>64.63 \pm 1.53</eq></td><td><eq>67.68 \pm 1.70</eq></td><td><eq>47.17 \pm 1.53</eq></td><td><eq>71.08 \pm 1.51</eq></td><td><eq>75.31 \pm 0.90</eq></td><td><eq>88.62 \pm 0.67</eq></td><td>69.08</td></tr><tr><td>FastGPT-M</td><td><eq>86.68 \pm 3.28</eq></td><td><eq>90.82 \pm 3.86</eq></td><td><eq>72.23 \pm 23.91</eq></td><td><eq>93.53 \pm 3.86</eq></td><td><eq>65.24 \pm 12.36</eq></td><td><eq>71.87 \pm 21.80</eq></td><td>80.06</td></tr><tr><td>DNA-GPT</td><td><eq>96.28 \pm 0.32</eq></td><td><eq>98.87 \pm 0.21</eq></td><td><eq>92.85 \pm 0.83</eq></td><td><eq>99.32 \pm 0.27</eq></td><td><eq>91.65 \pm 0.72</eq></td><td><eq>98.51 \pm 0.31</eq></td><td>96.25</td></tr><tr><td>DNA-GPT-M</td><td><eq>99.32 \pm 0.13</eq></td><td><eq>99.79 \pm 0.07</eq></td><td><eq>97.42 \pm 0.56</eq></td><td><eq>99.90 \pm 0.06</eq></td><td><eq>95.65 \pm 0.51</eq></td><td><eq>99.76 \pm 0.07</eq></td><td>98.64</td></tr></table>

## E.14 COMPARISON WITH MODEL-BASED DETECTORS

As shown in Fig. 34, we compare the enhanced versions of metric-based methods with model-based detection methods, including ChatGPT-D and MPU. Experimental results show that while model-based methods demonstrate superior performance on the DetectRL dataset, they underperform state-of-the-art metric-based methods, such as DNA-GPT, on the Essay and Reuters datasets. Notably, the significant performance gap between model-based methods in intra-LLM and cross-LLM further confirms their increased risk of overfitting to the training data. This observation is consistent with our intention of focusing on metric-based detection methods. 

## E.15 RUNNING TIME

Table 24 shows the training and inference runtimes on different datasets. As discussed in the main text, our proposed Markov-based score refinement module can be implemented in constant time via sparse-dense matrix multiplication. Therefore, the additional time overhead introduced by this module is negligible compared to the time-consuming score calculation, highlighting the flexibility and practicality of our approach in practical applications. 

## F THE USE OF LARGE LANGUAGE MODELS

In our paper, we used LLMs to polish the language and correct grammatical errors. LLMs were not used to generate novel research ideas, design experiments, analyze results, or write substantive technical content. We ensure that the use of large language models is responsible and adheres to academic and ethical standards. 


Table 5: Performance on Essay dataset. The detection models are trained on text generated by ChatGPT-turbo.


<table><tr><td>Metric</td><td>Method</td><td>GPT4All</td><td>ChatGPT</td><td>Dolly</td><td>ChatGLM</td><td>Claude</td><td>ChatGPT-turbo</td><td>avg</td></tr><tr><td rowspan="12">TPR@FPR-1%</td><td>Likelihood</td><td><eq>46.29 \pm 16.55</eq></td><td><eq>68.62 \pm 13.32</eq></td><td><eq>20.67 \pm 10.79</eq></td><td><eq>92.86 \pm 4.84</eq></td><td><eq>12.31 \pm 6.29</eq></td><td><eq>73.60 \pm 14.63</eq></td><td>52.39</td></tr><tr><td>Likelihood-M</td><td><eq>87.61 \pm 3.74</eq></td><td><eq>88.31 \pm 3.64</eq></td><td><eq>61.29 \pm 7.12</eq></td><td><eq>94.20 \pm 2.40</eq></td><td><eq>41.11 \pm 7.93</eq></td><td><eq>95.38 \pm 2.27</eq></td><td>77.98</td></tr><tr><td>Log-Rank</td><td><eq>62.69 \pm 13.36</eq></td><td><eq>79.07 \pm 7.89</eq></td><td><eq>25.11 \pm 8.35</eq></td><td><eq>95.71 \pm 2.05</eq></td><td><eq>19.16 \pm 8.22</eq></td><td><eq>80.89 \pm 10.73</eq></td><td>60.44</td></tr><tr><td>Log-Rank-M</td><td><eq>86.70 \pm 0.59</eq></td><td><eq>87.47 \pm 3.17</eq></td><td><eq>54.61 \pm 4.05</eq></td><td><eq>93.75 \pm 3.55</eq></td><td><eq>35.91 \pm 2.65</eq></td><td><eq>94.13 \pm 1.68</eq></td><td>75.43</td></tr><tr><td>Entropy</td><td><eq>2.73 \pm 0.69</eq></td><td><eq>7.16 \pm 3.17</eq></td><td><eq>2.29 \pm 1.36</eq></td><td><eq>6.88 \pm 3.05</eq></td><td><eq>3.91 \pm 2.24</eq></td><td><eq>13.24 \pm 7.60</eq></td><td>6.04</td></tr><tr><td>Entropy-M</td><td><eq>11.62 \pm 0.59</eq></td><td><eq>30.67 \pm 6.02</eq></td><td><eq>13.56 \pm 1.59</eq></td><td><eq>32.54 \pm 2.63</eq></td><td><eq>17.33 \pm 5.32</eq></td><td><eq>39.91 \pm 8.11</eq></td><td>24.27</td></tr><tr><td>DetectGPT</td><td><eq>0.00 \pm 0.00</eq></td><td><eq>0.00 \pm 0.00</eq></td><td><eq>0.00 \pm 0.00</eq></td><td><eq>0.00 \pm 0.00</eq></td><td><eq>0.53 \pm 0.52</eq></td><td><eq>3.07 \pm 1.96</eq></td><td>0.60</td></tr><tr><td>DetectGPT-M</td><td><eq>14.03 \pm 13.72</eq></td><td><eq>17.07 \pm 8.79</eq></td><td><eq>3.05 \pm 2.54</eq></td><td><eq>6.43 \pm 3.80</eq></td><td><eq>5.11 \pm 3.98</eq></td><td><eq>44.71 \pm 11.72</eq></td><td>15.07</td></tr><tr><td>FastGPT</td><td><eq>1.59 \pm 0.32</eq></td><td><eq>1.64 \pm 0.67</eq></td><td><eq>0.29 \pm 0.10</eq></td><td><eq>2.72 \pm 0.68</eq></td><td><eq>3.47 \pm 1.83</eq></td><td><eq>16.89 \pm 6.66</eq></td><td>4.43</td></tr><tr><td>FastGPT-M</td><td><eq>8.97 \pm 4.52</eq></td><td><eq>15.33 \pm 15.63</eq></td><td><eq>0.72 \pm 0.60</eq></td><td><eq>20.09 \pm 18.21</eq></td><td><eq>7.51 \pm 6.83</eq></td><td><eq>49.82 \pm 22.08</eq></td><td>17.07</td></tr><tr><td>DNA-GPT</td><td><eq>0.73 \pm 0.33</eq></td><td><eq>0.76 \pm 0.30</eq></td><td><eq>2.63 \pm 0.81</eq></td><td><eq>0.13 \pm 0.11</eq></td><td><eq>21.64 \pm 3.45</eq></td><td><eq>0.00 \pm 0.00</eq></td><td>4.31</td></tr><tr><td>DNA-GPT-M</td><td><eq>39.41 \pm 2.68</eq></td><td><eq>38.62 \pm 2.34</eq></td><td><eq>39.47 \pm 2.11</eq></td><td><eq>38.04 \pm 2.49</eq></td><td><eq>39.38 \pm 1.86</eq></td><td><eq>38.67 \pm 2.19</eq></td><td>38.93</td></tr><tr><td rowspan="12">AUROC</td><td>Likelihood</td><td><eq>96.16 \pm 0.30</eq></td><td><eq>98.79 \pm 0.19</eq></td><td><eq>90.90 \pm 1.33</eq></td><td><eq>99.29 \pm 0.25</eq></td><td><eq>92.76 \pm 0.23</eq></td><td><eq>99.13 \pm 0.19</eq></td><td>96.17</td></tr><tr><td>Likelihood-M</td><td><eq>98.65 \pm 0.21</eq></td><td><eq>99.39 \pm 0.20</eq></td><td><eq>93.95 \pm 1.15</eq></td><td><eq>99.13 \pm 0.43</eq></td><td><eq>94.66 \pm 0.39</eq></td><td><eq>99.77 \pm 0.12</eq></td><td>97.59</td></tr><tr><td>Log-Rank</td><td><eq>96.55 \pm 0.31</eq></td><td><eq>98.95 \pm 0.13</eq></td><td><eq>90.08 \pm 1.28</eq></td><td><eq>99.36 \pm 0.13</eq></td><td><eq>92.01 \pm 0.20</eq></td><td><eq>99.24 \pm 0.15</eq></td><td>96.03</td></tr><tr><td>Log-Rank-M</td><td><eq>98.37 \pm 0.21</eq></td><td><eq>99.17 \pm 0.29</eq></td><td><eq>92.96 \pm 1.51</eq></td><td><eq>99.17 \pm 0.37</eq></td><td><eq>92.96 \pm 0.43</eq></td><td><eq>99.62 \pm 0.11</eq></td><td>97.04</td></tr><tr><td>Entropy</td><td><eq>74.19 \pm 1.62</eq></td><td><eq>89.49 \pm 0.34</eq></td><td><eq>73.26 \pm 1.48</eq></td><td><eq>84.11 \pm 0.77</eq></td><td><eq>86.58 \pm 0.66</eq></td><td><eq>95.94 \pm 0.35</eq></td><td>83.93</td></tr><tr><td>Entropy-M</td><td><eq>82.88 \pm 0.85</eq></td><td><eq>93.08 \pm 0.29</eq></td><td><eq>80.90 \pm 0.89</eq></td><td><eq>91.18 \pm 0.37</eq></td><td><eq>88.57 \pm 0.54</eq></td><td><eq>96.90 \pm 0.24</eq></td><td>88.92</td></tr><tr><td>DetectGPT</td><td><eq>49.19 \pm 0.58</eq></td><td><eq>53.60 \pm 0.77</eq></td><td><eq>42.52 \pm 0.84</eq></td><td><eq>49.59 \pm 1.70</eq></td><td><eq>58.46 \pm 0.60</eq></td><td><eq>82.10 \pm 1.25</eq></td><td>55.91</td></tr><tr><td>DetectGPT-M</td><td><eq>91.65 \pm 2.28</eq></td><td><eq>93.38 \pm 2.31</eq></td><td><eq>77.58 \pm 5.63</eq></td><td><eq>90.54 \pm 2.74</eq></td><td><eq>79.82 \pm 9.52</eq></td><td><eq>97.61 \pm 0.92</eq></td><td>88.43</td></tr><tr><td>FastGPT</td><td><eq>64.63 \pm 1.53</eq></td><td><eq>67.68 \pm 1.70</eq></td><td><eq>47.17 \pm 1.53</eq></td><td><eq>71.08 \pm 1.51</eq></td><td><eq>75.31 \pm 0.90</eq></td><td><eq>88.62 \pm 0.67</eq></td><td>69.08</td></tr><tr><td>FastGPT-M</td><td><eq>85.05 \pm 3.63</eq></td><td><eq>86.41 \pm 10.08</eq></td><td><eq>49.85 \pm 11.80</eq></td><td><eq>89.08 \pm 6.67</eq></td><td><eq>75.84 \pm 17.10</eq></td><td><eq>96.80 \pm 2.60</eq></td><td>80.51</td></tr><tr><td>DNA-GPT</td><td><eq>53.28 \pm 1.23</eq></td><td><eq>43.65 \pm 0.76</eq></td><td><eq>61.90 \pm 0.39</eq></td><td><eq>40.46 \pm 1.26</eq></td><td><eq>67.53 \pm 0.55</eq></td><td><eq>43.31 \pm 0.94</eq></td><td>51.69</td></tr><tr><td>DNA-GPT-M</td><td><eq>83.94 \pm 1.09</eq></td><td><eq>84.02 \pm 1.06</eq></td><td><eq>83.93 \pm 1.20</eq></td><td><eq>83.89 \pm 1.10</eq></td><td><eq>84.23 \pm 1.07</eq></td><td><eq>84.07 \pm 1.06</eq></td><td>84.01</td></tr></table>


Table 6: Performance on Essay dataset. The detection models are trained on text generated by ChatGLM.


<table><tr><td>Metric</td><td>Method</td><td>GPT4All</td><td>ChatGPT</td><td>Dolly</td><td>ChatGLM</td><td>Claude</td><td>ChatGPT-turbo</td><td>avg</td></tr><tr><td rowspan="12">TPR@FPR-1%</td><td>Likelihood</td><td><eq>46.29 \pm 16.54</eq></td><td><eq>68.62 \pm 13.32</eq></td><td><eq>20.67 \pm 10.79</eq></td><td><eq>92.81 \pm 4.82</eq></td><td><eq>12.31 \pm 6.29</eq></td><td><eq>73.60 \pm 14.63</eq></td><td>52.38</td></tr><tr><td>Likelihood-M</td><td><eq>78.82 \pm 5.30</eq></td><td><eq>87.87 \pm 3.22</eq></td><td><eq>46.73 \pm 10.16</eq></td><td><eq>98.71 \pm 1.11</eq></td><td><eq>31.64 \pm 10.61</eq></td><td><eq>93.78 \pm 3.41</eq></td><td>72.92</td></tr><tr><td>Log-Rank</td><td><eq>62.69 \pm 13.36</eq></td><td><eq>79.07 \pm 7.89</eq></td><td><eq>25.11 \pm 8.35</eq></td><td><eq>95.71 \pm 2.05</eq></td><td><eq>19.20 \pm 8.26</eq></td><td><eq>80.89 \pm 10.73</eq></td><td>60.44</td></tr><tr><td>Log-Rank-M</td><td><eq>81.32 \pm 7.05</eq></td><td><eq>86.76 \pm 3.68</eq></td><td><eq>47.35 \pm 8.02</eq></td><td><eq>98.62 \pm 0.26</eq></td><td><eq>25.24 \pm 7.46</eq></td><td><eq>90.62 \pm 5.22</eq></td><td>71.65</td></tr><tr><td>Entropy</td><td><eq>2.73 \pm 0.69</eq></td><td><eq>7.16 \pm 3.17</eq></td><td><eq>2.29 \pm 1.36</eq></td><td><eq>6.88 \pm 3.05</eq></td><td><eq>3.91 \pm 2.24</eq></td><td><eq>13.24 \pm 7.60</eq></td><td>6.04</td></tr><tr><td>Entropy-M</td><td><eq>12.76 \pm 1.21</eq></td><td><eq>32.09 \pm 4.82</eq></td><td><eq>13.99 \pm 3.01</eq></td><td><eq>35.62 \pm 4.65</eq></td><td><eq>18.40 \pm 4.10</eq></td><td><eq>41.42 \pm 6.98</eq></td><td>25.71</td></tr><tr><td>DetectGPT</td><td><eq>0.00 \pm 0.00</eq></td><td><eq>0.00 \pm 0.00</eq></td><td><eq>0.00 \pm 0.00</eq></td><td><eq>0.00 \pm 0.00</eq></td><td><eq>0.89 \pm 0.24</eq></td><td><eq>0.00 \pm 0.00</eq></td><td>0.15</td></tr><tr><td>DetectGPT-M</td><td><eq>18.36 \pm 0.94</eq></td><td><eq>17.87 \pm 5.87</eq></td><td><eq>20.67 \pm 3.81</eq></td><td><eq>60.98 \pm 6.42</eq></td><td><eq>4.89 \pm 2.75</eq></td><td><eq>6.36 \pm 2.01</eq></td><td>21.52</td></tr><tr><td>FastGPT</td><td><eq>1.59 \pm 0.32</eq></td><td><eq>1.64 \pm 0.67</eq></td><td><eq>0.29 \pm 0.10</eq></td><td><eq>2.72 \pm 0.68</eq></td><td><eq>3.47 \pm 1.83</eq></td><td><eq>16.89 \pm 6.66</eq></td><td>4.43</td></tr><tr><td>FastGPT-M</td><td><eq>23.87 \pm 10.38</eq></td><td><eq>29.87 \pm 9.03</eq></td><td><eq>32.89 \pm 16.52</eq></td><td><eq>55.49 \pm 7.87</eq></td><td><eq>0.67 \pm 0.56</eq></td><td><eq>16.22 \pm 31.34</eq></td><td>26.50</td></tr><tr><td>DNA-GPT</td><td><eq>57.54 \pm 4.18</eq></td><td><eq>69.91 \pm 9.92</eq></td><td><eq>39.19 \pm 4.23</eq></td><td><eq>97.32 \pm 0.76</eq></td><td><eq>31.24 \pm 5.75</eq></td><td><eq>65.16 \pm 10.11</eq></td><td>60.06</td></tr><tr><td>DNA-GPT-M</td><td><eq>93.58 \pm 1.29</eq></td><td><eq>92.53 \pm 3.44</eq></td><td><eq>66.54 \pm 9.55</eq></td><td><eq>99.55 \pm 0.20</eq></td><td><eq>44.36 \pm 9.19</eq></td><td><eq>91.47 \pm 4.41</eq></td><td>81.34</td></tr><tr><td rowspan="12">AUROC</td><td>Likelihood</td><td><eq>96.16 \pm 0.30</eq></td><td><eq>98.79 \pm 0.19</eq></td><td><eq>90.90 \pm 1.33</eq></td><td><eq>99.29 \pm 0.25</eq></td><td><eq>92.76 \pm 0.23</eq></td><td><eq>99.13 \pm 0.19</eq></td><td>96.17</td></tr><tr><td>Likelihood-M</td><td><eq>98.03 \pm 0.26</eq></td><td><eq>99.22 \pm 0.20</eq></td><td><eq>93.10 \pm 1.43</eq></td><td><eq>99.65 \pm 0.07</eq></td><td><eq>93.47 \pm 1.14</eq></td><td><eq>99.54 \pm 0.16</eq></td><td>97.17</td></tr><tr><td>Log-Rank</td><td><eq>96.55 \pm 0.31</eq></td><td><eq>98.95 \pm 0.13</eq></td><td><eq>90.08 \pm 1.28</eq></td><td><eq>99.36 \pm 0.13</eq></td><td><eq>92.01 \pm 0.20</eq></td><td><eq>99.24 \pm 0.15</eq></td><td>96.03</td></tr><tr><td>Log-Rank-M</td><td><eq>98.14 \pm 0.30</eq></td><td><eq>99.20 \pm 0.22</eq></td><td><eq>92.86 \pm 1.66</eq></td><td><eq>99.54 \pm 0.10</eq></td><td><eq>91.39 \pm 1.08</eq></td><td><eq>99.53 \pm 0.16</eq></td><td>96.78</td></tr><tr><td>Entropy</td><td><eq>74.19 \pm 1.62</eq></td><td><eq>89.49 \pm 0.33</eq></td><td><eq>73.26 \pm 1.48</eq></td><td><eq>84.11 \pm 0.77</eq></td><td><eq>86.58 \pm 0.66</eq></td><td><eq>95.94 \pm 0.35</eq></td><td>83.93</td></tr><tr><td>Entropy-M</td><td><eq>82.99 \pm 1.01</eq></td><td><eq>93.13 \pm 0.23</eq></td><td><eq>81.03 \pm 1.11</eq></td><td><eq>91.15 \pm 0.28</eq></td><td><eq>88.45 \pm 0.44</eq></td><td><eq>96.88 \pm 0.18</eq></td><td>88.94</td></tr><tr><td>DetectGPT</td><td><eq>50.81 \pm 0.58</eq></td><td><eq>46.40 \pm 0.77</eq></td><td><eq>57.48 \pm 0.84</eq></td><td><eq>50.41 \pm 1.70</eq></td><td><eq>41.54 \pm 0.60</eq></td><td><eq>17.90 \pm 1.25</eq></td><td>44.09</td></tr><tr><td>DetectGPT-M</td><td><eq>80.39 \pm 1.33</eq></td><td><eq>80.53 \pm 0.58</eq></td><td><eq>80.67 \pm 1.98</eq></td><td><eq>94.02 \pm 0.56</eq></td><td><eq>62.63 \pm 1.24</eq></td><td><eq>67.25 \pm 1.24</eq></td><td>77.58</td></tr><tr><td>FastGPT</td><td><eq>64.63 \pm 1.53</eq></td><td><eq>67.68 \pm 1.70</eq></td><td><eq>47.17 \pm 1.53</eq></td><td><eq>71.08 \pm 1.51</eq></td><td><eq>75.31 \pm 0.90</eq></td><td><eq>88.62 \pm 0.67</eq></td><td>69.08</td></tr><tr><td>FastGPT-M</td><td><eq>87.19 \pm 3.46</eq></td><td><eq>91.55 \pm 3.32</eq></td><td><eq>82.58 \pm 18.04</eq></td><td><eq>95.35 \pm 0.49</eq></td><td><eq>59.28 \pm 1.90</eq></td><td><eq>63.48 \pm 18.12</eq></td><td>79.91</td></tr><tr><td>DNA-GPT</td><td><eq>96.97 \pm 0.20</eq></td><td><eq>98.09 \pm 0.47</eq></td><td><eq>94.11 \pm 0.47</eq></td><td><eq>99.87 \pm 0.04</eq></td><td><eq>92.84 \pm 0.71</eq></td><td><eq>97.85 \pm 0.37</eq></td><td>96.62</td></tr><tr><td>DNA-GPT-M</td><td><eq>99.37 \pm 0.05</eq></td><td><eq>99.58 \pm 0.10</eq></td><td><eq>97.68 \pm 0.29</eq></td><td><eq>99.92 \pm 0.02</eq></td><td><eq>96.24 \pm 0.47</eq></td><td><eq>99.54 \pm 0.09</eq></td><td>98.72</td></tr></table>


Table 7: Performance on Essay dataset. The detection models are trained on text generated by Dolly.


<table><tr><td>Metric</td><td>Method</td><td>GPT4All</td><td>ChatGPT</td><td>Dolly</td><td>ChatGLM</td><td>Claude</td><td>ChatGPT-turbo</td><td>avg</td></tr><tr><td rowspan="12">TPR@FPR-1%</td><td>Likelihood</td><td><eq>46.29 \pm 16.54</eq></td><td><eq>68.62 \pm 13.32</eq></td><td><eq>20.67 \pm 10.79</eq></td><td><eq>92.86 \pm 4.84</eq></td><td><eq>12.31 \pm 6.29</eq></td><td><eq>73.60 \pm 14.63</eq></td><td>52.39</td></tr><tr><td>Likelihood-M</td><td><eq>89.02 \pm 1.68</eq></td><td><eq>88.09 \pm 4.50</eq></td><td><eq>57.14 \pm 5.55</eq></td><td><eq>97.59 \pm 1.09</eq></td><td><eq>34.36 \pm 13.54</eq></td><td><eq>94.36 \pm 3.67</eq></td><td>76.76</td></tr><tr><td>Log-Rank</td><td><eq>62.69 \pm 13.36</eq></td><td><eq>79.07 \pm 7.89</eq></td><td><eq>25.11 \pm 8.35</eq></td><td><eq>95.71 \pm 2.05</eq></td><td><eq>19.16 \pm 8.22</eq></td><td><eq>80.89 \pm 10.73</eq></td><td>60.44</td></tr><tr><td>Log-Rank-M</td><td><eq>87.70 \pm 2.60</eq></td><td><eq>89.02 \pm 2.13</eq></td><td><eq>51.65 \pm 3.99</eq></td><td><eq>98.12 \pm 0.54</eq></td><td><eq>31.69 \pm 3.32</eq></td><td><eq>94.36 \pm 1.93</eq></td><td>75.42</td></tr><tr><td>Entropy</td><td><eq>2.73 \pm 0.69</eq></td><td><eq>7.16 \pm 3.17</eq></td><td><eq>2.29 \pm 1.36</eq></td><td><eq>6.88 \pm 3.05</eq></td><td><eq>3.91 \pm 2.24</eq></td><td><eq>13.24 \pm 7.60</eq></td><td>6.04</td></tr><tr><td>Entropy-M</td><td><eq>11.94 \pm 1.85</eq></td><td><eq>32.09 \pm 4.77</eq></td><td><eq>13.22 \pm 4.31</eq></td><td><eq>34.78 \pm 8.04</eq></td><td><eq>18.49 \pm 3.20</eq></td><td><eq>40.62 \pm 6.34</eq></td><td>25.19</td></tr><tr><td>DetectGPT</td><td><eq>0.00 \pm 0.00</eq></td><td><eq>0.00 \pm 0.00</eq></td><td><eq>0.00 \pm 0.00</eq></td><td><eq>0.00 \pm 0.00</eq></td><td><eq>0.89 \pm 0.24</eq></td><td><eq>0.00 \pm 0.00</eq></td><td>0.15</td></tr><tr><td>DetectGPT-M</td><td><eq>32.89 \pm 22.87</eq></td><td><eq>46.09 \pm 19.75</eq></td><td><eq>24.44 \pm 9.98</eq></td><td><eq>68.84 \pm 21.48</eq></td><td><eq>6.04 \pm 4.69</eq></td><td><eq>48.09 \pm 25.20</eq></td><td>37.73</td></tr><tr><td>FastGPT</td><td><eq>0.82 \pm 0.31</eq></td><td><eq>0.00 \pm 0.00</eq></td><td><eq>0.38 \pm 0.42</eq></td><td><eq>0.09 \pm 0.11</eq></td><td><eq>0.00 \pm 0.00</eq></td><td><eq>0.00 \pm 0.00</eq></td><td>0.22</td></tr><tr><td>FastGPT-M</td><td><eq>27.24 \pm 2.77</eq></td><td><eq>28.04 \pm 7.40</eq></td><td><eq>39.43 \pm 3.59</eq></td><td><eq>55.89 \pm 5.30</eq></td><td><eq>0.98 \pm 0.65</eq></td><td><eq>0.84 \pm 0.71</eq></td><td>25.41</td></tr><tr><td>DNA-GPT</td><td><eq>58.95 \pm 4.76</eq></td><td><eq>64.00 \pm 6.45</eq></td><td><eq>48.64 \pm 4.04</eq></td><td><eq>84.29 \pm 3.64</eq></td><td><eq>29.24 \pm 3.40</eq></td><td><eq>59.02 \pm 8.29</eq></td><td>57.36</td></tr><tr><td>DNA-GPT-M</td><td><eq>88.70 \pm 1.78</eq></td><td><eq>87.82 \pm 1.27</eq></td><td><eq>72.60 \pm 4.17</eq></td><td><eq>92.59 \pm 0.38</eq></td><td><eq>44.71 \pm 4.67</eq></td><td><eq>86.93 \pm 2.13</eq></td><td>78.89</td></tr><tr><td rowspan="12">AUROC</td><td>Likelihood</td><td><eq>96.16 \pm 0.30</eq></td><td><eq>98.79 \pm 0.19</eq></td><td><eq>90.90 \pm 1.33</eq></td><td><eq>99.29 \pm 0.25</eq></td><td><eq>92.76 \pm 0.24</eq></td><td><eq>99.13 \pm 0.19</eq></td><td>96.17</td></tr><tr><td>Likelihood-M</td><td><eq>98.63 \pm 0.29</eq></td><td><eq>99.48 \pm 0.16</eq></td><td><eq>94.44 \pm 0.71</eq></td><td><eq>99.62 \pm 0.08</eq></td><td><eq>94.49 \pm 0.73</eq></td><td><eq>99.70 \pm 0.15</eq></td><td>97.73</td></tr><tr><td>Log-Rank</td><td><eq>96.55 \pm 0.31</eq></td><td><eq>98.95 \pm 0.13</eq></td><td><eq>90.08 \pm 1.28</eq></td><td><eq>99.36 \pm 0.13</eq></td><td><eq>92.01 \pm 0.20</eq></td><td><eq>99.24 \pm 0.15</eq></td><td>96.03</td></tr><tr><td>Log-Rank-M</td><td><eq>98.55 \pm 0.08</eq></td><td><eq>99.40 \pm 0.09</eq></td><td><eq>93.78 \pm 0.93</eq></td><td><eq>99.56 \pm 0.09</eq></td><td><eq>92.81 \pm 0.38</eq></td><td><eq>99.65 \pm 0.10</eq></td><td>97.29</td></tr><tr><td>Entropy</td><td><eq>74.19 \pm 1.62</eq></td><td><eq>89.49 \pm 0.33</eq></td><td><eq>73.26 \pm 1.48</eq></td><td><eq>84.11 \pm 0.77</eq></td><td><eq>86.58 \pm 0.66</eq></td><td><eq>95.94 \pm 0.35</eq></td><td>83.93</td></tr><tr><td>Entropy-M</td><td><eq>83.09 \pm 0.84</eq></td><td><eq>93.15 \pm 0.17</eq></td><td><eq>81.05 \pm 1.14</eq></td><td><eq>91.26 \pm 0.24</eq></td><td><eq>88.42 \pm 0.52</eq></td><td><eq>96.88 \pm 0.20</eq></td><td>88.97</td></tr><tr><td>DetectGPT</td><td><eq>50.81 \pm 0.58</eq></td><td><eq>46.40 \pm 0.77</eq></td><td><eq>57.48 \pm 0.84</eq></td><td><eq>50.41 \pm 1.70</eq></td><td><eq>41.54 \pm 0.60</eq></td><td><eq>17.90 \pm 1.25</eq></td><td>44.09</td></tr><tr><td>DetectGPT-M</td><td><eq>93.62 \pm 5.70</eq></td><td><eq>93.95 \pm 6.36</eq></td><td><eq>90.01 \pm 3.69</eq></td><td><eq>97.82 \pm 1.98</eq></td><td><eq>81.42 \pm 9.59</eq></td><td><eq>92.20 \pm 11.34</eq></td><td>91.50</td></tr><tr><td>FastGPT</td><td><eq>35.37 \pm 1.53</eq></td><td><eq>32.32 \pm 1.70</eq></td><td><eq>52.83 \pm 1.53</eq></td><td><eq>28.92 \pm 1.51</eq></td><td><eq>24.69 \pm 0.90</eq></td><td><eq>11.38 \pm 0.67</eq></td><td>30.92</td></tr><tr><td>FastGPT-M</td><td><eq>88.35 \pm 0.89</eq></td><td><eq>89.67 \pm 1.11</eq></td><td><eq>91.55 \pm 0.28</eq></td><td><eq>95.27 \pm 0.38</eq></td><td><eq>59.65 \pm 1.54</eq></td><td><eq>54.61 \pm 2.30</eq></td><td>79.85</td></tr><tr><td>DNA-GPT</td><td><eq>94.50 \pm 0.07</eq></td><td><eq>93.95 \pm 0.19</eq></td><td><eq>95.91 \pm 0.49</eq></td><td><eq>95.04 \pm 0.15</eq></td><td><eq>89.35 \pm 0.39</eq></td><td><eq>93.81 \pm 0.20</eq></td><td>93.76</td></tr><tr><td>DNA-GPT-M</td><td><eq>98.23 \pm 0.21</eq></td><td><eq>97.67 \pm 0.16</eq></td><td><eq>98.01 \pm 0.33</eq></td><td><eq>98.00 \pm 0.14</eq></td><td><eq>94.63 \pm 0.40</eq></td><td><eq>97.64 \pm 0.17</eq></td><td>97.37</td></tr></table>


Table 8: Performance on Essay dataset. The detection models are trained on text generated by Claude.


<table><tr><td>Metric</td><td>Method</td><td>GPT4All</td><td>ChatGPT</td><td>Dolly</td><td>ChatGLM</td><td>Claude</td><td>ChatGPT-turbo</td><td>avg</td></tr><tr><td rowspan="12">TPR@FPR-1%</td><td>Likelihood</td><td><eq>46.20 \pm 16.47</eq></td><td><eq>68.58 \pm 13.28</eq></td><td><eq>20.67 \pm 10.79</eq></td><td><eq>92.86 \pm 4.84</eq></td><td><eq>12.31 \pm 6.29</eq></td><td><eq>73.60 \pm 14.63</eq></td><td>52.37</td></tr><tr><td>Likelihood-M</td><td><eq>86.74 \pm 3.37</eq></td><td><eq>90.09 \pm 2.80</eq></td><td><eq>50.07 \pm 9.80</eq></td><td><eq>98.08 \pm 0.27</eq></td><td><eq>34.00 \pm 6.66</eq></td><td><eq>95.78 \pm 1.40</eq></td><td>75.79</td></tr><tr><td>Log-Rank</td><td><eq>62.69 \pm 13.36</eq></td><td><eq>79.07 \pm 7.89</eq></td><td><eq>25.11 \pm 8.35</eq></td><td><eq>95.71 \pm 2.05</eq></td><td><eq>19.20 \pm 8.26</eq></td><td><eq>80.89 \pm 10.73</eq></td><td>60.44</td></tr><tr><td>Log-Rank-M</td><td><eq>89.20 \pm 2.33</eq></td><td><eq>88.18 \pm 2.58</eq></td><td><eq>54.42 \pm 3.96</eq></td><td><eq>95.00 \pm 3.56</eq></td><td><eq>37.82 \pm 2.28</eq></td><td><eq>94.80 \pm 1.37</eq></td><td>76.57</td></tr><tr><td>Entropy</td><td><eq>2.73 \pm 0.69</eq></td><td><eq>7.16 \pm 3.17</eq></td><td><eq>2.29 \pm 1.36</eq></td><td><eq>6.88 \pm 3.05</eq></td><td><eq>3.91 \pm 2.24</eq></td><td><eq>13.24 \pm 7.60</eq></td><td>6.04</td></tr><tr><td>Entropy-M</td><td><eq>12.16 \pm 0.86</eq></td><td><eq>29.07 \pm 3.22</eq></td><td><eq>10.74 \pm 4.12</eq></td><td><eq>30.94 \pm 5.62</eq></td><td><eq>16.80 \pm 2.73</eq></td><td><eq>39.69 \pm 4.99</eq></td><td>23.23</td></tr><tr><td>DetectGPT</td><td><eq>0.00 \pm 0.00</eq></td><td><eq>0.00 \pm 0.00</eq></td><td><eq>0.00 \pm 0.00</eq></td><td><eq>0.00 \pm 0.00</eq></td><td><eq>0.84 \pm 0.51</eq></td><td><eq>2.36 \pm 2.52</eq></td><td>0.53</td></tr><tr><td>DetectGPT-M</td><td><eq>34.62 \pm 17.61</eq></td><td><eq>38.71 \pm 23.27</eq></td><td><eq>24.73 \pm 8.38</eq></td><td><eq>74.06 \pm 17.99</eq></td><td><eq>3.91 \pm 2.05</eq></td><td><eq>29.11 \pm 24.44</eq></td><td>34.19</td></tr><tr><td>FastGPT</td><td><eq>1.59 \pm 0.32</eq></td><td><eq>1.64 \pm 0.67</eq></td><td><eq>0.29 \pm 0.10</eq></td><td><eq>2.72 \pm 0.68</eq></td><td><eq>3.47 \pm 1.83</eq></td><td><eq>16.89 \pm 6.66</eq></td><td>4.43</td></tr><tr><td>FastGPT-M</td><td><eq>27.47 \pm 2.92</eq></td><td><eq>28.04 \pm 7.49</eq></td><td><eq>39.57 \pm 3.76</eq></td><td><eq>56.47 \pm 6.13</eq></td><td><eq>0.98 \pm 0.65</eq></td><td><eq>0.84 \pm 0.71</eq></td><td>25.56</td></tr><tr><td>DNA-GPT</td><td><eq>0.50 \pm 0.17</eq></td><td><eq>0.31 \pm 0.11</eq></td><td><eq>0.67 \pm 0.55</eq></td><td><eq>0.00 \pm 0.00</eq></td><td><eq>0.13 \pm 0.11</eq></td><td><eq>0.53 \pm 0.36</eq></td><td>0.36</td></tr><tr><td>DNA-GPT-M</td><td><eq>12.76 \pm 2.00</eq></td><td><eq>13.16 \pm 2.36</eq></td><td><eq>14.03 \pm 3.12</eq></td><td><eq>12.14 \pm 1.73</eq></td><td><eq>13.69 \pm 2.59</eq></td><td><eq>13.33 \pm 2.38</eq></td><td>13.19</td></tr><tr><td rowspan="12">AUROC</td><td>Likelihood</td><td><eq>96.16 \pm 0.30</eq></td><td><eq>98.79 \pm 0.19</eq></td><td><eq>90.90 \pm 1.33</eq></td><td><eq>99.29 \pm 0.25</eq></td><td><eq>92.76 \pm 0.23</eq></td><td><eq>99.13 \pm 0.19</eq></td><td>96.17</td></tr><tr><td>Likelihood-M</td><td><eq>98.52 \pm 0.24</eq></td><td><eq>99.45 \pm 0.09</eq></td><td><eq>94.32 \pm 0.95</eq></td><td><eq>99.63 \pm 0.09</eq></td><td><eq>94.35 \pm 0.42</eq></td><td><eq>99.66 \pm 0.16</eq></td><td>97.66</td></tr><tr><td>Log-Rank</td><td><eq>96.55 \pm 0.31</eq></td><td><eq>98.95 \pm 0.13</eq></td><td><eq>90.08 \pm 1.28</eq></td><td><eq>99.36 \pm 0.13</eq></td><td><eq>92.01 \pm 0.20</eq></td><td><eq>99.24 \pm 0.15</eq></td><td>96.03</td></tr><tr><td>Log-Rank-M</td><td><eq>98.47 \pm 0.29</eq></td><td><eq>99.28 \pm 0.22</eq></td><td><eq>93.15 \pm 1.37</eq></td><td><eq>99.25 \pm 0.32</eq></td><td><eq>92.93 \pm 0.43</eq></td><td><eq>99.66 \pm 0.10</eq></td><td>97.13</td></tr><tr><td>Entropy</td><td><eq>74.19 \pm 1.62</eq></td><td><eq>89.49 \pm 0.33</eq></td><td><eq>73.26 \pm 1.48</eq></td><td><eq>84.11 \pm 0.77</eq></td><td><eq>86.58 \pm 0.66</eq></td><td><eq>95.94 \pm 0.35</eq></td><td>83.93</td></tr><tr><td>Entropy-M</td><td><eq>82.36 \pm 1.15</eq></td><td><eq>92.90 \pm 0.26</eq></td><td><eq>80.64 \pm 1.12</eq></td><td><eq>90.88 \pm 0.52</eq></td><td><eq>88.62 \pm 0.48</eq></td><td><eq>96.91 \pm 0.23</eq></td><td>88.72</td></tr><tr><td>DetectGPT</td><td><eq>50.15 \pm 0.99</eq></td><td><eq>50.53 \pm 3.65</eq></td><td><eq>48.04 \pm 7.27</eq></td><td><eq>49.25 \pm 1.58</eq></td><td><eq>51.21 \pm 8.39</eq></td><td><eq>55.34 \pm 31.67</eq></td><td>50.75</td></tr><tr><td>DetectGPT-M</td><td><eq>86.96 \pm 7.59</eq></td><td><eq>87.54 \pm 7.67</eq></td><td><eq>85.22 \pm 5.50</eq></td><td><eq>96.46 \pm 2.24</eq></td><td><eq>71.40 \pm 9.73</eq></td><td><eq>80.33 \pm 14.18</eq></td><td>84.65</td></tr><tr><td>FastGPT</td><td><eq>64.63 \pm 1.53</eq></td><td><eq>67.68 \pm 1.70</eq></td><td><eq>47.17 \pm 1.53</eq></td><td><eq>71.08 \pm 1.51</eq></td><td><eq>75.31 \pm 0.90</eq></td><td><eq>88.62 \pm 0.67</eq></td><td>69.08</td></tr><tr><td>FastGPT-M</td><td><eq>88.45 \pm 0.98</eq></td><td><eq>89.73 \pm 1.25</eq></td><td><eq>91.61 \pm 0.36</eq></td><td><eq>95.35 \pm 0.47</eq></td><td><eq>59.77 \pm 1.62</eq></td><td><eq>54.76 \pm 2.39</eq></td><td>79.95</td></tr><tr><td>DNA-GPT</td><td><eq>45.85 \pm 2.15</eq></td><td><eq>39.61 \pm 1.52</eq></td><td><eq>53.93 \pm 1.47</eq></td><td><eq>36.19 \pm 1.75</eq></td><td><eq>53.01 \pm 1.67</eq></td><td><eq>40.65 \pm 1.60</eq></td><td>44.87</td></tr><tr><td>DNA-GPT-M</td><td><eq>76.26 \pm 1.36</eq></td><td><eq>76.25 \pm 1.29</eq></td><td><eq>76.38 \pm 1.33</eq></td><td><eq>75.98 \pm 1.39</eq></td><td><eq>76.74 \pm 1.26</eq></td><td><eq>76.30 \pm 1.29</eq></td><td>76.32</td></tr></table>


Table 9: Performance on DetectRL. The detection models are trained on text generated by ChatGPT and Google-PaLM.


<table><tr><td rowspan="2">Metric</td><td rowspan="2">Method</td><td colspan="4">DetectRL (Training Text: ChatGPT)</td><td colspan="4">DetectRL (Training Text: Google-PaLM)</td></tr><tr><td>Llama-2-70b</td><td>ChatGPT</td><td>Google-PaLM</td><td>Avg.</td><td>Llama-2-70b</td><td>ChatGPT</td><td>Google-PaLM</td><td>Avg.</td></tr><tr><td rowspan="12">TPR@FPR-1%</td><td>Likelihood</td><td><eq>38.37 \pm 0.92</eq></td><td><eq>10.21 \pm 1.40</eq></td><td><eq>25.66 \pm 2.41</eq></td><td>24.75</td><td><eq>38.32 \pm 0.88</eq></td><td><eq>10.19 \pm 1.38</eq></td><td><eq>25.66 \pm 2.41</eq></td><td>24.72</td></tr><tr><td>Likelihood-M</td><td><eq>45.61 \pm 3.57</eq></td><td><eq>12.14 \pm 2.66</eq></td><td><eq>33.18 \pm 2.46</eq></td><td>30.31</td><td><eq>42.94 \pm 4.14</eq></td><td><eq>9.91 \pm 3.87</eq></td><td><eq>35.48 \pm 2.70</eq></td><td>29.44</td></tr><tr><td>Log-Rank</td><td><eq>42.05 \pm 0.84</eq></td><td><eq>12.44 \pm 1.71</eq></td><td><eq>22.84 \pm 1.28</eq></td><td>25.78</td><td><eq>42.05 \pm 0.84</eq></td><td><eq>12.41 \pm 1.72</eq></td><td><eq>22.84 \pm 1.28</eq></td><td>25.77</td></tr><tr><td>Log-Rank-M</td><td><eq>50.66 \pm 1.43</eq></td><td><eq>12.73 \pm 1.65</eq></td><td><eq>41.41 \pm 2.56</eq></td><td>34.93</td><td><eq>50.58 \pm 1.35</eq></td><td><eq>13.00 \pm 2.43</eq></td><td><eq>41.09 \pm 4.07</eq></td><td>34.89</td></tr><tr><td>Entropy</td><td><eq>2.03 \pm 0.73</eq></td><td><eq>0.25 \pm 0.16</eq></td><td><eq>6.95 \pm 0.78</eq></td><td>3.07</td><td><eq>2.03 \pm 0.73</eq></td><td><eq>0.25 \pm 0.16</eq></td><td><eq>6.95 \pm 0.78</eq></td><td>3.07</td></tr><tr><td>Entropy-M</td><td><eq>2.13 \pm 0.30</eq></td><td><eq>0.27 \pm 0.20</eq></td><td><eq>9.27 \pm 0.87</eq></td><td>3.89</td><td><eq>2.22 \pm 0.37</eq></td><td><eq>0.27 \pm 0.20</eq></td><td><eq>9.37 \pm 0.93</eq></td><td>3.96</td></tr><tr><td>DetectGPT</td><td><eq>3.29 \pm 0.97</eq></td><td><eq>5.36 \pm 2.20</eq></td><td><eq>2.79 \pm 1.32</eq></td><td>3.82</td><td><eq>2.87 \pm 0.56</eq></td><td><eq>6.77 \pm 1.50</eq></td><td><eq>3.86 \pm 1.44</eq></td><td>4.50</td></tr><tr><td>DetectGPT-M</td><td><eq>25.29 \pm 10.05</eq></td><td><eq>7.66 \pm 4.17</eq></td><td><eq>14.17 \pm 11.10</eq></td><td>15.71</td><td><eq>22.00 \pm 4.30</eq></td><td><eq>4.57 \pm 0.37</eq></td><td><eq>24.94 \pm 2.68</eq></td><td>17.17</td></tr><tr><td>FastGPT</td><td><eq>11.35 \pm 2.01</eq></td><td><eq>3.78 \pm 1.20</eq></td><td><eq>5.02 \pm 0.85</eq></td><td>6.72</td><td><eq>11.35 \pm 2.01</eq></td><td><eq>3.78 \pm 1.20</eq></td><td><eq>5.02 \pm 0.85</eq></td><td>6.72</td></tr><tr><td>FastGPT-M</td><td><eq>27.71 \pm 15.63</eq></td><td><eq>13.42 \pm 8.91</eq></td><td><eq>17.68 \pm 11.92</eq></td><td>19.60</td><td><eq>12.41 \pm 0.89</eq></td><td><eq>6.18 \pm 0.83</eq></td><td><eq>20.10 \pm 1.78</eq></td><td>12.90</td></tr><tr><td>DNA-GPT</td><td><eq>31.03 \pm 1.59</eq></td><td><eq>11.05 \pm 1.69</eq></td><td><eq>26.40 \pm 2.08</eq></td><td>22.83</td><td><eq>41.71 \pm 1.32</eq></td><td><eq>29.49 \pm 0.84</eq></td><td><eq>38.84 \pm 1.58</eq></td><td>36.68</td></tr><tr><td>DNA-GPT-M</td><td><eq>41.16 \pm 1.63</eq></td><td><eq>12.53 \pm 2.06</eq></td><td><eq>33.57 \pm 2.15</eq></td><td>29.09</td><td><eq>47.10 \pm 0.93</eq></td><td><eq>22.72 \pm 1.50</eq></td><td><eq>40.54 \pm 1.98</eq></td><td>36.79</td></tr><tr><td rowspan="12">AUROC</td><td>Likelihood</td><td><eq>78.58 \pm 0.41</eq></td><td><eq>66.61 \pm 0.99</eq></td><td><eq>71.42 \pm 0.49</eq></td><td>72.2</td><td><eq>78.57 \pm 0.41</eq></td><td><eq>66.61 \pm 0.99</eq></td><td><eq>71.42 \pm 0.49</eq></td><td>72.20</td></tr><tr><td>Likelihood-M</td><td><eq>78.15 \pm 4.58</eq></td><td><eq>67.06 \pm 4.19</eq></td><td><eq>73.40 \pm 4.38</eq></td><td>72.87</td><td><eq>87.44 \pm 0.70</eq></td><td><eq>73.70 \pm 0.57</eq></td><td><eq>81.58 \pm 0.93</eq></td><td>80.91</td></tr><tr><td>Log-Rank</td><td><eq>79.67 \pm 0.46</eq></td><td><eq>65.85 \pm 0.94</eq></td><td><eq>70.66 \pm 0.40</eq></td><td>72.06</td><td><eq>79.67 \pm 0.46</eq></td><td><eq>65.85 \pm 0.94</eq></td><td><eq>70.66 \pm 0.40</eq></td><td>72.06</td></tr><tr><td>Log-Rank-M</td><td><eq>90.23 \pm 0.45</eq></td><td><eq>75.40 \pm 0.97</eq></td><td><eq>84.44 \pm 0.52</eq></td><td>83.35</td><td><eq>90.14 \pm 0.53</eq></td><td><eq>75.61 \pm 0.83</eq></td><td><eq>84.63 \pm 0.59</eq></td><td>83.46</td></tr><tr><td>Entropy</td><td><eq>66.21 \pm 0.88</eq></td><td><eq>63.09 \pm 1.05</eq></td><td><eq>60.73 \pm 0.91</eq></td><td>63.34</td><td><eq>66.21 \pm 0.88</eq></td><td><eq>63.09 \pm 1.05</eq></td><td><eq>60.73 \pm 0.91</eq></td><td>63.34</td></tr><tr><td>Entropy-M</td><td><eq>68.61 \pm 0.94</eq></td><td><eq>66.83 \pm 0.95</eq></td><td><eq>67.93 \pm 1.08</eq></td><td>67.79</td><td><eq>68.91 \pm 0.83</eq></td><td><eq>66.86 \pm 0.95</eq></td><td><eq>67.94 \pm 1.04</eq></td><td>67.90</td></tr><tr><td>DetectGPT</td><td><eq>49.21 \pm 2.31</eq></td><td><eq>49.64 \pm 0.53</eq></td><td><eq>50.72 \pm 6.91</eq></td><td>49.85</td><td><eq>47.63 \pm 0.59</eq></td><td><eq>49.78 \pm 0.60</eq></td><td><eq>56.81 \pm 1.41</eq></td><td>51.40</td></tr><tr><td>DetectGPT-M</td><td><eq>73.50 \pm 9.13</eq></td><td><eq>62.93 \pm 4.59</eq></td><td><eq>61.30 \pm 15.58</eq></td><td>65.91</td><td><eq>77.77 \pm 4.05</eq></td><td><eq>60.34 \pm 3.37</eq></td><td><eq>76.40 \pm 1.66</eq></td><td>71.50</td></tr><tr><td>FastGPT</td><td><eq>67.72 \pm 1.02</eq></td><td><eq>58.50 \pm 1.22</eq></td><td><eq>56.68 \pm 1.21</eq></td><td>60.97</td><td><eq>67.72 \pm 1.02</eq></td><td><eq>58.50 \pm 1.22</eq></td><td><eq>56.68 \pm 1.21</eq></td><td>60.97</td></tr><tr><td>FastGPT-M</td><td><eq>71.60 \pm 6.86</eq></td><td><eq>61.26 \pm 5.62</eq></td><td><eq>66.05 \pm 9.70</eq></td><td>66.30</td><td><eq>62.35 \pm 0.49</eq></td><td><eq>51.87 \pm 1.38</eq></td><td><eq>68.88 \pm 1.00</eq></td><td>61.03</td></tr><tr><td>DNA-GPT</td><td><eq>72.53 \pm 1.19</eq></td><td><eq>66.07 \pm 1.38</eq></td><td><eq>68.35 \pm 0.98</eq></td><td>68.98</td><td><eq>79.00 \pm 0.86</eq></td><td><eq>72.27 \pm 0.81</eq></td><td><eq>77.96 \pm 0.94</eq></td><td>76.41</td></tr><tr><td>DNA-GPT-M</td><td><eq>75.37 \pm 1.13</eq></td><td><eq>66.19 \pm 1.40</eq></td><td><eq>71.77 \pm 0.79</eq></td><td>71.11</td><td><eq>80.09 \pm 1.26</eq></td><td><eq>71.09 \pm 1.12</eq></td><td><eq>77.15 \pm 1.02</eq></td><td>76.11</td></tr></table>


Table 10: Performance on Reuters dataset. The detection models are trained on text generated by GPT4All.


<table><tr><td>Metric</td><td>Method</td><td>GPT4All</td><td>ChatGPT</td><td>Dolly</td><td>ChatGLM</td><td>Claude</td><td>ChatGPT-turbo</td><td>Avg.</td></tr><tr><td rowspan="12">TPR@FPR-1%</td><td>Likelihood</td><td><eq>19.47 \pm 2.45</eq></td><td><eq>80.89 \pm 2.17</eq></td><td><eq>12.44 \pm 1.38</eq></td><td><eq>94.36 \pm 1.15</eq></td><td><eq>18.89 \pm 3.41</eq></td><td><eq>90.22 \pm 1.23</eq></td><td>52.71</td></tr><tr><td>Likelihood-M</td><td><eq>61.42 \pm 5.74</eq></td><td><eq>88.44 \pm 2.19</eq></td><td><eq>35.56 \pm 2.86</eq></td><td><eq>95.24 \pm 1.16</eq></td><td><eq>32.36 \pm 5.93</eq></td><td><eq>92.76 \pm 0.71</eq></td><td>67.63</td></tr><tr><td>Log-Rank</td><td><eq>29.91 \pm 3.27</eq></td><td><eq>86.22 \pm 1.87</eq></td><td><eq>15.20 \pm 1.70</eq></td><td><eq>97.33 \pm 0.64</eq></td><td><eq>24.58 \pm 3.35</eq></td><td><eq>93.16 \pm 1.07</eq></td><td>57.73</td></tr><tr><td>Log-Rank-M</td><td><eq>73.82 \pm 3.32</eq></td><td><eq>89.56 \pm 1.79</eq></td><td><eq>38.31 \pm 2.26</eq></td><td><eq>96.71 \pm 1.62</eq></td><td><eq>35.47 \pm 4.63</eq></td><td><eq>94.49 \pm 0.62</eq></td><td>71.39</td></tr><tr><td>Entropy</td><td><eq>6.18 \pm 1.40</eq></td><td><eq>0.04 \pm 0.09</eq></td><td><eq>13.07 \pm 1.76</eq></td><td><eq>0.22 \pm 0.20</eq></td><td><eq>0.00 \pm 0.00</eq></td><td><eq>0.22 \pm 0.00</eq></td><td>3.29</td></tr><tr><td>Entropy-M</td><td><eq>17.42 \pm 2.84</eq></td><td><eq>0.36 \pm 0.30</eq></td><td><eq>12.36 \pm 2.86</eq></td><td><eq>0.98 \pm 0.87</eq></td><td><eq>0.00 \pm 0.00</eq></td><td><eq>0.40 \pm 0.38</eq></td><td>5.25</td></tr><tr><td>DetectGPT</td><td><eq>0.00 \pm 0.00</eq></td><td><eq>0.00 \pm 0.00</eq></td><td><eq>0.00 \pm 0.00</eq></td><td><eq>0.00 \pm 0.00</eq></td><td><eq>0.44 \pm 0.40</eq></td><td><eq>0.04 \pm 0.09</eq></td><td>0.08</td></tr><tr><td>DetectGPT-M</td><td><eq>40.89 \pm 6.07</eq></td><td><eq>15.33 \pm 8.81</eq></td><td><eq>46.40 \pm 2.46</eq></td><td><eq>40.00 \pm 8.10</eq></td><td><eq>2.40 \pm 0.38</eq></td><td><eq>6.40 \pm 2.50</eq></td><td>25.24</td></tr><tr><td>FastGPT</td><td><eq>3.56 \pm 3.08</eq></td><td><eq>0.04 \pm 0.09</eq></td><td><eq>12.18 \pm 6.70</eq></td><td><eq>0.09 \pm 0.11</eq></td><td><eq>0.09 \pm 0.11</eq></td><td><eq>0.04 \pm 0.09</eq></td><td>2.67</td></tr><tr><td>FastGPT-M</td><td><eq>61.69 \pm 7.18</eq></td><td><eq>21.78 \pm 6.77</eq></td><td><eq>65.24 \pm 5.12</eq></td><td><eq>31.78 \pm 9.97</eq></td><td><eq>1.38 \pm 0.84</eq></td><td><eq>1.51 \pm 0.55</eq></td><td>30.56</td></tr><tr><td>DNA-GPT</td><td><eq>85.20 \pm 6.79</eq></td><td><eq>92.58 \pm 3.94</eq></td><td><eq>68.49 \pm 4.37</eq></td><td><eq>96.93 \pm 3.14</eq></td><td><eq>79.24 \pm 4.93</eq></td><td><eq>95.29 \pm 3.53</eq></td><td>86.29</td></tr><tr><td>DNA-GPT-M</td><td><eq>97.51 \pm 1.31</eq></td><td><eq>98.40 \pm 0.88</eq></td><td><eq>81.29 \pm 3.75</eq></td><td><eq>99.87 \pm 0.11</eq></td><td><eq>89.07 \pm 3.57</eq></td><td><eq>99.24 \pm 0.30</eq></td><td>94.23</td></tr><tr><td rowspan="12">AUROC</td><td>Likelihood</td><td><eq>75.19 \pm 1.39</eq></td><td><eq>97.55 \pm 0.33</eq></td><td><eq>58.77 \pm 1.76</eq></td><td><eq>99.62 \pm 0.11</eq></td><td><eq>86.45 \pm 0.70</eq></td><td><eq>98.42 \pm 0.25</eq></td><td>86.00</td></tr><tr><td>Likelihood-M</td><td><eq>93.74 \pm 1.23</eq></td><td><eq>98.42 \pm 0.26</eq></td><td><eq>77.05 \pm 0.53</eq></td><td><eq>99.61 \pm 0.11</eq></td><td><eq>90.68 \pm 0.56</eq></td><td><eq>98.78 \pm 0.33</eq></td><td>93.05</td></tr><tr><td>Log-Rank</td><td><eq>78.39 \pm 1.23</eq></td><td><eq>97.77 \pm 0.32</eq></td><td><eq>58.22 \pm 2.03</eq></td><td><eq>99.61 \pm 0.19</eq></td><td><eq>85.68 \pm 0.67</eq></td><td><eq>98.67 \pm 0.24</eq></td><td>86.39</td></tr><tr><td>Log-Rank-M</td><td><eq>95.50 \pm 0.81</eq></td><td><eq>98.60 \pm 0.23</eq></td><td><eq>76.60 \pm 1.33</eq></td><td><eq>99.64 \pm 0.29</eq></td><td><eq>90.45 \pm 0.69</eq></td><td><eq>98.99 \pm 0.31</eq></td><td>93.30</td></tr><tr><td>Entropy</td><td><eq>72.48 \pm 1.24</eq></td><td><eq>35.35 \pm 1.65</eq></td><td><eq>61.92 \pm 1.32</eq></td><td><eq>50.96 \pm 1.57</eq></td><td><eq>23.76 \pm 1.05</eq></td><td><eq>17.55 \pm 0.92</eq></td><td>43.67</td></tr><tr><td>Entropy-M</td><td><eq>70.31 \pm 1.98</eq></td><td><eq>34.84 \pm 2.95</eq></td><td><eq>50.09 \pm 1.57</eq></td><td><eq>46.62 \pm 1.12</eq></td><td><eq>27.55 \pm 2.60</eq></td><td><eq>29.08 \pm 4.18</eq></td><td>43.08</td></tr><tr><td>DetectGPT</td><td><eq>75.62 \pm 1.20</eq></td><td><eq>57.18 \pm 1.91</eq></td><td><eq>77.25 \pm 0.66</eq></td><td><eq>61.97 \pm 0.90</eq></td><td><eq>58.01 \pm 1.75</eq></td><td><eq>20.42 \pm 1.07</eq></td><td>58.41</td></tr><tr><td>DetectGPT-M</td><td><eq>87.58 \pm 0.93</eq></td><td><eq>78.19 \pm 2.33</eq></td><td><eq>86.29 \pm 1.18</eq></td><td><eq>89.73 \pm 0.97</eq></td><td><eq>58.40 \pm 1.54</eq></td><td><eq>65.72 \pm 1.36</eq></td><td>77.65</td></tr><tr><td>FastGPT</td><td><eq>76.50 \pm 1.39</eq></td><td><eq>39.56 \pm 0.54</eq></td><td><eq>82.45 \pm 0.47</eq></td><td><eq>33.21 \pm 1.82</eq></td><td><eq>41.31 \pm 0.89</eq></td><td><eq>11.29 \pm 0.49</eq></td><td>47.39</td></tr><tr><td>FastGPT-M</td><td><eq>96.46 \pm 0.31</eq></td><td><eq>90.21 \pm 0.75</eq></td><td><eq>96.23 \pm 0.36</eq></td><td><eq>92.36 \pm 0.74</eq></td><td><eq>66.98 \pm 1.06</eq></td><td><eq>56.28 \pm 1.80</eq></td><td>83.09</td></tr><tr><td>DNA-GPT</td><td><eq>99.51 \pm 0.14</eq></td><td><eq>99.72 \pm 0.07</eq></td><td><eq>98.41 \pm 0.21</eq></td><td><eq>99.85 \pm 0.05</eq></td><td><eq>99.21 \pm 0.14</eq></td><td><eq>99.81 \pm 0.06</eq></td><td>99.42</td></tr><tr><td>DNA-GPT-M</td><td><eq>99.80 \pm 0.09</eq></td><td><eq>99.88 \pm 0.07</eq></td><td><eq>98.39 \pm 0.17</eq></td><td><eq>99.94 \pm 0.06</eq></td><td><eq>99.53 \pm 0.09</eq></td><td><eq>99.89 \pm 0.09</eq></td><td>99.57</td></tr></table>


Table 11: Performance on Reuters dataset. The detection models are trained on text generated by ChatGPT.


<table><tr><td>Metric</td><td>Method</td><td>GPT4All</td><td>ChatGPT</td><td>Dolly</td><td>ChatGLM</td><td>Claude</td><td>ChatGPT-turbo</td><td>Avg.</td></tr><tr><td rowspan="12">TPR@FPR-1%</td><td>Likelihood</td><td><eq>19.78 \pm 2.21</eq></td><td><eq>81.07 \pm 2.05</eq></td><td><eq>12.53 \pm 1.35</eq></td><td><eq>94.36 \pm 1.15</eq></td><td><eq>19.42 \pm 3.17</eq></td><td><eq>90.31 \pm 1.11</eq></td><td>52.91</td></tr><tr><td>Likelihood-M</td><td><eq>59.73 \pm 8.56</eq></td><td><eq>88.98 \pm 1.58</eq></td><td><eq>35.91 \pm 3.82</eq></td><td><eq>96.00 \pm 2.14</eq></td><td><eq>26.71 \pm 5.93</eq></td><td><eq>93.24 \pm 0.94</eq></td><td>66.76</td></tr><tr><td>Log-Rank</td><td><eq>29.96 \pm 3.29</eq></td><td><eq>86.22 \pm 1.87</eq></td><td><eq>15.20 \pm 1.70</eq></td><td><eq>97.33 \pm 0.64</eq></td><td><eq>24.58 \pm 3.35</eq></td><td><eq>93.20 \pm 1.13</eq></td><td>57.75</td></tr><tr><td>Log-Rank-M</td><td><eq>63.91 \pm 8.04</eq></td><td><eq>90.62 \pm 2.38</eq></td><td><eq>34.58 \pm 4.71</eq></td><td><eq>98.13 \pm 0.46</eq></td><td><eq>29.73 \pm 8.71</eq></td><td><eq>94.53 \pm 0.67</eq></td><td>68.59</td></tr><tr><td>Entropy</td><td><eq>0.09 \pm 0.11</eq></td><td><eq>2.49 \pm 1.25</eq></td><td><eq>2.04 \pm 0.57</eq></td><td><eq>1.69 \pm 0.54</eq></td><td><eq>4.36 \pm 1.81</eq></td><td><eq>9.56 \pm 2.90</eq></td><td>3.37</td></tr><tr><td>Entropy-M</td><td><eq>7.07 \pm 3.05</eq></td><td><eq>17.47 \pm 5.59</eq></td><td><eq>13.60 \pm 2.23</eq></td><td><eq>9.02 \pm 3.21</eq></td><td><eq>18.09 \pm 4.26</eq></td><td><eq>33.29 \pm 7.84</eq></td><td>16.42</td></tr><tr><td>DetectGPT</td><td><eq>0.00 \pm 0.00</eq></td><td><eq>0.00 \pm 0.00</eq></td><td><eq>0.00 \pm 0.00</eq></td><td><eq>0.00 \pm 0.00</eq></td><td><eq>0.44 \pm 0.40</eq></td><td><eq>0.04 \pm 0.09</eq></td><td>0.08</td></tr><tr><td>DetectGPT-M</td><td><eq>22.76 \pm 25.66</eq></td><td><eq>47.91 \pm 28.49</eq></td><td><eq>11.69 \pm 12.56</eq></td><td><eq>23.64 \pm 28.60</eq></td><td><eq>8.27 \pm 5.92</eq></td><td><eq>71.24 \pm 17.28</eq></td><td>30.92</td></tr><tr><td>FastGPT</td><td><eq>0.31 \pm 0.23</eq></td><td><eq>2.04 \pm 0.86</eq></td><td><eq>0.04 \pm 0.09</eq></td><td><eq>3.16 \pm 1.03</eq></td><td><eq>1.42 \pm 0.57</eq></td><td><eq>23.51 \pm 4.15</eq></td><td>5.08</td></tr><tr><td>FastGPT-M</td><td><eq>56.04 \pm 11.84</eq></td><td><eq>17.07 \pm 10.24</eq></td><td><eq>61.51 \pm 9.34</eq></td><td><eq>26.67 \pm 13.24</eq></td><td><eq>0.89 \pm 0.92</eq></td><td><eq>1.29 \pm 0.68</eq></td><td>27.24</td></tr><tr><td>DNA-GPT</td><td><eq>29.82 \pm 18.89</eq></td><td><eq>69.78 \pm 22.51</eq></td><td><eq>15.64 \pm 12.02</eq></td><td><eq>83.47 \pm 14.87</eq></td><td><eq>21.96 \pm 19.50</eq></td><td><eq>82.22 \pm 15.63</eq></td><td>50.48</td></tr><tr><td>DNA-GPT-M</td><td><eq>86.80 \pm 5.37</eq></td><td><eq>96.76 \pm 1.16</eq></td><td><eq>59.33 \pm 8.10</eq></td><td><eq>99.29 \pm 0.38</eq></td><td><eq>61.91 \pm 13.48</eq></td><td><eq>97.51 \pm 1.24</eq></td><td>83.60</td></tr><tr><td rowspan="12">AUROC</td><td>Likelihood</td><td><eq>75.19 \pm 1.39</eq></td><td><eq>97.55 \pm 0.33</eq></td><td><eq>58.77 \pm 1.76</eq></td><td><eq>99.62 \pm 0.11</eq></td><td><eq>86.46 \pm 0.70</eq></td><td><eq>98.42 \pm 0.25</eq></td><td>86.00</td></tr><tr><td>Likelihood-M</td><td><eq>93.97 \pm 1.27</eq></td><td><eq>98.68 \pm 0.41</eq></td><td><eq>79.54 \pm 1.12</eq></td><td><eq>99.70 \pm 0.24</eq></td><td><eq>90.39 \pm 1.08</eq></td><td><eq>98.94 \pm 0.23</eq></td><td>93.54</td></tr><tr><td>Log-Rank</td><td><eq>78.39 \pm 1.23</eq></td><td><eq>97.77 \pm 0.32</eq></td><td><eq>58.22 \pm 2.03</eq></td><td><eq>99.61 \pm 0.19</eq></td><td><eq>85.68 \pm 0.67</eq></td><td><eq>98.67 \pm 0.24</eq></td><td>86.39</td></tr><tr><td>Log-Rank-M</td><td><eq>93.41 \pm 1.37</eq></td><td><eq>98.69 \pm 0.25</eq></td><td><eq>76.76 \pm 1.45</eq></td><td><eq>99.75 \pm 0.20</eq></td><td><eq>89.55 \pm 1.04</eq></td><td><eq>99.03 \pm 0.27</eq></td><td>92.86</td></tr><tr><td>Entropy</td><td><eq>27.52 \pm 1.24</eq></td><td><eq>64.65 \pm 1.65</eq></td><td><eq>38.08 \pm 1.32</eq></td><td><eq>49.04 \pm 1.57</eq></td><td><eq>76.24 \pm 1.05</eq></td><td><eq>82.45 \pm 0.92</eq></td><td>56.33</td></tr><tr><td>Entropy-M</td><td><eq>55.30 \pm 2.65</eq></td><td><eq>80.07 \pm 1.89</eq></td><td><eq>56.54 \pm 2.20</eq></td><td><eq>69.75 \pm 2.62</eq></td><td><eq>83.33 \pm 1.09</eq></td><td><eq>90.46 \pm 1.17</eq></td><td>72.58</td></tr><tr><td>DetectGPT</td><td><eq>75.62 \pm 1.20</eq></td><td><eq>57.18 \pm 1.91</eq></td><td><eq>77.25 \pm 0.66</eq></td><td><eq>61.97 \pm 0.90</eq></td><td><eq>58.01 \pm 1.75</eq></td><td><eq>20.42 \pm 1.07</eq></td><td>58.41</td></tr><tr><td>DetectGPT-M</td><td><eq>84.60 \pm 11.69</eq></td><td><eq>92.37 \pm 5.63</eq></td><td><eq>69.47 \pm 12.90</eq></td><td><eq>80.47 \pm 13.77</eq></td><td><eq>75.78 \pm 9.49</eq></td><td><eq>97.20 \pm 1.23</eq></td><td>83.32</td></tr><tr><td>FastGPT</td><td><eq>23.50 \pm 1.39</eq></td><td><eq>60.44 \pm 0.54</eq></td><td><eq>17.55 \pm 0.47</eq></td><td><eq>66.79 \pm 1.82</eq></td><td><eq>58.69 \pm 0.89</eq></td><td><eq>88.71 \pm 0.49</eq></td><td>52.61</td></tr><tr><td>FastGPT-M</td><td><eq>96.25 \pm 0.33</eq></td><td><eq>89.97 \pm 0.74</eq></td><td><eq>96.04 \pm 0.33</eq></td><td><eq>92.12 \pm 0.76</eq></td><td><eq>66.72 \pm 1.11</eq></td><td><eq>56.04 \pm 1.73</eq></td><td>82.86</td></tr><tr><td>DNA-GPT</td><td><eq>97.34 \pm 0.54</eq></td><td><eq>99.17 \pm 0.30</eq></td><td><eq>95.82 \pm 0.62</eq></td><td><eq>99.45 \pm 0.23</eq></td><td><eq>97.21 \pm 0.52</eq></td><td><eq>99.38 \pm 0.26</eq></td><td>98.06</td></tr><tr><td>DNA-GPT-M</td><td><eq>99.42 \pm 0.15</eq></td><td><eq>99.78 \pm 0.09</eq></td><td><eq>97.31 \pm 0.24</eq></td><td><eq>99.89 \pm 0.10</eq></td><td><eq>98.78 \pm 0.24</eq></td><td><eq>99.82 \pm 0.12</eq></td><td>99.16</td></tr></table>


Table 12: Performance on Reuters dataset. The detection models are trained on text generated by ChatGPT-turbo.


<table><tr><td>Metric</td><td>Method</td><td>GPT4All</td><td>ChatGPT</td><td>Dolly</td><td>ChatGLM</td><td>Claude</td><td>ChatGPT-turbo</td><td>Avg.</td></tr><tr><td rowspan="12">TPR@FPR-1%</td><td>Likelihood</td><td><eq>19.82 \pm 2.17</eq></td><td><eq>81.07 \pm 2.05</eq></td><td><eq>12.53 \pm 1.35</eq></td><td><eq>94.44 \pm 1.10</eq></td><td><eq>19.42 \pm 3.17</eq></td><td><eq>90.27 \pm 1.07</eq></td><td>52.93</td></tr><tr><td>Likelihood-M</td><td><eq>66.62 \pm 5.21</eq></td><td><eq>90.13 \pm 1.15</eq></td><td><eq>38.62 \pm 2.74</eq></td><td><eq>95.96 \pm 1.29</eq></td><td><eq>29.60 \pm 2.32</eq></td><td><eq>93.91 \pm 1.03</eq></td><td>69.14</td></tr><tr><td>Log-Rank</td><td><eq>29.96 \pm 3.29</eq></td><td><eq>86.22 \pm 1.87</eq></td><td><eq>15.20 \pm 1.70</eq></td><td><eq>97.33 \pm 0.64</eq></td><td><eq>24.58 \pm 3.35</eq></td><td><eq>93.20 \pm 1.13</eq></td><td>57.75</td></tr><tr><td>Log-Rank-M</td><td><eq>60.18 \pm 8.59</eq></td><td><eq>90.27 \pm 1.98</eq></td><td><eq>33.07 \pm 3.84</eq></td><td><eq>98.22 \pm 0.40</eq></td><td><eq>27.29 \pm 6.31</eq></td><td><eq>94.00 \pm 0.40</eq></td><td>67.17</td></tr><tr><td>Entropy</td><td><eq>0.09 \pm 0.11</eq></td><td><eq>2.49 \pm 1.25</eq></td><td><eq>2.04 \pm 0.57</eq></td><td><eq>1.69 \pm 0.54</eq></td><td><eq>4.36 \pm 1.81</eq></td><td><eq>9.60 \pm 2.94</eq></td><td>3.38</td></tr><tr><td>Entropy-M</td><td><eq>5.11 \pm 2.70</eq></td><td><eq>16.18 \pm 4.91</eq></td><td><eq>12.27 \pm 1.51</eq></td><td><eq>7.69 \pm 2.70</eq></td><td><eq>17.42 \pm 3.93</eq></td><td><eq>31.69 \pm 6.61</eq></td><td>15.06</td></tr><tr><td>DetectGPT</td><td><eq>0.00 \pm 0.00</eq></td><td><eq>0.00 \pm 0.00</eq></td><td><eq>0.00 \pm 0.00</eq></td><td><eq>0.00 \pm 0.00</eq></td><td><eq>0.62 \pm 0.53</eq></td><td><eq>9.24 \pm 2.65</eq></td><td>1.64</td></tr><tr><td>DetectGPT-M</td><td><eq>6.22 \pm 3.99</eq></td><td><eq>26.71 \pm 14.32</eq></td><td><eq>2.49 \pm 0.97</eq></td><td><eq>4.04 \pm 3.33</eq></td><td><eq>4.18 \pm 2.22</eq></td><td><eq>60.67 \pm 14.38</eq></td><td>17.39</td></tr><tr><td>FastGPT</td><td><eq>0.31 \pm 0.23</eq></td><td><eq>2.04 \pm 0.86</eq></td><td><eq>0.04 \pm 0.09</eq></td><td><eq>3.16 \pm 1.03</eq></td><td><eq>1.42 \pm 0.57</eq></td><td><eq>23.51 \pm 4.15</eq></td><td>5.08</td></tr><tr><td>FastGPT-M</td><td><eq>0.00 \pm 0.00</eq></td><td><eq>0.22 \pm 0.28</eq></td><td><eq>0.04 \pm 0.09</eq></td><td><eq>0.53 \pm 0.33</eq></td><td><eq>0.62 \pm 0.29</eq></td><td><eq>7.38 \pm 2.42</eq></td><td>1.47</td></tr><tr><td>DNA-GPT</td><td><eq>2.09 \pm 1.29</eq></td><td><eq>4.31 \pm 1.94</eq></td><td><eq>1.69 \pm 1.21</eq></td><td><eq>5.51 \pm 1.67</eq></td><td><eq>1.96 \pm 1.60</eq></td><td><eq>5.24 \pm 2.29</eq></td><td>3.47</td></tr><tr><td>DNA-GPT-M</td><td><eq>5.82 \pm 1.06</eq></td><td><eq>6.44 \pm 0.97</eq></td><td><eq>3.56 \pm 1.27</eq></td><td><eq>6.49 \pm 0.87</eq></td><td><eq>3.78 \pm 1.29</eq></td><td><eq>6.67 \pm 0.84</eq></td><td>5.46</td></tr><tr><td rowspan="12">AUROC</td><td>Likelihood</td><td><eq>75.20 \pm 1.39</eq></td><td><eq>97.55 \pm 0.33</eq></td><td><eq>58.77 \pm 1.77</eq></td><td><eq>99.62 \pm 0.11</eq></td><td><eq>86.46 \pm 0.70</eq></td><td><eq>98.42 \pm 0.25</eq></td><td>86.00</td></tr><tr><td>Likelihood-M</td><td><eq>94.79 \pm 0.60</eq></td><td><eq>98.72 \pm 0.40</eq></td><td><eq>79.93 \pm 1.44</eq></td><td><eq>99.72 \pm 0.16</eq></td><td><eq>90.83 \pm 0.53</eq></td><td><eq>98.96 \pm 0.26</eq></td><td>93.82</td></tr><tr><td>Log-Rank</td><td><eq>78.39 \pm 1.23</eq></td><td><eq>97.77 \pm 0.32</eq></td><td><eq>58.22 \pm 2.03</eq></td><td><eq>99.61 \pm 0.19</eq></td><td><eq>85.68 \pm 0.67</eq></td><td><eq>98.67 \pm 0.24</eq></td><td>86.39</td></tr><tr><td>Log-Rank-M</td><td><eq>92.21 \pm 2.45</eq></td><td><eq>98.63 \pm 0.23</eq></td><td><eq>76.27 \pm 1.24</eq></td><td><eq>99.73 \pm 0.21</eq></td><td><eq>89.02 \pm 0.89</eq></td><td><eq>98.99 \pm 0.27</eq></td><td>92.48</td></tr><tr><td>Entropy</td><td><eq>27.52 \pm 1.24</eq></td><td><eq>64.65 \pm 1.65</eq></td><td><eq>38.08 \pm 1.32</eq></td><td><eq>49.04 \pm 1.57</eq></td><td><eq>76.24 \pm 1.05</eq></td><td><eq>82.45 \pm 0.92</eq></td><td>56.33</td></tr><tr><td>Entropy-M</td><td><eq>54.11 \pm 3.57</eq></td><td><eq>79.38 \pm 1.96</eq></td><td><eq>55.94 \pm 1.82</eq></td><td><eq>68.88 \pm 2.66</eq></td><td><eq>83.14 \pm 1.07</eq></td><td><eq>89.91 \pm 0.91</eq></td><td>71.89</td></tr><tr><td>DetectGPT</td><td><eq>24.38 \pm 1.20</eq></td><td><eq>42.82 \pm 1.91</eq></td><td><eq>22.75 \pm 0.66</eq></td><td><eq>38.03 \pm 0.90</eq></td><td><eq>41.99 \pm 1.75</eq></td><td><eq>79.58 \pm 1.07</eq></td><td>41.59</td></tr><tr><td>DetectGPT-M</td><td><eq>79.17 \pm 7.47</eq></td><td><eq>89.36 \pm 4.32</eq></td><td><eq>59.50 \pm 5.37</eq></td><td><eq>71.41 \pm 7.62</eq></td><td><eq>73.92 \pm 8.27</eq></td><td><eq>96.53 \pm 1.14</eq></td><td>78.32</td></tr><tr><td>FastGPT</td><td><eq>23.50 \pm 1.39</eq></td><td><eq>60.44 \pm 0.54</eq></td><td><eq>17.55 \pm 0.47</eq></td><td><eq>66.79 \pm 1.82</eq></td><td><eq>58.69 \pm 0.89</eq></td><td><eq>88.71 \pm 0.49</eq></td><td>52.61</td></tr><tr><td>FastGPT-M</td><td><eq>12.73 \pm 1.55</eq></td><td><eq>35.93 \pm 2.31</eq></td><td><eq>9.43 \pm 0.56</eq></td><td><eq>36.46 \pm 3.11</eq></td><td><eq>50.18 \pm 2.00</eq></td><td><eq>76.28 \pm 1.01</eq></td><td>36.83</td></tr><tr><td>DNA-GPT</td><td><eq>45.62 \pm 1.73</eq></td><td><eq>63.85 \pm 1.00</eq></td><td><eq>35.96 \pm 1.28</eq></td><td><eq>71.24 \pm 1.14</eq></td><td><eq>43.56 \pm 1.49</eq></td><td><eq>70.63 \pm 1.33</eq></td><td>55.14</td></tr><tr><td>DNA-GPT-M</td><td><eq>79.05 \pm 1.13</eq></td><td><eq>87.32 \pm 0.51</eq></td><td><eq>64.13 \pm 1.27</eq></td><td><eq>91.42 \pm 0.64</eq></td><td><eq>67.14 \pm 1.48</eq></td><td><eq>89.37 \pm 0.79</eq></td><td>79.74</td></tr></table>


Table 13: Performance on Reuters dataset. The detection models are trained on text generated by ChatGLM.


<table><tr><td>Metric</td><td>Method</td><td>GPT4All</td><td>ChatGPT</td><td>Dolly</td><td>ChatGLM</td><td>Claude</td><td>ChatGPT-turbo</td><td>Avg.</td></tr><tr><td rowspan="12">TPR@FPR-1%</td><td>Likelihood</td><td><eq>19.78 \pm 2.13</eq></td><td><eq>81.07 \pm 2.05</eq></td><td><eq>12.49 \pm 1.33</eq></td><td><eq>94.44 \pm 1.10</eq></td><td><eq>19.42 \pm 3.17</eq></td><td><eq>90.36 \pm 1.06</eq></td><td>52.93</td></tr><tr><td>Likelihood-M</td><td><eq>41.42 \pm 8.12</eq></td><td><eq>86.84 \pm 2.44</eq></td><td><eq>31.47 \pm 2.22</eq></td><td><eq>98.58 \pm 0.23</eq></td><td><eq>20.84 \pm 2.97</eq></td><td><eq>91.56 \pm 1.28</eq></td><td>61.79</td></tr><tr><td>Log-Rank</td><td><eq>29.96 \pm 3.29</eq></td><td><eq>86.22 \pm 1.87</eq></td><td><eq>15.20 \pm 1.70</eq></td><td><eq>97.33 \pm 0.64</eq></td><td><eq>24.58 \pm 3.35</eq></td><td><eq>93.20 \pm 1.13</eq></td><td>57.75</td></tr><tr><td>Log-Rank-M</td><td><eq>62.40 \pm 4.54</eq></td><td><eq>90.49 \pm 2.34</eq></td><td><eq>33.82 \pm 3.87</eq></td><td><eq>98.22 \pm 0.47</eq></td><td><eq>30.44 \pm 6.61</eq></td><td><eq>94.13 \pm 0.75</eq></td><td>68.25</td></tr><tr><td>Entropy</td><td><eq>4.89 \pm 2.71</eq></td><td><eq>0.36 \pm 0.71</eq></td><td><eq>10.49 \pm 4.75</eq></td><td><eq>0.44 \pm 0.37</eq></td><td><eq>0.58 \pm 1.16</eq></td><td><eq>1.38 \pm 2.31</eq></td><td>3.02</td></tr><tr><td>Entropy-M</td><td><eq>6.13 \pm 4.80</eq></td><td><eq>17.20 \pm 4.05</eq></td><td><eq>13.11 \pm 3.21</eq></td><td><eq>9.73 \pm 4.98</eq></td><td><eq>18.76 \pm 2.64</eq></td><td><eq>33.87 \pm 5.03</eq></td><td>16.47</td></tr><tr><td>DetectGPT</td><td><eq>0.00 \pm 0.00</eq></td><td><eq>0.00 \pm 0.00</eq></td><td><eq>0.00 \pm 0.00</eq></td><td><eq>0.00 \pm 0.00</eq></td><td><eq>0.44 \pm 0.40</eq></td><td><eq>0.04 \pm 0.09</eq></td><td>0.08</td></tr><tr><td>DetectGPT-M</td><td><eq>40.53 \pm 6.43</eq></td><td><eq>14.18 \pm 7.67</eq></td><td><eq>45.73 \pm 2.50</eq></td><td><eq>38.98 \pm 8.03</eq></td><td><eq>2.18 \pm 0.53</eq></td><td><eq>5.47 \pm 2.15</eq></td><td>24.51</td></tr><tr><td>FastGPT</td><td><eq>0.31 \pm 0.23</eq></td><td><eq>2.04 \pm 0.86</eq></td><td><eq>0.04 \pm 0.09</eq></td><td><eq>3.16 \pm 1.03</eq></td><td><eq>1.42 \pm 0.57</eq></td><td><eq>23.51 \pm 4.15</eq></td><td>5.08</td></tr><tr><td>FastGPT-M</td><td><eq>56.04 \pm 11.84</eq></td><td><eq>17.07 \pm 10.24</eq></td><td><eq>61.51 \pm 9.34</eq></td><td><eq>26.67 \pm 13.24</eq></td><td><eq>0.89 \pm 0.92</eq></td><td><eq>1.29 \pm 0.68</eq></td><td>27.24</td></tr><tr><td>DNA-GPT</td><td><eq>27.96 \pm 18.31</eq></td><td><eq>66.84 \pm 18.61</eq></td><td><eq>16.00 \pm 12.45</eq></td><td><eq>85.51 \pm 12.79</eq></td><td><eq>20.49 \pm 17.96</eq></td><td><eq>80.40 \pm 13.38</eq></td><td>49.53</td></tr><tr><td>DNA-GPT-M</td><td><eq>82.89 \pm 7.12</eq></td><td><eq>94.93 \pm 1.99</eq></td><td><eq>56.31 \pm 9.16</eq></td><td><eq>98.98 \pm 0.18</eq></td><td><eq>57.78 \pm 15.40</eq></td><td><eq>96.53 \pm 1.23</eq></td><td>81.24</td></tr><tr><td rowspan="12">AUROC</td><td>Likelihood</td><td><eq>75.19 \pm 1.39</eq></td><td><eq>97.55 \pm 0.33</eq></td><td><eq>58.77 \pm 1.77</eq></td><td><eq>99.62 \pm 0.11</eq></td><td><eq>86.46 \pm 0.70</eq></td><td><eq>98.42 \pm 0.25</eq></td><td>86.00</td></tr><tr><td>Likelihood-M</td><td><eq>87.15 \pm 3.71</eq></td><td><eq>98.44 \pm 0.43</eq></td><td><eq>77.92 \pm 1.74</eq></td><td><eq>99.95 \pm 0.00</eq></td><td><eq>87.99 \pm 1.16</eq></td><td><eq>98.69 \pm 0.26</eq></td><td>91.69</td></tr><tr><td>Log-Rank</td><td><eq>78.39 \pm 1.23</eq></td><td><eq>97.77 \pm 0.32</eq></td><td><eq>58.22 \pm 2.03</eq></td><td><eq>99.61 \pm 0.19</eq></td><td><eq>85.68 \pm 0.67</eq></td><td><eq>98.67 \pm 0.24</eq></td><td>86.39</td></tr><tr><td>Log-Rank-M</td><td><eq>93.39 \pm 0.75</eq></td><td><eq>98.68 \pm 0.24</eq></td><td><eq>76.11 \pm 1.61</eq></td><td><eq>99.75 \pm 0.19</eq></td><td><eq>89.70 \pm 0.41</eq></td><td><eq>98.99 \pm 0.25</eq></td><td>92.77</td></tr><tr><td>Entropy</td><td><eq>62.76 \pm 18.55</eq></td><td><eq>40.53 \pm 11.30</eq></td><td><eq>56.16 \pm 10.29</eq></td><td><eq>49.39 \pm 1.74</eq></td><td><eq>33.61 \pm 20.52</eq></td><td><eq>30.23 \pm 25.75</eq></td><td>45.45</td></tr><tr><td>Entropy-M</td><td><eq>54.30 \pm 3.63</eq></td><td><eq>79.38 \pm 1.33</eq></td><td><eq>55.89 \pm 1.46</eq></td><td><eq>69.08 \pm 2.00</eq></td><td><eq>83.03 \pm 0.44</eq></td><td><eq>89.77 \pm 1.24</eq></td><td>71.91</td></tr><tr><td>DetectGPT</td><td><eq>75.62 \pm 1.20</eq></td><td><eq>57.18 \pm 1.91</eq></td><td><eq>77.25 \pm 0.66</eq></td><td><eq>61.97 \pm 0.90</eq></td><td><eq>58.01 \pm 1.75</eq></td><td><eq>20.42 \pm 1.07</eq></td><td>58.41</td></tr><tr><td>DetectGPT-M</td><td><eq>87.17 \pm 0.76</eq></td><td><eq>77.46 \pm 1.85</eq></td><td><eq>86.14 \pm 1.14</eq></td><td><eq>89.36 \pm 1.25</eq></td><td><eq>57.83 \pm 1.55</eq></td><td><eq>64.40 \pm 1.89</eq></td><td>77.06</td></tr><tr><td>FastGPT</td><td><eq>23.50 \pm 1.39</eq></td><td><eq>60.44 \pm 0.54</eq></td><td><eq>17.55 \pm 0.47</eq></td><td><eq>66.79 \pm 1.82</eq></td><td><eq>58.69 \pm 0.89</eq></td><td><eq>88.71 \pm 0.49</eq></td><td>52.61</td></tr><tr><td>FastGPT-M</td><td><eq>96.27 \pm 0.33</eq></td><td><eq>89.98 \pm 0.74</eq></td><td><eq>96.07 \pm 0.33</eq></td><td><eq>92.13 \pm 0.76</eq></td><td><eq>66.74 \pm 1.10</eq></td><td><eq>56.05 \pm 1.74</eq></td><td>82.87</td></tr><tr><td>DNA-GPT</td><td><eq>97.10 \pm 0.48</eq></td><td><eq>98.94 \pm 0.26</eq></td><td><eq>95.56 \pm 0.57</eq></td><td><eq>99.29 \pm 0.17</eq></td><td><eq>96.95 \pm 0.46</eq></td><td><eq>99.22 \pm 0.22</eq></td><td>97.85</td></tr><tr><td>DNA-GPT-M</td><td><eq>99.33 \pm 0.15</eq></td><td><eq>99.72 \pm 0.09</eq></td><td><eq>97.15 \pm 0.21</eq></td><td><eq>99.82 \pm 0.10</eq></td><td><eq>98.68 \pm 0.24</eq></td><td><eq>99.77 \pm 0.12</eq></td><td>99.08</td></tr></table>


Table 14: Performance on Reuters dataset. The detection models are trained on text generated by Dolly.


<table><tr><td>Metric</td><td>Method</td><td>GPT4All</td><td>ChatGPT</td><td>Dolly</td><td>ChatGLM</td><td>Claude</td><td>ChatGPT-turbo</td><td>Avg.</td></tr><tr><td rowspan="12">TPR@FPR-1%</td><td>Likelihood</td><td><eq>19.78 \pm 2.21</eq></td><td><eq>81.07 \pm 2.05</eq></td><td><eq>12.40 \pm 1.37</eq></td><td><eq>94.44 \pm 1.10</eq></td><td><eq>19.38 \pm 3.23</eq></td><td><eq>90.27 \pm 1.07</eq></td><td>52.89</td></tr><tr><td>Likelihood-M</td><td><eq>59.96 \pm 5.45</eq></td><td><eq>88.58 \pm 2.29</eq></td><td><eq>35.42 \pm 2.84</eq></td><td><eq>95.91 \pm 1.21</eq></td><td><eq>31.96 \pm 6.22</eq></td><td><eq>92.93 \pm 0.77</eq></td><td>67.46</td></tr><tr><td>Log-Rank</td><td><eq>6.62 \pm 11.36</eq></td><td><eq>17.24 \pm 34.49</eq></td><td><eq>10.49 \pm 2.42</eq></td><td><eq>19.42 \pm 38.62</eq></td><td><eq>4.93 \pm 9.87</eq></td><td><eq>18.58 \pm 37.16</eq></td><td>12.88</td></tr><tr><td>Log-Rank-M</td><td><eq>66.76 \pm 7.01</eq></td><td><eq>90.49 \pm 1.80</eq></td><td><eq>35.42 \pm 4.26</eq></td><td><eq>98.09 \pm 0.72</eq></td><td><eq>31.78 \pm 6.22</eq></td><td><eq>94.31 \pm 0.39</eq></td><td>69.47</td></tr><tr><td>Entropy</td><td><eq>6.18 \pm 1.40</eq></td><td><eq>0.04 \pm 0.09</eq></td><td><eq>13.07 \pm 1.76</eq></td><td><eq>0.22 \pm 0.20</eq></td><td><eq>0.00 \pm 0.00</eq></td><td><eq>0.22 \pm 0.00</eq></td><td>3.29</td></tr><tr><td>Entropy-M</td><td><eq>2.40 \pm 0.74</eq></td><td><eq>7.29 \pm 5.90</eq></td><td><eq>9.91 \pm 1.09</eq></td><td><eq>3.16 \pm 2.61</eq></td><td><eq>9.24 \pm 7.67</eq></td><td><eq>16.13 \pm 13.02</eq></td><td>8.02</td></tr><tr><td>DetectGPT</td><td><eq>0.00 \pm 0.00</eq></td><td><eq>0.00 \pm 0.00</eq></td><td><eq>0.00 \pm 0.00</eq></td><td><eq>0.00 \pm 0.00</eq></td><td><eq>0.44 \pm 0.40</eq></td><td><eq>0.04 \pm 0.09</eq></td><td>0.08</td></tr><tr><td>DetectGPT-M</td><td><eq>42.93 \pm 25.42</eq></td><td><eq>37.20 \pm 35.68</eq></td><td><eq>39.69 \pm 19.14</eq></td><td><eq>51.73 \pm 37.94</eq></td><td><eq>6.27 \pm 7.20</eq></td><td><eq>31.56 \pm 34.65</eq></td><td>34.90</td></tr><tr><td>FastGPT</td><td><eq>3.56 \pm 3.08</eq></td><td><eq>0.04 \pm 0.09</eq></td><td><eq>12.18 \pm 6.70</eq></td><td><eq>0.09 \pm 0.11</eq></td><td><eq>0.09 \pm 0.11</eq></td><td><eq>0.04 \pm 0.09</eq></td><td>2.67</td></tr><tr><td>FastGPT-M</td><td><eq>61.69 \pm 7.18</eq></td><td><eq>21.78 \pm 6.77</eq></td><td><eq>65.24 \pm 5.12</eq></td><td><eq>31.78 \pm 9.97</eq></td><td><eq>1.38 \pm 0.84</eq></td><td><eq>1.51 \pm 0.55</eq></td><td>30.56</td></tr><tr><td>DNA-GPT</td><td><eq>78.71 \pm 8.56</eq></td><td><eq>91.47 \pm 7.33</eq></td><td><eq>70.00 \pm 10.40</eq></td><td><eq>95.51 \pm 4.74</eq></td><td><eq>76.76 \pm 8.45</eq></td><td><eq>94.80 \pm 5.80</eq></td><td>84.54</td></tr><tr><td>DNA-GPT-M</td><td><eq>93.24 \pm 5.57</eq></td><td><eq>98.09 \pm 1.34</eq></td><td><eq>79.07 \pm 7.29</eq></td><td><eq>99.87 \pm 0.27</eq></td><td><eq>85.20 \pm 6.66</eq></td><td><eq>98.31 \pm 1.35</eq></td><td>92.30</td></tr><tr><td rowspan="12">AUROC</td><td>Likelihood</td><td><eq>75.19 \pm 1.39</eq></td><td><eq>97.55 \pm 0.33</eq></td><td><eq>58.77 \pm 1.76</eq></td><td><eq>99.62 \pm 0.10</eq></td><td><eq>86.45 \pm 0.70</eq></td><td><eq>98.42 \pm 0.25</eq></td><td>86.00</td></tr><tr><td>Likelihood-M</td><td><eq>93.42 \pm 0.91</eq></td><td><eq>98.45 \pm 0.29</eq></td><td><eq>77.41 \pm 1.04</eq></td><td><eq>99.67 \pm 0.16</eq></td><td><eq>90.57 \pm 0.61</eq></td><td><eq>98.78 \pm 0.33</eq></td><td>93.05</td></tr><tr><td>Log-Rank</td><td><eq>32.84 \pm 22.66</eq></td><td><eq>21.28 \pm 38.17</eq></td><td><eq>43.51 \pm 5.44</eq></td><td><eq>20.22 \pm 39.67</eq></td><td><eq>28.30 \pm 28.33</eq></td><td><eq>20.77 \pm 38.91</eq></td><td>27.82</td></tr><tr><td>Log-Rank-M</td><td><eq>94.18 \pm 0.71</eq></td><td><eq>98.71 \pm 0.19</eq></td><td><eq>76.52 \pm 1.64</eq></td><td><eq>99.73 \pm 0.23</eq></td><td><eq>89.98 \pm 0.70</eq></td><td><eq>99.02 \pm 0.31</eq></td><td>93.02</td></tr><tr><td>Entropy</td><td><eq>72.48 \pm 1.24</eq></td><td><eq>35.35 \pm 1.65</eq></td><td><eq>61.92 \pm 1.32</eq></td><td><eq>50.96 \pm 1.57</eq></td><td><eq>23.76 \pm 1.05</eq></td><td><eq>17.55 \pm 0.92</eq></td><td>43.67</td></tr><tr><td>Entropy-M</td><td><eq>49.80 \pm 2.95</eq></td><td><eq>56.16 \pm 26.44</eq></td><td><eq>52.47 \pm 1.86</eq></td><td><eq>53.09 \pm 16.63</eq></td><td><eq>56.78 \pm 31.30</eq></td><td><eq>58.33 \pm 37.31</eq></td><td>54.44</td></tr><tr><td>DetectGPT</td><td><eq>75.62 \pm 1.20</eq></td><td><eq>57.18 \pm 1.91</eq></td><td><eq>77.25 \pm 0.66</eq></td><td><eq>61.97 \pm 0.90</eq></td><td><eq>58.01 \pm 1.75</eq></td><td><eq>20.42 \pm 1.07</eq></td><td>58.41</td></tr><tr><td>DetectGPT-M</td><td><eq>80.34 \pm 21.91</eq></td><td><eq>73.22 \pm 28.90</eq></td><td><eq>81.89 \pm 11.41</eq></td><td><eq>91.04 \pm 8.24</eq></td><td><eq>57.53 \pm 18.26</eq></td><td><eq>66.30 \pm 30.95</eq></td><td>75.05</td></tr><tr><td>FastGPT</td><td><eq>76.50 \pm 1.39</eq></td><td><eq>39.56 \pm 0.54</eq></td><td><eq>82.45 \pm 0.47</eq></td><td><eq>33.21 \pm 1.82</eq></td><td><eq>41.31 \pm 0.89</eq></td><td><eq>11.29 \pm 0.49</eq></td><td>47.39</td></tr><tr><td>FastGPT-M</td><td><eq>96.47 \pm 0.30</eq></td><td><eq>90.23 \pm 0.73</eq></td><td><eq>96.24 \pm 0.37</eq></td><td><eq>92.37 \pm 0.72</eq></td><td><eq>67.04 \pm 1.07</eq></td><td><eq>56.34 \pm 1.90</eq></td><td>83.12</td></tr><tr><td>DNA-GPT</td><td><eq>99.19 \pm 0.21</eq></td><td><eq>99.74 \pm 0.09</eq></td><td><eq>98.90 \pm 0.24</eq></td><td><eq>99.83 \pm 0.07</eq></td><td><eq>99.21 \pm 0.16</eq></td><td><eq>99.80 \pm 0.08</eq></td><td>99.45</td></tr><tr><td>DNA-GPT-M</td><td><eq>99.66 \pm 0.19</eq></td><td><eq>99.84 \pm 0.13</eq></td><td><eq>98.81 \pm 0.22</eq></td><td><eq>99.91 \pm 0.10</eq></td><td><eq>99.47 \pm 0.20</eq></td><td><eq>99.86 \pm 0.14</eq></td><td>99.59</td></tr></table>


Table 15: Performance on Reuters dataset. The detection models are trained on text generated by Claude.


<table><tr><td>Metric</td><td>Method</td><td>GPT4All</td><td>ChatGPT</td><td>ChatGPT-turbo</td><td>ChatGLM</td><td>StableLM</td><td>Claude</td><td>Avg.</td></tr><tr><td rowspan="12">TPR@FPR-1%</td><td>Likelihood</td><td><eq>19.82 \pm 2.17</eq></td><td><eq>81.07 \pm 2.05</eq></td><td><eq>12.53 \pm 1.35</eq></td><td><eq>94.44 \pm 1.10</eq></td><td><eq>19.42 \pm 3.17</eq></td><td><eq>90.36 \pm 1.06</eq></td><td>52.94</td></tr><tr><td>Likelihood-M</td><td><eq>60.18 \pm 5.26</eq></td><td><eq>84.84 \pm 2.91</eq></td><td><eq>32.13 \pm 2.78</eq></td><td><eq>91.78 \pm 1.51</eq></td><td><eq>34.58 \pm 4.58</eq></td><td><eq>91.47 \pm 1.34</eq></td><td>65.83</td></tr><tr><td>Log-Rank</td><td><eq>29.96 \pm 3.29</eq></td><td><eq>86.22 \pm 1.87</eq></td><td><eq>15.20 \pm 1.70</eq></td><td><eq>97.33 \pm 0.64</eq></td><td><eq>24.58 \pm 3.35</eq></td><td><eq>93.20 \pm 1.13</eq></td><td>57.75</td></tr><tr><td>Log-Rank-M</td><td><eq>71.33 \pm 4.00</eq></td><td><eq>89.82 \pm 2.47</eq></td><td><eq>37.60 \pm 1.64</eq></td><td><eq>97.07 \pm 1.44</eq></td><td><eq>34.84 \pm 5.56</eq></td><td><eq>94.36 \pm 0.87</eq></td><td>70.84</td></tr><tr><td>Entropy</td><td><eq>0.09 \pm 0.11</eq></td><td><eq>2.49 \pm 1.25</eq></td><td><eq>2.04 \pm 0.57</eq></td><td><eq>1.69 \pm 0.54</eq></td><td><eq>4.36 \pm 1.81</eq></td><td><eq>9.60 \pm 2.94</eq></td><td>3.38</td></tr><tr><td>Entropy-M</td><td><eq>5.51 \pm 2.56</eq></td><td><eq>17.42 \pm 5.92</eq></td><td><eq>13.24 \pm 2.62</eq></td><td><eq>7.91 \pm 2.58</eq></td><td><eq>18.00 \pm 4.28</eq></td><td><eq>34.09 \pm 8.23</eq></td><td>16.03</td></tr><tr><td>DetectGPT</td><td><eq>0.00 \pm 0.00</eq></td><td><eq>0.00 \pm 0.00</eq></td><td><eq>0.00 \pm 0.00</eq></td><td><eq>0.00 \pm 0.00</eq></td><td><eq>0.44 \pm 0.40</eq></td><td><eq>0.04 \pm 0.09</eq></td><td>0.08</td></tr><tr><td>DetectGPT-M</td><td><eq>39.91 \pm 20.48</eq></td><td><eq>51.60 \pm 22.14</eq></td><td><eq>28.80 \pm 20.74</eq></td><td><eq>57.47 \pm 35.18</eq></td><td><eq>6.00 \pm 3.22</eq></td><td><eq>78.53 \pm 8.07</eq></td><td>43.72</td></tr><tr><td>FastGPT</td><td><eq>0.31 \pm 0.23</eq></td><td><eq>2.04 \pm 0.86</eq></td><td><eq>0.04 \pm 0.09</eq></td><td><eq>3.16 \pm 1.03</eq></td><td><eq>1.42 \pm 0.57</eq></td><td><eq>23.51 \pm 4.15</eq></td><td>5.08</td></tr><tr><td>FastGPT-M</td><td><eq>59.96 \pm 10.36</eq></td><td><eq>33.11 \pm 28.98</eq></td><td><eq>56.09 \pm 16.45</eq></td><td><eq>42.44 \pm 27.14</eq></td><td><eq>8.62 \pm 15.04</eq></td><td><eq>19.64 \pm 36.52</eq></td><td>36.64</td></tr><tr><td>DNA-GPT</td><td><eq>3.64 \pm 1.52</eq></td><td><eq>6.58 \pm 1.66</eq></td><td><eq>2.18 \pm 1.15</eq></td><td><eq>7.78 \pm 0.95</eq></td><td><eq>2.44 \pm 2.03</eq></td><td><eq>7.20 \pm 1.15</eq></td><td>4.97</td></tr><tr><td>DNA-GPT-M</td><td><eq>6.67 \pm 2.96</eq></td><td><eq>7.16 \pm 3.18</eq></td><td><eq>4.22 \pm 2.00</eq></td><td><eq>7.51 \pm 3.26</eq></td><td><eq>5.38 \pm 2.87</eq></td><td><eq>7.16 \pm 3.17</eq></td><td>6.35</td></tr><tr><td rowspan="12">AUROC</td><td>Likelihood</td><td><eq>75.19 \pm 1.39</eq></td><td><eq>97.55 \pm 0.33</eq></td><td><eq>58.77 \pm 1.77</eq></td><td><eq>99.62 \pm 0.10</eq></td><td><eq>86.45 \pm 0.70</eq></td><td><eq>98.42 \pm 0.25</eq></td><td>86.00</td></tr><tr><td>Likelihood-M</td><td><eq>91.12 \pm 2.58</eq></td><td><eq>97.69 \pm 0.49</eq></td><td><eq>69.45 \pm 3.64</eq></td><td><eq>98.38 \pm 0.57</eq></td><td><eq>90.57 \pm 0.59</eq></td><td><eq>98.54 \pm 0.43</eq></td><td>90.96</td></tr><tr><td>Log-Rank</td><td><eq>78.40 \pm 1.23</eq></td><td><eq>97.77 \pm 0.32</eq></td><td><eq>58.22 \pm 2.03</eq></td><td><eq>99.61 \pm 0.19</eq></td><td><eq>85.68 \pm 0.67</eq></td><td><eq>98.67 \pm 0.24</eq></td><td>86.39</td></tr><tr><td>Log-Rank-M</td><td><eq>95.06 \pm 0.64</eq></td><td><eq>98.59 \pm 0.28</eq></td><td><eq>76.35 \pm 1.49</eq></td><td><eq>99.65 \pm 0.22</eq></td><td><eq>90.36 \pm 0.68</eq></td><td><eq>98.98 \pm 0.27</eq></td><td>93.16</td></tr><tr><td>Entropy</td><td><eq>27.52 \pm 1.24</eq></td><td><eq>64.65 \pm 1.65</eq></td><td><eq>38.08 \pm 1.32</eq></td><td><eq>49.04 \pm 1.57</eq></td><td><eq>76.24 \pm 1.05</eq></td><td><eq>82.45 \pm 0.92</eq></td><td>56.33</td></tr><tr><td>Entropy-M</td><td><eq>53.47 \pm 2.35</eq></td><td><eq>79.33 \pm 1.73</eq></td><td><eq>55.92 \pm 1.77</eq></td><td><eq>68.50 \pm 2.29</eq></td><td><eq>83.03 \pm 1.03</eq></td><td><eq>90.09 \pm 0.94</eq></td><td>71.72</td></tr><tr><td>DetectGPT</td><td><eq>75.62 \pm 1.20</eq></td><td><eq>57.18 \pm 1.91</eq></td><td><eq>77.25 \pm 0.66</eq></td><td><eq>61.97 \pm 0.90</eq></td><td><eq>58.01 \pm 1.75</eq></td><td><eq>20.42 \pm 1.07</eq></td><td>58.41</td></tr><tr><td>DetectGPT-M</td><td><eq>91.23 \pm 5.89</eq></td><td><eq>94.13 \pm 1.77</eq></td><td><eq>81.29 \pm 8.98</eq></td><td><eq>94.63 \pm 5.95</eq></td><td><eq>70.99 \pm 6.72</eq></td><td><eq>96.97 \pm 1.09</eq></td><td>88.21</td></tr><tr><td>FastGPT</td><td><eq>23.50 \pm 1.39</eq></td><td><eq>60.44 \pm 0.54</eq></td><td><eq>17.55 \pm 0.47</eq></td><td><eq>66.79 \pm 1.82</eq></td><td><eq>58.69 \pm 0.89</eq></td><td><eq>88.71 \pm 0.49</eq></td><td>52.61</td></tr><tr><td>FastGPT-M</td><td><eq>96.53 \pm 0.51</eq></td><td><eq>91.64 \pm 3.61</eq></td><td><eq>94.46 \pm 2.87</eq></td><td><eq>93.63 \pm 3.03</eq></td><td><eq>71.74 \pm 10.00</eq></td><td><eq>63.82 \pm 17.64</eq></td><td>85.30</td></tr><tr><td>DNA-GPT</td><td><eq>65.77 \pm 1.29</eq></td><td><eq>78.38 \pm 0.64</eq></td><td><eq>54.85 \pm 1.20</eq></td><td><eq>82.40 \pm 0.73</eq></td><td><eq>65.30 \pm 0.67</eq></td><td><eq>81.64 \pm 0.76</eq></td><td>71.39</td></tr><tr><td>DNA-GPT-M</td><td><eq>78.84 \pm 12.13</eq></td><td><eq>83.32 \pm 13.75</eq></td><td><eq>68.13 \pm 8.88</eq></td><td><eq>85.53 \pm 14.38</eq></td><td><eq>72.95 \pm 9.83</eq></td><td><eq>84.29 \pm 14.01</eq></td><td>78.84</td></tr></table>


Table 16: Performance on TruthfulQA dataset. The detection models are trained on text generated by GPT4.


<table><tr><td>Metric</td><td>Method</td><td>GPT4</td><td>ChatGPT-turbo</td><td>ChatGLM</td><td>Dolly</td><td>ChatGPT</td><td>StableLM</td><td>Avg.</td></tr><tr><td rowspan="12">TPR@FPR-1%</td><td>Likelihood</td><td><eq>38.45 \pm 3.34</eq></td><td><eq>41.67 \pm 1.51</eq></td><td><eq>73.54 \pm 7.83</eq></td><td><eq>17.74 \pm 2.72</eq></td><td><eq>52.01 \pm 10.78</eq></td><td><eq>50.14 \pm 8.09</eq></td><td>45.59</td></tr><tr><td>Likelihood-E</td><td><eq>29.17 \pm 11.92</eq></td><td><eq>56.89 \pm 19.43</eq></td><td><eq>75.91 \pm 17.62</eq></td><td><eq>52.17 \pm 2.62</eq></td><td><eq>42.04 \pm 19.77</eq></td><td><eq>54.79 \pm 10.29</eq></td><td>51.83</td></tr><tr><td>Log-Rank</td><td><eq>40.17 \pm 4.46</eq></td><td><eq>39.77 \pm 7.89</eq></td><td><eq>83.22 \pm 4.08</eq></td><td><eq>16.45 \pm 5.07</eq></td><td><eq>54.50 \pm 2.09</eq></td><td><eq>59.26 \pm 2.20</eq></td><td>48.9</td></tr><tr><td>Log-Rank-E</td><td><eq>47.74 \pm 1.72</eq></td><td><eq>75.39 \pm 4.05</eq></td><td><eq>90.55 \pm 2.10</eq></td><td><eq>44.48 \pm 4.99</eq></td><td><eq>83.57 \pm 1.73</eq></td><td><eq>79.26 \pm 2.52</eq></td><td>70.16</td></tr><tr><td>Entropy</td><td><eq>24.30 \pm 4.47</eq></td><td><eq>38.39 \pm 6.85</eq></td><td><eq>60.48 \pm 15.23</eq></td><td><eq>18.38 \pm 6.58</eq></td><td><eq>38.02 \pm 9.81</eq></td><td><eq>32.29 \pm 14.76</eq></td><td>35.31</td></tr><tr><td>Entropy-E</td><td><eq>39.94 \pm 3.25</eq></td><td><eq>78.44 \pm 6.63</eq></td><td><eq>84.65 \pm 18.12</eq></td><td><eq>52.70 \pm 13.35</eq></td><td><eq>78.87 \pm 9.18</eq></td><td><eq>69.52 \pm 11.82</eq></td><td>67.35</td></tr><tr><td>DetectGPT</td><td><eq>21.66 \pm 7.94</eq></td><td><eq>42.65 \pm 6.28</eq></td><td><eq>60.82 \pm 6.22</eq></td><td><eq>13.75 \pm 3.45</eq></td><td><eq>52.41 \pm 7.71</eq></td><td><eq>39.49 \pm 4.15</eq></td><td>38.46</td></tr><tr><td>DetectGPT-E</td><td><eq>46.30 \pm 2.60</eq></td><td><eq>82.31 \pm 4.48</eq></td><td><eq>94.85 \pm 0.92</eq></td><td><eq>59.52 \pm 4.78</eq></td><td><eq>86.91 \pm 3.54</eq></td><td><eq>84.14 \pm 2.65</eq></td><td>75.67</td></tr><tr><td>FastGPT</td><td><eq>21.20 \pm 4.54</eq></td><td><eq>47.15 \pm 3.59</eq></td><td><eq>66.61 \pm 6.47</eq></td><td><eq>20.63 \pm 4.37</eq></td><td><eq>52.58 \pm 9.86</eq></td><td><eq>49.92 \pm 6.27</eq></td><td>43.01</td></tr><tr><td>FastGPT-E</td><td><eq>37.25 \pm 4.38</eq></td><td><eq>74.64 \pm 4.96</eq></td><td><eq>89.80 \pm 5.01</eq></td><td><eq>49.06 \pm 2.23</eq></td><td><eq>83.80 \pm 4.03</eq></td><td><eq>76.32 \pm 5.07</eq></td><td>68.48</td></tr><tr><td>DNAGPT</td><td><eq>10.37 \pm 3.51</eq></td><td><eq>15.62 \pm 4.68</eq></td><td><eq>30.59 \pm 5.19</eq></td><td><eq>8.28 \pm 2.27</eq></td><td><eq>20.74 \pm 5.36</eq></td><td><eq>22.04 \pm 1.97</eq></td><td>17.94</td></tr><tr><td>DNAGPT</td><td><eq>8.08 \pm 2.57</eq></td><td><eq>15.68 \pm 6.48</eq></td><td><eq>20.39 \pm 3.96</eq></td><td><eq>11.52 \pm 6.76</eq></td><td><eq>21.47 \pm 10.11</eq></td><td><eq>13.99 \pm 6.07</eq></td><td>15.19</td></tr><tr><td rowspan="12">AUROC</td><td>Likelihood</td><td><eq>84.67 \pm 0.88</eq></td><td><eq>91.95 \pm 0.30</eq></td><td><eq>97.08 \pm 0.29</eq></td><td><eq>79.28 \pm 0.48</eq></td><td><eq>95.65 \pm 0.40</eq></td><td><eq>94.17 \pm 0.30</eq></td><td>90.47</td></tr><tr><td>Likelihood-E</td><td><eq>86.61 \pm 0.56</eq></td><td><eq>92.72 \pm 2.67</eq></td><td><eq>97.14 \pm 0.92</eq></td><td><eq>83.70 \pm 0.57</eq></td><td><eq>94.20 \pm 1.87</eq></td><td><eq>94.37 \pm 0.92</eq></td><td>91.46</td></tr><tr><td>Log-Rank</td><td><eq>82.55 \pm 0.80</eq></td><td><eq>90.60 \pm 0.30</eq></td><td><eq>97.03 \pm 0.31</eq></td><td><eq>78.08 \pm 0.70</eq></td><td><eq>95.11 \pm 0.44</eq></td><td><eq>94.09 \pm 0.38</eq></td><td>89.57</td></tr><tr><td>Log-Rank-E</td><td><eq>84.16 \pm 0.43</eq></td><td><eq>93.54 \pm 0.60</eq></td><td><eq>97.56 \pm 0.46</eq></td><td><eq>82.14 \pm 0.91</eq></td><td><eq>96.36 \pm 0.58</eq></td><td><eq>95.10 \pm 0.51</eq></td><td>91.48</td></tr><tr><td>Entropy</td><td><eq>79.29 \pm 0.53</eq></td><td><eq>92.19 \pm 0.39</eq></td><td><eq>96.02 \pm 0.49</eq></td><td><eq>80.50 \pm 0.40</eq></td><td><eq>94.04 \pm 0.88</eq></td><td><eq>91.30 \pm 0.58</eq></td><td>88.89</td></tr><tr><td>Entropy-E</td><td><eq>85.48 \pm 0.50</eq></td><td><eq>95.17 \pm 0.79</eq></td><td><eq>97.45 \pm 1.27</eq></td><td><eq>84.57 \pm 0.93</eq></td><td><eq>96.83 \pm 0.88</eq></td><td><eq>95.42 \pm 0.84</eq></td><td>92.49</td></tr><tr><td>DetectGPT</td><td><eq>75.73 \pm 1.30</eq></td><td><eq>86.86 \pm 0.70</eq></td><td><eq>92.84 \pm 0.66</eq></td><td><eq>72.25 \pm 1.87</eq></td><td><eq>92.84 \pm 1.08</eq></td><td><eq>86.70 \pm 0.65</eq></td><td>84.54</td></tr><tr><td>DetectGPT-E</td><td><eq>70.03 \pm 0.91</eq></td><td><eq>90.38 \pm 1.10</eq></td><td><eq>97.23 \pm 0.41</eq></td><td><eq>78.97 \pm 0.76</eq></td><td><eq>93.82 \pm 1.07</eq></td><td><eq>91.37 \pm 0.42</eq></td><td>86.96</td></tr><tr><td>FastGPT</td><td><eq>82.61 \pm 1.26</eq></td><td><eq>90.99 \pm 0.86</eq></td><td><eq>96.26 \pm 0.29</eq></td><td><eq>81.63 \pm 0.76</eq></td><td><eq>94.38 \pm 0.65</eq></td><td><eq>91.84 \pm 0.57</eq></td><td>89.62</td></tr><tr><td>FastGPT-E</td><td><eq>82.80 \pm 0.75</eq></td><td><eq>94.26 \pm 0.28</eq></td><td><eq>97.68 \pm 0.68</eq></td><td><eq>84.02 \pm 0.99</eq></td><td><eq>96.47 \pm 0.41</eq></td><td><eq>94.84 \pm 1.08</eq></td><td>91.68</td></tr><tr><td>DNAGPT</td><td><eq>77.49 \pm 1.00</eq></td><td><eq>79.64 \pm 1.25</eq></td><td><eq>88.69 \pm 0.51</eq></td><td><eq>67.60 \pm 0.40</eq></td><td><eq>84.15 \pm 1.03</eq></td><td><eq>83.46 \pm 0.91</eq></td><td>80.17</td></tr><tr><td>DNAGPT</td><td><eq>83.92 \pm 0.38</eq></td><td><eq>90.18 \pm 0.76</eq></td><td><eq>92.96 \pm 0.50</eq></td><td><eq>79.81 \pm 1.07</eq></td><td><eq>92.16 \pm 0.66</eq></td><td><eq>90.90 \pm 0.46</eq></td><td>88.32</td></tr></table>


Table 17: Performance on short texts on Essay. The detection models are trained on text generated by GPT4All.


<table><tr><td>Metric</td><td>Method</td><td>GPT4All</td><td>ChatGPT</td><td>Dolly</td><td>ChatGLM</td><td>Claude</td><td>ChatGPT-turbo</td><td>avg</td></tr><tr><td rowspan="12">TPR@FPR-1%</td><td>Likelihood</td><td><eq>19.86 \pm 5.55</eq></td><td><eq>40.93 \pm 4.86</eq></td><td><eq>11.69 \pm 2.93</eq></td><td><eq>58.75 \pm 6.73</eq></td><td><eq>15.38 \pm 1.78</eq></td><td><eq>23.47 \pm 4.56</eq></td><td>28.35</td></tr><tr><td>Likelihood-E</td><td><eq>30.80 \pm 6.67</eq></td><td><eq>45.91 \pm 7.98</eq></td><td><eq>14.99 \pm 4.58</eq></td><td><eq>74.51 \pm 4.38</eq></td><td><eq>18.98 \pm 5.26</eq></td><td><eq>27.24 \pm 6.18</eq></td><td>35.40</td></tr><tr><td>Log-Rank</td><td><eq>15.85 \pm 2.24</eq></td><td><eq>36.49 \pm 6.97</eq></td><td><eq>9.98 \pm 1.81</eq></td><td><eq>55.58 \pm 5.15</eq></td><td><eq>11.33 \pm 3.00</eq></td><td><eq>19.07 \pm 5.24</eq></td><td>24.72</td></tr><tr><td>Log-Rank-E</td><td><eq>33.30 \pm 2.53</eq></td><td><eq>44.18 \pm 4.07</eq></td><td><eq>13.27 \pm 3.77</eq></td><td><eq>72.41 \pm 2.13</eq></td><td><eq>17.60 \pm 1.81</eq></td><td><eq>24.22 \pm 2.48</eq></td><td>34.16</td></tr><tr><td>Entropy</td><td><eq>0.77 \pm 0.31</eq></td><td><eq>4.62 \pm 2.03</eq></td><td><eq>1.43 \pm 1.14</eq></td><td><eq>1.70 \pm 1.13</eq></td><td><eq>3.07 \pm 1.77</eq></td><td><eq>3.47 \pm 1.83</eq></td><td>2.51</td></tr><tr><td>Entropy-E</td><td><eq>1.96 \pm 0.42</eq></td><td><eq>5.11 \pm 1.82</eq></td><td><eq>2.86 \pm 1.17</eq></td><td><eq>3.26 \pm 1.79</eq></td><td><eq>2.89 \pm 1.79</eq></td><td><eq>3.11 \pm 1.63</eq></td><td>3.20</td></tr><tr><td>DetectGPT</td><td><eq>8.25 \pm 2.34</eq></td><td><eq>13.42 \pm 2.70</eq></td><td><eq>7.16 \pm 3.80</eq></td><td><eq>18.57 \pm 4.32</eq></td><td><eq>9.73 \pm 4.84</eq></td><td><eq>8.58 \pm 2.09</eq></td><td>10.95</td></tr><tr><td>DetectGPT-E</td><td><eq>23.92 \pm 7.43</eq></td><td><eq>41.96 \pm 6.95</eq></td><td><eq>15.32 \pm 3.89</eq></td><td><eq>53.44 \pm 7.22</eq></td><td><eq>22.36 \pm 5.76</eq></td><td><eq>30.49 \pm 5.42</eq></td><td>31.25</td></tr><tr><td>FastGPT</td><td><eq>5.28 \pm 2.94</eq></td><td><eq>11.02 \pm 1.41</eq></td><td><eq>3.96 \pm 2.41</eq></td><td><eq>14.20 \pm 6.19</eq></td><td><eq>7.42 \pm 2.08</eq></td><td><eq>6.27 \pm 1.27</eq></td><td>8.03</td></tr><tr><td>FastGPT-E</td><td><eq>13.71 \pm 9.14</eq></td><td><eq>20.89 \pm 1.26</eq></td><td><eq>8.78 \pm 3.14</eq></td><td><eq>27.50 \pm 4.97</eq></td><td><eq>15.73 \pm 1.90</eq></td><td><eq>12.84 \pm 2.35</eq></td><td>16.58</td></tr><tr><td>DNAGPT</td><td><eq>8.02 \pm 1.30</eq></td><td><eq>20.04 \pm 4.42</eq></td><td><eq>10.02 \pm 2.96</eq></td><td><eq>35.98 \pm 5.07</eq></td><td><eq>6.84 \pm 1.77</eq></td><td><eq>11.11 \pm 2.36</eq></td><td>15.34</td></tr><tr><td>DNAGPT</td><td><eq>13.85 \pm 1.23</eq></td><td><eq>23.87 \pm 4.14</eq></td><td><eq>16.09 \pm 5.47</eq></td><td><eq>49.78 \pm 4.61</eq></td><td><eq>8.09 \pm 1.80</eq></td><td><eq>13.38 \pm 2.56</eq></td><td>20.84</td></tr><tr><td rowspan="12">AUROC</td><td>Likelihood</td><td><eq>84.84 \pm 0.81</eq></td><td><eq>93.57 \pm 0.68</eq></td><td><eq>83.34 \pm 1.31</eq></td><td><eq>96.34 \pm 0.44</eq></td><td><eq>81.26 \pm 1.02</eq></td><td><eq>89.30 \pm 0.82</eq></td><td>88.11</td></tr><tr><td>Likelihood-E</td><td><eq>86.95 \pm 0.80</eq></td><td><eq>93.65 \pm 0.76</eq></td><td><eq>84.16 \pm 1.29</eq></td><td><eq>97.04 \pm 0.32</eq></td><td><eq>83.43 \pm 0.74</eq></td><td><eq>89.59 \pm 0.87</eq></td><td>89.14</td></tr><tr><td>Log-Rank</td><td><eq>82.90 \pm 0.94</eq></td><td><eq>92.29 \pm 0.84</eq></td><td><eq>81.18 \pm 1.11</eq></td><td><eq>95.97 \pm 0.52</eq></td><td><eq>77.58 \pm 0.92</eq></td><td><eq>87.83 \pm 0.92</eq></td><td>86.29</td></tr><tr><td>Log-Rank-E</td><td><eq>85.88 \pm 0.83</eq></td><td><eq>92.86 \pm 0.78</eq></td><td><eq>83.27 \pm 1.08</eq></td><td><eq>97.01 \pm 0.40</eq></td><td><eq>81.40 \pm 0.71</eq></td><td><eq>88.28 \pm 0.93</eq></td><td>88.12</td></tr><tr><td>Entropy</td><td><eq>58.86 \pm 1.57</eq></td><td><eq>72.56 \pm 1.03</eq></td><td><eq>62.13 \pm 0.86</eq></td><td><eq>66.71 \pm 1.77</eq></td><td><eq>67.08 \pm 1.13</eq></td><td><eq>73.88 \pm 0.80</eq></td><td>66.87</td></tr><tr><td>Entropy-E</td><td><eq>62.02 \pm 1.48</eq></td><td><eq>74.75 \pm 1.22</eq></td><td><eq>65.11 \pm 1.67</eq></td><td><eq>71.53 \pm 1.72</eq></td><td><eq>70.74 \pm 1.19</eq></td><td><eq>74.98 \pm 1.01</eq></td><td>69.86</td></tr><tr><td>DetectGPT</td><td><eq>78.67 \pm 1.06</eq></td><td><eq>84.93 \pm 0.84</eq></td><td><eq>76.56 \pm 0.97</eq></td><td><eq>89.40 \pm 0.75</eq></td><td><eq>79.27 \pm 0.74</eq></td><td><eq>81.54 \pm 0.96</eq></td><td>81.73</td></tr><tr><td>DetectGPT-E</td><td><eq>86.13 \pm 1.27</eq></td><td><eq>92.12 \pm 0.78</eq></td><td><eq>82.57 \pm 1.32</eq></td><td><eq>96.33 \pm 0.63</eq></td><td><eq>83.40 \pm 0.90</eq></td><td><eq>88.57 \pm 0.75</eq></td><td>88.19</td></tr><tr><td>FastGPT</td><td><eq>79.78 \pm 0.78</eq></td><td><eq>85.87 \pm 0.73</eq></td><td><eq>81.15 \pm 0.62</eq></td><td><eq>91.22 \pm 1.03</eq></td><td><eq>82.62 \pm 0.55</eq></td><td><eq>80.05 \pm 1.18</eq></td><td>83.45</td></tr><tr><td>FastGPT-E</td><td><eq>86.30 \pm 0.72</eq></td><td><eq>90.49 \pm 0.69</eq></td><td><eq>86.03 \pm 0.77</eq></td><td><eq>95.26 \pm 0.78</eq></td><td><eq>88.05 \pm 0.61</eq></td><td><eq>86.65 \pm 0.68</eq></td><td>88.80</td></tr><tr><td>DNAGPT</td><td><eq>79.59 \pm 0.70</eq></td><td><eq>86.99 \pm 1.18</eq></td><td><eq>79.45 \pm 1.14</eq></td><td><eq>90.86 \pm 0.80</eq></td><td><eq>76.84 \pm 1.03</eq></td><td><eq>83.48 \pm 1.08</eq></td><td>82.87</td></tr><tr><td>DNAGPT</td><td><eq>83.46 \pm 0.91</eq></td><td><eq>88.99 \pm 1.03</eq></td><td><eq>83.06 \pm 1.30</eq></td><td><eq>93.40 \pm 0.58</eq></td><td><eq>80.65 \pm 0.95</eq></td><td><eq>85.84 \pm 1.02</eq></td><td>85.90</td></tr></table>


Table 18: Enhancement results on more detectors in the Essay dataset. The detection models are trained on text generated by GPT4All.


<table><tr><td>Method</td><td>GPT4All</td><td>ChatGPT</td><td>Dolly</td><td>ChatGLM</td><td>Claude</td><td>ChatGPT-turbo</td><td>Avg.</td></tr><tr><td>RepreGuard</td><td><eq>94.66 \pm 0.35</eq></td><td><eq>97.35 \pm 0.28</eq></td><td><eq>90.40 \pm 0.83</eq></td><td><eq>91.95 \pm 0.71</eq></td><td><eq>47.51 \pm 1.53</eq></td><td><eq>43.42 \pm 1.34</eq></td><td>77.55</td></tr><tr><td>RepreGuard-M</td><td><eq>95.46 \pm 0.80</eq></td><td><eq>99.55 \pm 0.18</eq></td><td><eq>90.55 \pm 0.70</eq></td><td><eq>99.95 \pm 0.04</eq></td><td><eq>47.18 \pm 1.47</eq></td><td><eq>43.12 \pm 1.28</eq></td><td>79.30</td></tr><tr><td>Binoculars</td><td><eq>96.33 \pm 0.27</eq></td><td><eq>98.59 \pm 0.18</eq></td><td><eq>88.57 \pm 1.56</eq></td><td><eq>99.57 \pm 0.43</eq></td><td><eq>87.21 \pm 0.43</eq></td><td><eq>98.84 \pm 0.18</eq></td><td>94.85</td></tr><tr><td>Binoculars-M</td><td><eq>97.34 \pm 0.13</eq></td><td><eq>98.65 \pm 0.19</eq></td><td><eq>90.19 \pm 1.38</eq></td><td><eq>97.68 \pm 2.43</eq></td><td><eq>86.80 \pm 0.45</eq></td><td><eq>98.82 \pm 0.18</eq></td><td>94.91</td></tr><tr><td>Lastde</td><td><eq>99.54 \pm 0.08</eq></td><td><eq>99.72 \pm 0.08</eq></td><td><eq>97.38 \pm 0.36</eq></td><td><eq>99.98 \pm 0.01</eq></td><td><eq>75.98 \pm 1.22</eq></td><td><eq>91.16 \pm 0.72</eq></td><td>93.96</td></tr><tr><td>Lastde-M</td><td><eq>99.68 \pm 0.07</eq></td><td><eq>99.77 \pm 0.07</eq></td><td><eq>97.87 \pm 0.35</eq></td><td><eq>99.99 \pm 0.01</eq></td><td><eq>76.21 \pm 1.29</eq></td><td><eq>90.96 \pm 0.75</eq></td><td>94.08</td></tr><tr><td>FourierGPT</td><td><eq>91.70 \pm 16.06</eq></td><td><eq>90.18 \pm 19.03</eq></td><td><eq>93.74 \pm 12.07</eq></td><td><eq>98.26 \pm 2.82</eq></td><td><eq>44.90 \pm 6.18</eq></td><td><eq>42.11 \pm 6.15</eq></td><td>76.81</td></tr><tr><td>FourierGPT-M</td><td><eq>97.10 \pm 4.19</eq></td><td><eq>95.27 \pm 6.46</eq></td><td><eq>97.64 \pm 3.19</eq></td><td><eq>99.44 \pm 0.58</eq></td><td><eq>41.14 \pm 6.81</eq></td><td><eq>41.08 \pm 6.90</eq></td><td>78.61</td></tr></table>


Table 19: Enhancement results on more detectors in the Reuters dataset. The detection models are trained on text generated by GPT4All.


<table><tr><td>Method</td><td>GPT4All</td><td>ChatGPT</td><td>Dolly</td><td>ChatGLM</td><td>Claude</td><td>ChatGPT-turbo</td><td>Avg.</td></tr><tr><td>RepreGuard</td><td><eq>97.21 \pm 0.29</eq></td><td><eq>\textbf{98.32}_{\pm 0.30}</eq></td><td><eq>8.24 \pm 0.46</eq></td><td><eq>\textbf{97.26}_{\pm 0.27}</eq></td><td><eq>21.81 \pm 0.62</eq></td><td><eq>44.32 \pm 1.80</eq></td><td>61.19</td></tr><tr><td>RepreGuard-M</td><td><eq>\textbf{98.33}_{\pm 0.35}</eq></td><td><eq>97.87 \pm 0.21</eq></td><td><eq>\textbf{48.12}_{\pm 5.04}</eq></td><td><eq>97.00 \pm 0.29</eq></td><td><eq>\textbf{29.08}_{\pm 1.08}</eq></td><td><eq>\textbf{50.59}_{\pm 1.84}</eq></td><td><eq>\textbf{70.16}</eq></td></tr><tr><td>Binoculars</td><td><eq>80.96_{\pm 1.11}</eq></td><td><eq>97.99_{\pm 0.30}</eq></td><td><eq>61.25_{\pm 1.77}</eq></td><td><eq>99.80_{\pm 0.06}</eq></td><td><eq>83.23_{\pm 0.80}</eq></td><td><eq>98.70_{\pm 0.25}</eq></td><td>86.99</td></tr><tr><td>Binoculars-M</td><td><eq>\textbf{90.90}_{\pm 0.73}</eq></td><td><eq>\textbf{98.49}_{\pm 0.19}</eq></td><td><eq>\textbf{74.10}_{\pm 0.95}</eq></td><td><eq>\textbf{99.86}_{\pm 0.04}</eq></td><td><eq>\textbf{86.37}_{\pm 0.75}</eq></td><td><eq>\textbf{98.72}_{\pm 0.32}</eq></td><td><eq>\textbf{91.41}</eq></td></tr><tr><td>Lastde</td><td><eq>99.21_{\pm 0.18}</eq></td><td><eq>99.67_{\pm 0.04}</eq></td><td><eq>94.25_{\pm 0.63}</eq></td><td><eq>\textbf{99.83}_{\pm 0.07}</eq></td><td><eq>83.27_{\pm 0.49}</eq></td><td><eq>\textbf{99.09}_{\pm 0.18}</eq></td><td>95.89</td></tr><tr><td>Lastde-M</td><td><eq>\textbf{99.55}_{\pm 0.16}</eq></td><td><eq>\textbf{99.68}_{\pm 0.05}</eq></td><td><eq>\textbf{96.09}_{\pm 0.35}</eq></td><td><eq>99.82_{\pm 0.05}</eq></td><td><eq>\textbf{85.64}_{\pm 0.59}</eq></td><td><eq>99.00_{\pm 0.19}</eq></td><td><eq>\textbf{96.63}</eq></td></tr><tr><td>FourierGPT</td><td><eq>99.28_{\pm 0.35}</eq></td><td><eq>96.71_{\pm 3.25}</eq></td><td><eq>99.06_{\pm 0.41}</eq></td><td><eq>98.45_{\pm 0.25}</eq></td><td><eq>\textbf{56.82}_{\pm 10.56}</eq></td><td><eq>49.84_{\pm 0.89}</eq></td><td>83.36</td></tr><tr><td>FourierGPT-M</td><td><eq>\textbf{99.36}_{\pm 0.29}</eq></td><td><eq>\textbf{98.29}_{\pm 0.72}</eq></td><td><eq>\textbf{99.29}_{\pm 0.27}</eq></td><td><eq>\textbf{98.48}_{\pm 0.43}</eq></td><td><eq>55.13_{\pm 10.81}</eq></td><td><eq>\textbf{49.88}_{\pm 1.12}</eq></td><td><eq>\textbf{83.40}</eq></td></tr></table>


Table 20: Enhancement results on more detectors in the DetectRL dataset. The detection models are trained on text generated by GPT4All. Note that DetectRL lacks paired data for RepreGuard; therefore, this method is not compared.


<table><tr><td>Method</td><td>Llama-2-70b</td><td>ChatGPT</td><td>Google-PaLM</td><td>Avg.</td></tr><tr><td>Binoculars</td><td><eq>79.10 \pm 0.37</eq></td><td><eq>66.55 \pm 0.80</eq></td><td><eq>73.85 \pm 0.93</eq></td><td>73.17</td></tr><tr><td>Binoculars-M</td><td><eq>\mathbf{81.22} \pm 0.39</eq></td><td><eq>\mathbf{67.43} \pm 1.08</eq></td><td><eq>\mathbf{77.80} \pm 0.79</eq></td><td><eq>\mathbf{75.49}</eq></td></tr><tr><td>Lastde</td><td><eq>78.36 \pm 0.39</eq></td><td><eq>51.93 \pm 0.51</eq></td><td><eq>69.04 \pm 0.90</eq></td><td>66.45</td></tr><tr><td>Lastde-M</td><td><eq>\mathbf{78.70} \pm 0.36</eq></td><td><eq>\mathbf{52.55} \pm 0.43</eq></td><td><eq>\mathbf{71.06} \pm 0.76</eq></td><td><eq>\mathbf{67.44}</eq></td></tr><tr><td>FourierGPT</td><td><eq>58.11 \pm 7.29</eq></td><td><eq>50.57 \pm 3.63</eq></td><td><eq>59.99 \pm 8.53</eq></td><td>56.23</td></tr><tr><td>FourierGPT-M</td><td><eq>\mathbf{64.67} \pm 8.42</eq></td><td><eq>\mathbf{52.64} \pm 5.70</eq></td><td><eq>\mathbf{64.07} \pm 7.25</eq></td><td><eq>\mathbf{60.46}</eq></td></tr></table>


Table 21: Performance (AUROC) on perturbation texts (Back-Translation and DeepWordBug) on DetectRL. All detectors are trained on Llama-2-70b texts.


<table><tr><td>Perturbation</td><td>Method</td><td>Llama-2-70b</td><td>ChatGPT</td><td>Google-PaLM</td><td>Avg.</td></tr><tr><td rowspan="12">Back-Translation</td><td>Likelihood</td><td><eq>53.88 \pm 0.76</eq></td><td><eq>\textbf{29.91}_{\pm 1.34}</eq></td><td><eq>48.00 \pm 0.58</eq></td><td>43.93</td></tr><tr><td>Likelihood-M</td><td><eq>\textbf{58.85}_{\pm 0.95}</eq></td><td><eq>29.71_{\pm 1.26}</eq></td><td><eq>\textbf{55.80}_{\pm 0.90}</eq></td><td>48.12</td></tr><tr><td>Log-Rank</td><td><eq>56.40_{\pm 0.74}</eq></td><td><eq>30.91_{\pm 1.32}</eq></td><td><eq>46.85_{\pm 0.75}</eq></td><td>44.72</td></tr><tr><td>Log-Rank-M</td><td><eq>\textbf{62.41}_{\pm 0.61}</eq></td><td><eq>\textbf{32.17}_{\pm 1.19}</eq></td><td><eq>\textbf{59.55}_{\pm 0.69}</eq></td><td>51.37</td></tr><tr><td>Entropy</td><td><eq>46.43_{\pm 1.06}</eq></td><td><eq>\textbf{32.54}_{\pm 1.08}</eq></td><td><eq>41.99_{\pm 0.61}</eq></td><td>40.32</td></tr><tr><td>Entropy-M</td><td><eq>\textbf{48.99}_{\pm 1.33}</eq></td><td><eq>31.64_{\pm 1.18}</eq></td><td><eq>\textbf{46.16}_{\pm 0.49}</eq></td><td>42.26</td></tr><tr><td>DetectGPT</td><td><eq>20.30_{\pm 0.72}</eq></td><td><eq>10.90_{\pm 0.68}</eq></td><td><eq>16.59_{\pm 0.56}</eq></td><td>15.93</td></tr><tr><td>DetectGPT-M</td><td><eq>\textbf{62.24}_{\pm 2.17}</eq></td><td><eq>\textbf{38.63}_{\pm 6.75}</eq></td><td><eq>\textbf{61.72}_{\pm 4.74}</eq></td><td>54.20</td></tr><tr><td>FastGPT</td><td><eq>35.37_{\pm 1.40}</eq></td><td><eq>\textbf{24.10}_{\pm 1.08}</eq></td><td><eq>31.23_{\pm 0.57}</eq></td><td>30.23</td></tr><tr><td>FastGPT-M</td><td><eq>\textbf{51.87}_{\pm 3.25}</eq></td><td><eq>21.35_{\pm 3.21}</eq></td><td><eq>\textbf{53.16}_{\pm 3.04}</eq></td><td>42.13</td></tr><tr><td>DNA-GPT</td><td><eq>70.82_{\pm 0.84}</eq></td><td><eq>\textbf{55.34}_{\pm 0.73}</eq></td><td><eq>65.37_{\pm 0.51}</eq></td><td>63.84</td></tr><tr><td>DNA-GPT-M</td><td><eq>\textbf{71.83}_{\pm 0.94}</eq></td><td><eq>54.53_{\pm 1.32}</eq></td><td><eq>\textbf{67.63}_{\pm 0.62}</eq></td><td>64.67</td></tr><tr><td rowspan="12">DeepWordBug</td><td>Likelihood</td><td><eq>69.62_{\pm 0.71}</eq></td><td><eq>51.56_{\pm 0.73}</eq></td><td><eq>62.19_{\pm 1.42}</eq></td><td>61.12</td></tr><tr><td>Likelihood-M</td><td><eq>\textbf{85.71}_{\pm 1.22}</eq></td><td><eq>\textbf{68.11}_{\pm 1.64}</eq></td><td><eq>\textbf{79.18}_{\pm 0.63}</eq></td><td>77.67</td></tr><tr><td>Log-Rank</td><td><eq>72.76_{\pm 0.64}</eq></td><td><eq>50.60_{\pm 0.44}</eq></td><td><eq>62.64_{\pm 1.86}</eq></td><td>62</td></tr><tr><td>Log-Rank-M</td><td><eq>\textbf{90.93}_{\pm 1.14}</eq></td><td><eq>\textbf{71.16}_{\pm 2.75}</eq></td><td><eq>\textbf{83.41}_{\pm 1.12}</eq></td><td>81.84</td></tr><tr><td>Entropy</td><td><eq>65.85_{\pm 0.97}</eq></td><td><eq>63.63_{\pm 1.71}</eq></td><td><eq>58.00_{\pm 1.31}</eq></td><td>62.49</td></tr><tr><td>Entropy-M</td><td><eq>\textbf{69.35}_{\pm 0.95}</eq></td><td><eq>\textbf{66.54}_{\pm 1.58}</eq></td><td><eq>\textbf{63.95}_{\pm 1.25}</eq></td><td>66.61</td></tr><tr><td>DetectGPT</td><td><eq>34.33_{\pm 2.82}</eq></td><td><eq>33.28_{\pm 3.72}</eq></td><td><eq>32.72_{\pm 2.14}</eq></td><td>33.44</td></tr><tr><td>DetectGPT-M</td><td><eq>\textbf{82.95}_{\pm 1.70}</eq></td><td><eq>\textbf{64.13}_{\pm 2.85}</eq></td><td><eq>\textbf{79.67}_{\pm 2.41}</eq></td><td>75.58</td></tr><tr><td>FastGPT</td><td><eq>39.57_{\pm 2.14}</eq></td><td><eq>\textbf{34.59}_{\pm 2.49}</eq></td><td><eq>36.51_{\pm 2.30}</eq></td><td>36.89</td></tr><tr><td>FastGPT-M</td><td><eq>\textbf{53.71}_{\pm 1.90}</eq></td><td><eq>30.93_{\pm 2.38}</eq></td><td><eq>\textbf{54.63}_{\pm 2.88}</eq></td><td>46.42</td></tr><tr><td>DNA-GPT</td><td><eq>67.76_{\pm 2.07}</eq></td><td><eq>58.74_{\pm 1.64}</eq></td><td><eq>63.69_{\pm 1.06}</eq></td><td>63.4</td></tr><tr><td>DNA-GPT-M</td><td><eq>\textbf{71.43}_{\pm 6.30}</eq></td><td><eq>\textbf{59.59}_{\pm 4.16}</eq></td><td><eq>\textbf{67.61}_{\pm 4.09}</eq></td><td>66.21</td></tr></table>


Table 22: Performance and running time comparison with HMM-based method on Essay. The detection models are trained on text generated by GPT4All.


<table><tr><td rowspan="2">Method</td><td colspan="7">Detection Performance</td><td colspan="2">Running Time</td></tr><tr><td>GPT4All</td><td>ChatGPT</td><td>Dolly</td><td>ChatGLM</td><td>Claude</td><td>ChatGPT-turbo</td><td>Avg.</td><td>Train</td><td>Inference</td></tr><tr><td>Likelihood</td><td>96.16</td><td>98.79</td><td>90.90</td><td>99.29</td><td>92.76</td><td>99.13</td><td>96.17</td><td>13.58</td><td>61.21</td></tr><tr><td>Likelihood-HMM</td><td>96.39</td><td>98.87</td><td>91.24</td><td>99.33</td><td>92.91</td><td>99.17</td><td>96.32</td><td>20.64</td><td>90.63</td></tr><tr><td>Likelihood-M</td><td>98.58</td><td>99.47</td><td>94.59</td><td>99.54</td><td>94.82</td><td>99.72</td><td>97.79</td><td>15.01</td><td>72.67</td></tr><tr><td>Log-Rank</td><td>96.55</td><td>98.95</td><td>90.08</td><td>99.36</td><td>92.01</td><td>99.24</td><td>96.03</td><td>15.98</td><td>75.60</td></tr><tr><td>Log-Rank-HMM</td><td>96.68</td><td>98.99</td><td>90.24</td><td>99.38</td><td>92.08</td><td>99.27</td><td>96.10</td><td>21.77</td><td>103.70</td></tr><tr><td>Log-Rank-M</td><td>98.57</td><td>99.41</td><td>93.82</td><td>99.55</td><td>92.91</td><td>99.64</td><td>97.32</td><td>17.40</td><td>87.55</td></tr><tr><td>Entropy</td><td>74.19</td><td>89.49</td><td>73.26</td><td>84.11</td><td>86.58</td><td>95.94</td><td>83.93</td><td>13.26</td><td>64.49</td></tr><tr><td>Entropy-HMM</td><td>74.47</td><td>89.70</td><td>73.56</td><td>84.39</td><td>86.61</td><td>95.98</td><td>84.12</td><td>16.87</td><td>88.26</td></tr><tr><td>Entropy-M</td><td>83.52</td><td>93.28</td><td>81.11</td><td>91.44</td><td>87.96</td><td>96.75</td><td>89.01</td><td>14.69</td><td>73.66</td></tr></table>


Table 23: Performance (AUROC) on TOCSIN-enhanced baselines and our method on top of TOCSIN (suffix “-TM”). Detectors are trained on GPT4All texts.


<table><tr><td>Method</td><td>GPT4</td><td>ChatGPT-turbo</td><td>ChatGLM</td><td>Dolly</td><td>ChatGPT</td><td>StableLM</td><td>Avg.</td></tr><tr><td>Likelihood-T</td><td><eq>90.43 \pm 1.03</eq></td><td><eq>98.66 \pm 0.19</eq></td><td><eq>89.50 \pm 0.79</eq></td><td><eq>98.52 \pm 0.55</eq></td><td><eq>90.46 \pm 0.71</eq></td><td><eq>98.44 \pm 0.31</eq></td><td>94.33</td></tr><tr><td>Likelihood-TM</td><td><eq>98.08 \pm 0.19</eq></td><td><eq>99.19 \pm 0.15</eq></td><td><eq>92.83 \pm 1.06</eq></td><td><eq>99.01 \pm 0.51</eq></td><td><eq>91.06 \pm 1.26</eq></td><td><eq>99.20 \pm 0.09</eq></td><td>96.56</td></tr><tr><td>Log-Rank-T</td><td><eq>93.96 \pm 0.46</eq></td><td><eq>98.78 \pm 0.17</eq></td><td><eq>89.90 \pm 0.83</eq></td><td><eq>98.69 \pm 0.52</eq></td><td><eq>91.01 \pm 0.48</eq></td><td><eq>98.57 \pm 0.27</eq></td><td>95.15</td></tr><tr><td>Log-Rank-TM</td><td><eq>98.07 \pm 0.35</eq></td><td><eq>99.10 \pm 0.16</eq></td><td><eq>92.32 \pm 0.91</eq></td><td><eq>99.01 \pm 0.43</eq></td><td><eq>91.46 \pm 0.62</eq></td><td><eq>99.05 \pm 0.13</eq></td><td>96.50</td></tr><tr><td>Entropy-T</td><td><eq>76.90 \pm 1.57</eq></td><td><eq>98.41 \pm 0.25</eq></td><td><eq>87.20 \pm 0.92</eq></td><td><eq>98.11 \pm 0.61</eq></td><td><eq>90.91 \pm 0.40</eq></td><td><eq>98.72 \pm 0.19</eq></td><td>91.71</td></tr><tr><td>Entropy-TM</td><td><eq>83.89 \pm 0.82</eq></td><td><eq>98.34 \pm 0.35</eq></td><td><eq>87.23 \pm 0.96</eq></td><td><eq>98.25 \pm 0.53</eq></td><td><eq>89.70 \pm 0.44</eq></td><td><eq>98.63 \pm 0.19</eq></td><td>92.67</td></tr><tr><td>DetectGPT-T</td><td><eq>48.95 \pm 1.79</eq></td><td><eq>40.69 \pm 11.74</eq></td><td><eq>51.46 \pm 2.98</eq></td><td><eq>40.67 \pm 13.55</eq></td><td><eq>43.99 \pm 7.23</eq></td><td><eq>27.04 \pm 30.41</eq></td><td>42.13</td></tr><tr><td>DetectGPT-TM</td><td><eq>95.51 \pm 2.81</eq></td><td><eq>93.27 \pm 9.08</eq></td><td><eq>79.76 \pm 14.11</eq></td><td><eq>97.67 \pm 2.00</eq></td><td><eq>70.25 \pm 11.65</eq></td><td><eq>93.59 \pm 10.94</eq></td><td>88.34</td></tr><tr><td>FastGPT-T</td><td><eq>53.53 \pm 5.59</eq></td><td><eq>70.36 \pm 27.49</eq></td><td><eq>56.26 \pm 8.54</eq></td><td><eq>71.83 \pm 28.44</eq></td><td><eq>72.06 \pm 29.44</eq></td><td><eq>77.26 \pm 36.41</eq></td><td>66.88</td></tr><tr><td>FastGPT-TM</td><td><eq>87.73 \pm 0.15</eq></td><td><eq>66.40 \pm 1.22</eq></td><td><eq>80.65 \pm 0.81</eq></td><td><eq>75.36 \pm 1.49</eq></td><td><eq>41.91 \pm 1.52</eq></td><td><eq>28.10 \pm 2.13</eq></td><td>63.36</td></tr><tr><td>DNAGPT-T</td><td><eq>98.32 \pm 0.21</eq></td><td><eq>97.97 \pm 0.19</eq></td><td><eq>96.60 \pm 0.27</eq></td><td><eq>98.34 \pm 0.17</eq></td><td><eq>95.13 \pm 0.46</eq></td><td><eq>97.93 \pm 0.18</eq></td><td>97.38</td></tr><tr><td>DNAGPT-TM</td><td><eq>99.56 \pm 0.10</eq></td><td><eq>99.04 \pm 0.18</eq></td><td><eq>97.70 \pm 0.33</eq></td><td><eq>99.24 \pm 0.11</eq></td><td><eq>96.30 \pm 0.40</eq></td><td><eq>99.06 \pm 0.19</eq></td><td>98.48</td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/d922dab74b44af194924449a0bb097193aa03ecc1f4c15a0055a0c1c77fef2b4.jpg)



Entropy


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/cbe9357694a2b3ecec47228453d81057ed745535ffeac0326bb567e1bc9429c7.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/93047f976bb6ebb112ef2c71ca42adb9461281338834316a3d7ea6e40fa904db.jpg)



DetectGPT


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/01d369b738609820506dd6464ba81e5409b5f5039e918011976b0db071c176f6.jpg)



Figure 21: The performance improvement of the proposed method on Entropy and DetectGPT. Values greater than 0 indicate an enhanced effect.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/5e247e34f2c83c57a5aecd17f4a6b786828745ccf87db55ff6d8b1829274b164.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/85766feacaf89ba7fc4c5f05b4fea727b69b695361ec7c3220616979cad962db.jpg)



FastGPT


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/952bc97f7e31ef53dc8fc50a2841459632b6c5fbf9ee6a0fa8e95f8579f9ea31.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/a8fcbca4adfdb7e3a05e55202302b78c08e44a1e1150be98e1f2ed0982724581.jpg)



DNA-GPT



Figure 22: The performance improvement of the proposed method on FastGPT and DNA-GPT. Values greater than 0 indicate an enhanced effect.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/fff8f04044af394a601b1b0671c2774f148632e770fa7f431836d4496c28819c.jpg)



Test text: Llama-2-70b


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/152277bf134dd91a437fde7435023b087c46eb51f0b35217894f9f6892c170f8.jpg)



Test text: ChatGPT


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/65a0f1452888923c3e7eae05a7b78b45eb8e457ede3492732fc35ecbdcba7bea.jpg)



Test text: Google-PaLM



Figure 23: Detection performance concerning TPR@FPR-1% under different LLM mixed texts. All detectors are trained on Llama-2-70b texts.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/9354a607da018ec80a075afaf5dd58e5d6a8c29f0a51dda83aa5653d1b38d3e8.jpg)



Test text: Llama-2-70b


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/cdc2a930b223190422562a864998eda947fc120a2750aa790f25697043cff73e.jpg)



Test text: ChatGPT


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/4c6e12bb2ac5cf3da847e83dd2d4efb465b0cfded81dc59133b1bda7627238fc.jpg)



Figure 24: Detection performance concerning TPR@FPR-1% under different paraphrasing texts (Dipper and Polish). All detectors are trained on Llama-2-70b texts.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/7a062ecd91e24ee9458936dfcaa7c6ecd00e9977b6dfb82b8c0f4b8da90af0e1.jpg)



Likelihood


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/6a0a04e300ae8fdd9f831d94deb4cf350f55fde2dd252d86bc59b22a3137b0cc.jpg)



Log-Rank


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/a157854373fffaa226f20950a2c4c8d25d819956e4f9f8b25e587eabf393d599.jpg)



Entropy


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/098e81869171196895101b4e786deb56e77d7feb92a2b671c4e8824261fb70f0.jpg)



DetectGPT


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/cb1a6fddd43a70bacbd5c738d49f4197cb22b04afeeb6d73d78ee3e4ff6ea8b6.jpg)



FastGPT


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/6dc8ee970d367bcd81f6ef55796b29b7ef8c2eaa305d62ae472f930e3a6a538d.jpg)



DNA-GPT



Figure 25: Ablation results on the DetectRL dataset. The y-axis represents the LLM text on which the detector was trained, and the x-axis represents the average performance across LLMs.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/b18a628361fa7ef9c4da6fe638dc79d979ae5d17d03599d40ddabe970c605367.jpg)



Likelihood


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/1766946b59be3bd2bfe47448f7ae74938c43618bb60c083d9a9a990257a0cd0f.jpg)



Log-Rank


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/b5c43550518b72bbd38a6ed16591f16b45f7e7e26b78b853b2b5b846b80ed3fa.jpg)



Entropy


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/104051bda4ede95042d10f0fbc4f64bb7e7720d27e73832a383505d46ee8e4f3.jpg)



DetectGPT


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/390b42ea2ab71ccf7c117f7b451cd3709e3dea54bd194334022c1522d5ba45ff.jpg)



FastGPT


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/c8fec16ddb3c3bee9e7e2caf1f4983c7c18da9cd5adb10544331096d893c7e06.jpg)



DNA-GPT



Figure 26: Ablation results on the Reuters dataset. The y-axis represents the LLM text on which the detector was trained, and the x-axis represents the average performance across LLMs.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/17da0adab3c3ffcefa98a47f6f35fabcb60bdd3f2156ae049b741589099dc57f.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/5faaa5d4a22a944de840b937772a105ddf435e211aaf18d14f48d0e4ab30be85.jpg)



Essay


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/6b76f8c18adb16e253aac145ee21d21c35f8b646b0f82014cd36015aeaf38f58.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/e9a638f91ac00cfbc377714cdc2aa50a65291ca0ca0240a93f05053b403c68fc.jpg)



DetectRL



Figure 27: Detection performance of AUROC under different transition center $t_{0}$ .


t 

t 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/73fbf1b97404748800ae5ede1df37e2069b6cdf329a28e1b50f1f9cffb4f6275.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/4b0ec5ba8f605ace435ba3049d8fb8f9c10e4b1a88ba2e129ec95cc31406038b.jpg)



Essay


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/5d85fe0919d6dd8b4f06156c19d0d894df2aaaf844f2f96582278b61ec64a9e2.jpg)



DetectRL


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/989e89872c7efc5da5e69b2cff7ed1ab43da1b3e39e469d6efa59d51941a6dc7.jpg)



Figure 28: Detection performance of TPR@FPR-1% under different transition center $t_{0}$ .


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/7063c1e26f6b32431b2345f43930296d5df5341ce4145d985a8d4bc001910959.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/7c269476bd596afa19505cc901dc2e461757e1100f3543ff69efb2df5b0411f8.jpg)



Essay


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/d35dd882a39dc35eacb2c452b2ecc6c8d38e56f5e177e23b39fd37833b9d01b7.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/b078f436b392a6b09b6fecd55cb6a21a836c6f6ff86da3744ee42eabad375dec.jpg)



DetectRL



Figure 29: Detection performance of AUROC at different numbers of MRF layer iterations.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/9ea7e697e4c1ddef972ed6b647428b2bcd4cf65d27a8321db56d878a4fe40c1d.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/72b6d1ddc512e453e2679c35f1ed6377afa22a78aa6b886674c76e6413115ad8.jpg)



Essay


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/92a0b77b0a964b6958b6b28d4edca5b39dda3061620da7709e6e60e82813fb80.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/da062ce9c0502bf1aa3dff32b2ad218d6769034e851bdf7cf773227af935d944.jpg)



DetectRL



Figure 30: Detection performance of TPR@FPR-1% at different numbers of MRF layer iterations.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/0910c7c9ef860be58d086bb08796500bcb12cc60059f9a3234b231d0e74092b3.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/c08615ab31860ef3697d9e5f344f463a13665c4a354b49e96f8445a140923a37.jpg)



Essay Dataset


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/4e769534464512f5d3c8ec8e427c8b0d74cc6ba550afd671c87c079857bc367c.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/d2166961b4695b7f705274361effc1112ba3f8ee91927a6738d5d8ef5b0639fa.jpg)



DetectRL Dataset



Figure 31: Comparison with NN-based methods. The detector used is Log-Rank.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/7298f7903bad7028ac1158454dd1038a9d772a22e6315812c890bb4c27050e56.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/b049def35302b7fa3b8810d9d67d9151048ce29f97a99fc3952405676ddfb837.jpg)



Essay Dataset


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/46a7712c935c76f0a5d7cefba4317e66cd36a9b92ed8adb9c07087c4ef1f46de.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/7154632984722bb95bb47246da76180b46d4162a325b1464b6d87a0392c5ac5d.jpg)



DetectRL Dataset



Figure 32: Comparison with NN-based methods. The detector used is Entropy.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/b7792a5a39f6305e5b6e8dc3256b533cba43abc1150ab32d350e13c961ee7662.jpg)



(a) arxiv


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/6406d20331716568879102c456e3405f0db468da2ce986752535684c30bbccfe.jpg)



(b) writing


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/060334fd730a1a516cdd249eadf2e484643e64850cb25083f71e3e3b790808f2.jpg)



(c) xsum


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/6b54503a671c78aec7a8dba87c713833747c9a1b6029c089bfaf4736f8d1a7c9.jpg)



(d) yelp



Figure 33: Statistical information on the distance of neighbor detection scores.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/02dda4396d697a50f4bd6992f819fb27f7183dde706f88f963ec1d95d4d6c138.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/befeddeadc53fbe1ba962b472ccd49bb18c4f322bc7731e4b2403411c765badc.jpg)



Essay Dataset


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/640242647f8954a7139d40e1fba230bcb9ee713033e43e645d66500f1af26590.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/98d0fab306c84be7b3d0e14cfe9b33b1f52d2a663bdf561021409225fbaaef70.jpg)



Reuters Dataset


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/a4ddada3f5d9aa93ffa773f45f878f3c826ecd5a7d0a751b81941799a90580ea.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/9678f3f5-8a04-49c2-95ec-98b47b770508/eff79b3105366487fb48197e232077aadf01d99c3acb217d2628ef6240aefab7.jpg)



DetectRL Dataset



Figure 34: Comparison with Model-based detectors. Detectors are trained on GPT4All (Essay and Reuters) and Llama-2-70b (DetectRL).



Table 24: Running time (s) of training and inference phases.


<table><tr><td rowspan="2">Method</td><td colspan="3">Train</td><td colspan="3">Inference</td></tr><tr><td>Essay</td><td>Reuters</td><td>DetectRL</td><td>Essay</td><td>Reuters</td><td>DetectRL</td></tr><tr><td>Likelihood</td><td>13.58</td><td>12.77</td><td>12.40</td><td>61.21</td><td>61.05</td><td>59.34</td></tr><tr><td>Likelihood-M</td><td>15.01</td><td>14.19</td><td>13.10</td><td>72.67</td><td>72.38</td><td>61.66</td></tr><tr><td>Log-Rank</td><td>15.98</td><td>14.38</td><td>14.61</td><td>75.60</td><td>72.99</td><td>66.20</td></tr><tr><td>Log-Rank-M</td><td>17.40</td><td>15.81</td><td>16.29</td><td>87.55</td><td>72.06</td><td>82.21</td></tr><tr><td>Entropy</td><td>13.26</td><td>12.93</td><td>13.79</td><td>64.49</td><td>63.69</td><td>63.19</td></tr><tr><td>Entropy-M</td><td>14.69</td><td>14.38</td><td>15.61</td><td>73.66</td><td>72.80</td><td>72.63</td></tr><tr><td>DetectGPT</td><td>204.32</td><td>216.36</td><td>370.44</td><td>924.63</td><td>982.55</td><td>1672.85</td></tr><tr><td>DetectGPT-M</td><td>206.44</td><td>218.73</td><td>373.48</td><td>929.46</td><td>984.70</td><td>1682.78</td></tr><tr><td>FastGPT</td><td>60.21</td><td>58.67</td><td>52.97</td><td>279.50</td><td>272.18</td><td>240.20</td></tr><tr><td>FastGPT-M</td><td>63.00</td><td>61.47</td><td>55.76</td><td>286.44</td><td>284.75</td><td>254.85</td></tr><tr><td>DNA-GPT</td><td>469.10</td><td>532.02</td><td>267.84</td><td>2113.65</td><td>2398.61</td><td>1207.18</td></tr><tr><td>DNA-GPT-M</td><td>471.42</td><td>534.47</td><td>271.11</td><td>2125.17</td><td>2405.37</td><td>1221.74</td></tr></table>