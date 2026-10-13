# BRINGING STABILITY TO DIFFUSION: DECOMPOSING AND REDUCING VARIANCE OF TRAIN-ING MASKED DIFFUSION MODELS

Mengni Jia<sup>1</sup>, Mengyu Zhou<sup>2</sup>, Yihao Liu<sup>3</sup>, Xiaoxi Jiang<sup>2</sup>, Guanjun Jiang<sup>2</sup> 

<sup>1</sup>University of Cambridge, <sup>2</sup>Qwen Large Model Application Team, Alibaba, <sup>3</sup>Peking University 

## ABSTRACT

Masked diffusion models (MDMs) are a promising alternative to autoregressive models (ARMs), but they suffer from inherently much higher training variance. High variance leads to noisier gradient estimates and unstable optimization, so even pretrained MDMs and ARMs are competitive at initialization, they often diverge after task-specific training, with MDMs falling far behind. Currently, there has been no theoretical explanation or systematic solution. In this paper, we derive the first decomposition of MDM training variance into three sources: A masking pattern noise, B masking rate noise, and C data noise – while ARMs are only affected by C . This cleanly explains the fundamental training gap. Building on this foundation, we design six variance-reduction methods, including two core methods: (1) P-POTS, a Pareto-optimal t-sampler that minimizes training variance by sampling harder t values more often with appropriately smaller update steps, and (2) MIRROR, which uses negatively correlated samples to reduce A . Experiments show that, compared to standard MDM training, our methods improve accuracy by 7–8% on complex reasoning tasks, while simultaneously reducing run-to-run variability to near ARM levels, substantially narrowing the gap with strong ARM baselines; in most settings, even the best baseline method runs remain below the worst run of our method. The code will be released at https://github.com/Qwen-Applications/StableDLLM. 

## 1 INTRODUCTION

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/f0eb91ff-cd64-48a5-ae07-430a6a4f0bcb/1523655a796fbdeb6a9a46408d388b16e57c0870bdcb89933bef622ffbfd60d9.jpg)



Figure 1: Graphic illustration of three sources of training variance in MDMs. The left panel illustrates standard MDM training, compared to our core methods (P-POTS and MIRROR) on the right.


Recently, masked diffusion models (MDMs) (Austin et al., 2021; Lou et al., 2023; Sahoo et al., 2024) have been viewed as a strong alternative architecture for both large language models (Dream-7B (Ye et al., 2025), LLaDA-8B (Nie et al., 2025), DiffuCoder (Gong et al., 2025)) and multi-modal models (Dimple-7B (Yu et al., 2025), MMaDA-8B (Yang et al., 2025)). They operate by masking input tokens at a randomly sampled masking rate t and learning to reconstruct them. By moving beyond sequential token modeling, they potentially address limitations of auto-regressive models (ARMs), such as lack of parallelism, exposure bias, and reversal curse (Berglund et al., 2023). 

However, the noise-based training paradigm that powers MDMs also makes their training unstable. Numerous studies have reported that diffusion training suffers from high variance (Meng et al., 2021; Xu et al., 2023; Jeha et al., 2024; Kim et al., 2024; Zhu et al., 2025; Arriola et al., 2025). We refer to this phenomenon as the training divergence from ARMs, which manifests in two aspects: 

• Across-run variability. Even under identical training setups, MDMs can converge to very different solutions: since parameter updates scale with loss gradients, fluctuations in losses translate directly into erratic updates. By contrast, ARMs are more predictable: with the same data order, they converge reliably to the same solution across runs. 

• Within-run sub-optimality. Although pretrained MDMs and ARMs show comparable abilities at initialization, we consistently observe a sharp divergence after post-training: once fine-tuned on the same dataset, ARMs always outperform MDMs by wide margins, even in cases where MDMs begin from a stronger point. This highlights that, even within a single run, the variance gap makes MDMs use supervision resources less efficiently, as noisier updates slow convergence. 

To address this problem, several prior works have proposed mitigation strategies. However, these efforts remain isolated and ad-hoc, lacking a unified theoretical framework for explaining the high training variance of MDMs. For example, Zhu et al. (2025) evaluates both the policy and reference ELBOs on the same time-steps and masking patterns so that their Monte Carlo errors cancel out at comparison, but their method is limited to preference optimization. Arriola et al. (2025) proposes adaptively re-fitting masking-rates samplers at each evaluation, selecting from a fixed number of candidate clipped intervals $\{ \left[ \beta _ { j } , \omega _ { j } \right] \} _ { j = 1 } ^ { b } \subset [ 0 , 1 ]$ to determine the best interval for training until the next evaluation. While this strategy can adjust the learning process accordingly, its reliance on such coarse candidate intervals remains heuristic. These limitations call for more principled solutions. 

Motivated by this, we revisit the basic definitions to develop a systematic explanation and theoretically grounded solutions. Specifically, Sahoo et al. (2024) derived a principled training objective: 

$$
\mathcal {L} \mathrm{MDM} = \mathbb {E} _ {x \sim q (x),, t \sim U [ 0, 1 ],, x _ {t} \sim q (\cdot | x, t)} \big [ l _ {\theta} (x, t, x _ {t}) \big ], \quad l _ {\theta} = \frac {\alpha_ {t} ^ {\prime}}{1 - \alpha_ {t}} \sum_ {\ell = 1} ^ {L} \log \left\langle x _ {\theta} ^ {\ell} (x _ {t} ^ {1: L}, t), x ^ {\ell} \right\rangle\tag{1}
$$

for interpolating MDMs, where the loss $l _ { \theta }$ is a function of the clean data $x ,$ the masking rate $t ,$ and the noised (masked) input $x _ { t }$ obtained from the forward process $q ( x _ { t } \mid x , t )$ . To understand the variance origins in this training objective, we derive the first systematic variance decomposition (proof in A.5.1). Mathematically, this yields: 

$$
\operatorname{Var} _ {x _ {0}, t, x _ {t}} (l _ {\theta}) = \underbrace {\mathbb {E} _ {x _ {0} , t} \left[ \operatorname{Var} _ {x _ {t}} (l _ {\theta} \mid x _ {0} , t) \right]} _ {\text { Masking   Pattern   Noise } ⓐ} + \underbrace {\mathbb {E} _ {x _ {0}} \left[ \operatorname{Var} _ {t} (g _ {\theta} (x _ {0} , t) \mid x _ {0}) \right]} _ {\text { Masking   Rate   Noise } ⓑ} + \underbrace {\operatorname{Var} _ {x _ {0}} \left(\mathbb {E} _ {t} [ g _ {\theta} (x _ {0} , t) ]\right)} _ {\text { Data   Noise } ⓒ}\tag{2}
$$

with $g _ { \theta } ( x _ { 0 } , t ) = \mathbb { E } _ { x _ { t } } [ l _ { \theta } ~ \vert ~ x _ { 0 } , t ] .$ . This decomposition relies on no hard assumptions and emerges naturally, so it both explains existing methods and lays a foundation for understanding how they may complement one another. As shown in Figure 1, the MDM training variance is decomposed into three factors: $\textcircled{6}$ Masking Pattern Noise (randomness from $x _ { t }$ even if $x _ { 0 }$ and t are fixed), B Masking Rate Noise (variability in the expected loss $g ( x _ { 0 } , t )$ across different t for the same $x _ { 0 } )$ , and $©$ Data Noise (variability across $x _ { 0 } ,$ , as some are inherently easier or harder to predict). In contrast, ARMs are subject to only data noise $@$ Therefore, the high training variance of MDMs is explained by the extra sources $\textcircled{8}$ and B , highlighting the need to reduce them to stabilize MDM training. 

To address this challenge, we introduce six techniques, including two core methods (P-POTS and MIRROR in §3.1) and four additional ones. Specifically, P-POTS is the Pareto optimal t-sampler among all unbiased ones in minimizing $\textcircled{8} , \textcircled{8 }$ and $© :$ it focuses training on high-variance tregions while down-weighting their updates to prevent destabilizing optimization. MIRROR, on the other hand, constructs two complementary noised inputs and employs their negative correlation to explicitly reduce $\textcircled{4} .$ Crucially, these two methods exhibit strong synergy, leading us to recommend P-POTS alone for cost-efficiency and P-POTS+MIRROR for maximal performance. 

Experiments validate our claims. On GSM8K, the accuracy of standard MDM training ranges from 50.6% to 53.7%, while P-POTS+MIRROR improves to 58.6%–62.0%. On HiTab, the accuracy range rises from 52.9%–62.6% to 66.0%–68.6%. On these textual datasets, our method allows MDMs to match or even surpass ARMs, with run-to-run variability reduced to near-ARM levels. On text-to-image-2M, the CLIP score range narrows from 28.61-34.28 to 34.10-35.27. 

Our major contributions can be summarized as follows: 

• We derive the first systematic variance decomposition for MDM training as a foundational framework (Eq. 2) and link prior w orks to relevant variance factors they actually address (§2). 

• We introduce six variance-reduction methods, with P-POTS Pareto optimal and P-POTS+MIR-ROR giving the best results. Each method is theoretically grounded with proofs or analysis, simple to implement (requiring little or no hyperparameter tuning) and robust across datasets. 

• We validate our methods on both textual and multi-modal data across various masked diffusion models, demonstrating their effectiveness and generality in stabilizing MDM training. 

## 2 RELATED WORK

The inherently high training variance in diffusion models has been empirically observed. Several remedies have been proposed, yet most of them target continuous diffusion models that are widely used in image generation, and relatively few of them directly apply to masked diffusion models, the model architecture behind dLLMs (diffusion LLMs). Moreover, due to the lack of a theoretical decomposition of the variance sources, the methods remain largely isolated and heuristic. Concretely: 

• Meng et al. (2021) proposes using denoising models trained at antithetic levels of Gaussian noise, which reduces mask pattern noise $\textcircled{4} .$ but is limited to continuous diffusion models. 

• Xu et al. (2023) computes the denoising score as a self-normalized importance-sampled weighted average over a reference batch. It reduces $\textcircled{8}$ and B , but introduces systematic bias with finite reference batches and suffer weight degeneracy in high-dimensional or poorly covered posteriors. 

• Jeha et al. (2024) uses a kth order Taylor expansion of the denoising score as a control variate and subtracting it (with an optimally fitted coefficient) from the Monte Carlo estimate. However, it depends on closed-form Gaussian moment calculations that don’t generalize to non-Gaussian noise $( \mathrm { e . g . }$ , masked diffusion models). Moreover, the method’s reliance on low-order Taylor approximations leads to ineffective variance reduction under high noise or rapidly varying derivatives. 

• Kim et al. (2024) dynamically adjusts the probability of sampling each diffusion timestep based on online estimates of gradient variance. This reduces mask ratio noise B , but does not apply importance reweighting, which biases their loss estimates; moreover, the sampler is restricted to a simple $\mathrm { B e t a } ( a , b )$ , which may fail to capture the true optimal sampling distribution. 

• Zhu et al. (2025) formalizes the variance of ELBO-based gradient estimates in masked diffusion language models and employs optimal Monte Carlo budget allocation and antithetic sampling, reducing the mask pattern noise $\textcircled{8}$ , but is limited in preference optimization. 

• Arriola et al. (2025) select a t-sampling interval by evaluating candidate sub-intervals and choosing the one that minimizes evaluation variance, which reduces $\textcircled{4} + \textcircled { B } + \textcircled { C }$ . While they justify clipping away extreme ts (e.g., to [0.45, 0.95]), our analysis suggests extremes can be worth emphasizing, so their gains likely come from clipping that sometimes approximates the optimal $p ( t )$ 

## 3 METHOD

Let’s recap the standard training algorithm for masked diffusion models (MDMs). Let $x _ { 0 }$ denote the concatenation of prompt and response tokens and $P$ denote the number of response tokens, then: 

1. Sample clean data $x _ { 0 }$ from the data distribution. 

2. Sample a masking rate $t \sim \mathcal { U } [ 0 , 1 ]$ 

3. For each eligible token i, sample $U _ { i } \sim \mathcal { U } [ 0 , 1 ]$ , and construct $x _ { t }$ by replacing $x _ { 0 } ( i )$ with [MASK] if $U _ { i } < t ,$ , so each eligible token is independently masked with probability t. 

4. Compute the sample loss $\begin{array} { r } { l _ { \theta } ( x _ { 0 } , t , x _ { t } ) ~ = ~ - \frac { 1 } { P t } \sum _ { i = 1 } ^ { P } 1 [ x _ { t } ( i ) = \tt [ M A S K ] ] \log { p _ { \theta } ( x _ { 0 } ( i ) \mid x _ { t } ) } } \end{array}$ 

## 5. Average $l _ { \theta } ( x _ { 0 } , t , x _ { t } )$ across the batch to obtain $\hat { l } _ { \theta }$ and update θ via backpropagation.

Which tokens are eligible for masking depends on the training stage: in pretraining both prompts and responses are eligible, while in supervised fine-tuning only responses are. This algorithm is a special case of discrete diffusion (Austin et al., 2021) – MDMs with $\alpha _ { t } = t \ ( { \sec \ } \mathrm { E q } . \ 1 )$ . Nearly all pretrained MDMs adopt this setting. Beyond this, MDMs can model both text and image tokens (Yang et al., 2025), making it a unified framework for both language and multi-modal modeling. 

To minimize the training objective in Eq. 1, gradient-based methods such as stochastic gradient descent (SGD) are typically used (Robbins & Monro, 1951). However, these methods require an empirical estimate of Eq. 1, which is usually the batch mean of losses $\begin{array} { r l } { ~ } & { { } \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \ell ( x _ { 0 } ^ { ( i ) } , t _ { i } , x _ { t } ^ { ( i ) } } \end{array}$ $\theta )$ . For ARMs, this estimator is generally reliable, since the variance $( \operatorname { E q } . 2 )$ arises only from ${ \mathfrak { C } } .$ However, in MDM training, we question its reliability, since extra sources $\textcircled{8}$ and B come into play. While the standard MDM training procedure gives an unbiased estimator of Eq. 1, its variance can be significantly high. This raises a central question – can we construct an alternative estimator that remains unbiased but achieves lower variance, providing a more stable basis for optimization? 

In the rest of this section, we introduce six methods – two core methods and then four others – to address the problem. A qualitative analysis of their combinations is provided in A.6. 

## 3.1 CORE METHODS

## 3.1.1 P-POTS: PARAMETRIC-PARETO OPTIMAL T-SAMPLING

P-POTS (Parametric-Pareto Optimal Timestep Sampling) replaces the masking rate sampler $t \sim$ $U [ 0 , 1 ]$ ] with a data-fitted non-uniform one to reduce $\textcircled{8} + \textcircled { B } + \textcircled { C }$ . To estimate Eq. 1, standard training uses $t \sim U [ 0 , 1 ]$ directly: unbiased but high variance (§3). Instead, one could use another t-sampler that remains unbiased but reduces variance via importance weights (Kahn & Marshall, 1953). 

We now derive the optimal unbiased t-sampler in the view of minimizing $\textcircled{4} + \textcircled { B } + \textcircled { C }$ . If we instead sample $t \sim p ( t )$ , we can re-weight to form $\begin{array} { r } { \frac { 1 } { p ( t ) } l _ { \theta } } \end{array}$ . This estimator has the same expectation as $t \sim U [ 0 , 1 ]$ and is thus still unbiased in estimating the training objective 1, as shown below: 

$$
\mathbb {E} _ {t \sim p, x _ {t}} \left[ \frac {1}{p (t)} l _ {\theta} (x _ {0}, t, x _ {t}) \right] = \int_ {0} ^ {1} p (t) \frac {1}{p (t)} g (t) d t = \int_ {0} ^ {1} g (t) d t = \mathbb {E} _ {t \sim U, x _ {t}} (l _ {\theta}),\tag{3}
$$

With Eq. 3 establishing the unbiasedness, we next turn to minimizing its variance from $x _ { 0 } , t$ and $x _ { t } .$ , which is exactly $\widetilde { \mathrm { ( A ) } } + \widetilde { \mathrm { B } } + \widetilde { \mathrm { C } } \mathrm { ) }$ in Eq. 2. Define $v ( t ) = \operatorname { V a r } _ { x _ { 0 } , x _ { t } } ( l _ { \theta } ^ { \circ } | \ t )$ and $g ( t ) = E _ { x _ { 0 } , x _ { t } } ( l _ { \theta } \ |$ $t )$ . Under regularity conditions, Fubini’s theorem (Rudin, 1987) allows us to change the order of expectation between t and $x _ { 0 } , x _ { t }$ . The variance can then be written as (detailed proof in A.5.2): 

$$
Ⓐ + ⓐ + Ⓒ = \operatorname{Var} _ {x _ {0}, t, x _ {t}} (\frac {1}{p (t)} l _ {\theta} (x _ {0}, t, x _ {t}))) = \int_ {0} ^ {1} \frac {g (t) ^ {2} + v (t)}{p (t)} d t - \Big (\int_ {0} ^ {1} g (t) d t \Big) ^ {2}.\tag{4}
$$

$$
p ^ {*} (t) = \frac {\sqrt {g (t) ^ {2} + v (t)}}{\int_ {0} ^ {1} \sqrt {g (t) ^ {2} + v (t)} d t} \propto \sqrt {g (t) ^ {2} + v (t)}
$$

$$
Ⓐ, Ⓑ
$$

$$
Ⓒ
$$

$$
U [ 0, 1 ]
$$

Algorithm of P-POTS. Since $g ( t )$ and $v ( t )$ are unknown, P-POTS estimates them empirically: before training, for each of $\{ x _ { 0 } ^ { ( i ) } \} _ { i = 1 } ^ { a }$ , draw b values $\{ t _ { j } \} _ { j = 1 } ^ { b }$ and, for each $( x _ { 0 } ^ { ( i ) } , t _ { j } )$ , draw c masked samples $\{ x _ { t } ^ { ( i , j , k ) } \} _ { j = 1 } ^ { c }$ with losses $\ell _ { i , j , k } : = l _ { \theta } ( x _ { 0 } ^ { ( i ) } , t _ { j } , x _ { t } ^ { ( i , j , k ) } )$ . We then compute 

$$
\hat {g} _ {j} = \frac {1}{a c} \sum_ {i = 1} ^ {a} \sum_ {k = 1} ^ {c} \ell_ {i, j, k}, \hat {v} _ {j} = \frac {1}{a} \sum_ {i = 1} ^ {a} \frac {1}{c - 1} \sum_ {k = 1} ^ {c} (\ell_ {i, j, k} - \frac {1}{c} \sum_ {k ^ {\prime} = 1} ^ {c} \ell_ {i, j, k ^ {\prime}}) ^ {2}, \hat {p} _ {j} = \sqrt {\hat {g} _ {j} ^ {2} + \hat {v} _ {j}} / \sum_ {j = 1} ^ {c} \sqrt {\hat {g} _ {j} ^ {2} + \hat {v} _ {j}}
$$

for $j = 1 , \dotsc , b$ . The weights $\{ \hat { p } _ { j } \} _ { j = 1 } ^ { b }$ represent an empirical scatter of the underlying optimal $p ^ { * } ( t )$ . To capture its structure, we design a model called EPR (Exponential-Polynomial Root) as: 

$$
p ^ {\mathrm{EPR}} (t) = \sqrt {a t ^ {r} + b (1 - t) ^ {q} + A ^ {2} \exp (2 \kappa t ^ {m})}, \qquad a, b, A, \kappa > 0, r, q \geq 0, m > 1\tag{5}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/f0eb91ff-cd64-48a5-ae07-430a6a4f0bcb/b7f908c7ffb738b996e72c5115acd1147d9055a21e9d41fd9ec9d1d59ea59e50.jpg)



(a) OpenScience


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/f0eb91ff-cd64-48a5-ae07-430a6a4f0bcb/8042ce480403501d39d3e8f351dae922943b82b617729d79628b67fbf952a4c3.jpg)



(b) GSM8K


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/f0eb91ff-cd64-48a5-ae07-430a6a4f0bcb/1354357bfa2e51691658f9d17e10ab31a1295decbc9c3307d6bd8ad328303181.jpg)



(c) HiTab



Figure 2: Empirical $\{ p _ { j } \} _ { j = 1 } ^ { b }$ (scatter) and the fitted curve $p ^ { * } ( t ) = \sqrt { g ( t ) ^ { 2 } + v ( t ) }$ (line) on three datasets: (a) OpenScience, (b) GSM8K, and (c) HiTab. The equations in each subplot show the fitted forms of $g ( t )$ and $v ( t )$ which together characterize the sampling distribution across masking rates t.


with $g ^ { \mathrm { E P R } } ( ( t ) = A \exp ( \kappa t ^ { m } )$ and $v ^ { \mathrm { E P R } } ( ( t ) = b ( 1 - t ) ^ { q } + a t ^ { r }$ , and we fit it via KL divergence. By the EPV rule that suggests 10 data points per fitted parameter (Peduzzi et al., 1996), we set $b = 7 0$ and slightly raise a and c to 15, which proved reliable. As shown in Figure 2, EPR fits $\{ \hat { p } _ { j } \} _ { j = 1 } ^ { 7 0 }$ almost perfectly with only seven parameters, suggesting it captures the characteristic form of optimal $p ^ { * } ( t )$ 

Intuition behind P-POTS. We call a t hard to train if $\sqrt { g ^ { 2 } ( t ) + v ( t ) }$ is large, typically at higher t (Figure 2). Our P-POTS allocates more samples to these regions, providing extra training on them, while the reweighting factor $1 / p ^ { * } ( t )$ keeps their contribution small so that their noisy signals do not dominate and hurt global optimization. In this way, $p ^ { * } ( t )$ acts as an adaptive sampler, directing training effort where it is most needed and using resources more efficiently. 

Intuition behind EPR. To motivate our forms of $v ^ { \mathrm { E P R } } ( t )$ and $g ^ { \mathrm { E P R } } ( t )$ in Eq. 5, we consider loss behavior at different t. The mean loss $g ( t )$ reflects average task difficulty at t. Its growth is not just because more tokens are masked, but because entire reasoning chains are broken. Once a chain is cut, the model must guess at multiple points, and errors compound multiplicatively (or equivalently, add in log space), motivating an exponential form with delayed ignition. The design of $m > 1$ ensures $g ^ { \mathrm { E P R ^ { \prime } } } ( 0 ) = 0$ , since the average loss $g ( t )$ at $t = 0$ is 0 and must be a local minimum. 

Meanwhile, empirically, we find that $v ( t )$ falls when t is small, rises after a local minimum, and falls again after a local maximum – thus exhibiting up to two turning points. We explain this behavior as follows. At small t, a loss spike occurs if a critical set of r key tokens is masked simultaneously $( \sim t ^ { r } )$ . At large t, loss remains high unless a set of $q$ key tokens survives $( \sim ( 1 - t ) ^ { q } )$ . Since the variance of a rare event is approximately its probability, and multiple nearly independent rare events contribute additively, these mechanisms together generate the observed curvature . Based on this explanation, we propose the model $v ( t ) = \bar { a } t ^ { r } + \bar { b ( 1 - t ) } ^ { q }$ 

While one may wonder if it suffices to naively fit $\{ p _ { j } \} _ { j = 1 } ^ { b }$ with polynomials, we argue this is problematic for three reasons. First, as discussed in the next paragraph, $p ( t )$ faces a drift problem: its shape may evolve during training. Although this drift is typically slow, it can become severe if the training loss remains high $( e . g .$ ., MMaDA on text-to-image-2M). Consequently, polynomials quickly lose effectiveness if the true functional form of $p ^ { * } ( t )$ is not well understood. Experiments in A.8 show that EPR performs better in this case. Second, polynomials require manual tuning of degree, while P-POTS requires no tuning at all. Third, they lack interpretability and ignore known structure. 

Drift of $p ( t )$ . While $p ( t )$ reduces variance, it may become outdated as model evolves during training—a phenomenon we refer to as the drift $o f p ( t )$ . In practice, P-POTS only fits $p ^ { * } ( t )$ once before training and then replaces the uniform sampler. This incurs negligible overhead, but leaves the sampler static and therefore potentially exposed to drift. Empirically, however, the drift tends to be slow, since our experiments show strong gains even without correction; see §4. An adaptive strategy, such as periodically re-estimating $p ( t )$ to re-estimate $p ( t )$ , could address this. Given the strong empirical performance of P-POTS, we leave such validation to future work. 

## 3.1.2 MIRROR: VARIANCE REDUCTION WITH MIRRORED MASKS

MIRROR (Variance reduction with MIRRORed masks) targets $\textcircled{8}$ by creating two masked samples complementarily from the same $x _ { 0 }$ and t. With the notation introduced in $\ S 3$ , it works as follows: 

1. Sample clean data $x _ { 0 }$ from the data distribution. 

2. Sample a masking rate $t \sim \mathcal { U } [ 0 , 1 ]$ 

3. After sampling $x _ { 0 }$ and $t ,$ for each eligible token $i ,$ sample, and generate two noisy samples: 

• $\boldsymbol { x } _ { t } ^ { 1 }$ by masking $x _ { 0 } ( i )$ if $U _ { i } < t$ 

• $\boldsymbol { x } _ { t } ^ { 2 }$ by masking $x _ { 0 } ( i )$ if $U _ { i } > 1 - t$ 

4. Compute $\begin{array} { r } { l _ { j } = - \frac { 1 } { t } \sum _ { i = 1 } ^ { P } 1 \big [ x _ { t } ^ { j } ( i ) = [ \mathtt { M A S K } ] \big ] \log p _ { \theta } ( x _ { 0 } ( i ) \mid x _ { t } ^ { j } ) \mathrm { f o r } j \in \{ 1 , 2 \} } \end{array}$ 

5. Average $\begin{array} { r } { \bar { l } = \frac { 1 } { 2 } ( l _ { 1 } + l _ { 2 } ) } \end{array}$ across batch for backpropagation. 

Compared to standard training that averages $l _ { 1 }$ from a single $\boldsymbol { x } _ { t } ^ { 1 }$ across the batch, MIRROR uses $\bar { l } ,$ which remains unbiased in estimating the the training objective $( \mathrm { E q . ~ 1 ) }$ . It uses the same number of $x _ { 0 }$ samples, thereby avoiding overfitting. While it doubles training cost due to an extra forward pass, it is particularly effective on long-response datasets and shows strong synergy with P-POTS. 

Intuition behind MIRROR. MIRROR can be understood as a hedging strategy between two assets. To reduce $\textcircled{8} .$ one needs a loss estimate from $x _ { 0 }$ with lower variance that arises solely from masking patterns. Suppose the token sequence is masked at n positions, if the easier tokens are masked $( e . g .$ uninformative prefix tokens) versus the harder ones $( e . g .$ , answer tokens), the resulting losses can differ significantly. To reduce this variance, MIRROR samples two noised inputs as described above. Due to the “complementary” masking pattern design, the two noised samples tend to yield negatively correlated losses. Consequently, regardless of whether $\boldsymbol { x } _ { t } ^ { 1 }$ correspond to an easier or harder case, $x _ { t } ^ { 2 }$ provides the complementary one, and the averaged loss from them still remains a reliable estimate. 

$\textcircled{4}$ Reduced by at Least a Half. Given $x _ { 0 }$ and t, the only randomness comes from $x _ { t } .$ Since $l _ { 1 }$ and $l _ { 2 }$ are identically distributed, define $\sigma ^ { 2 } = \mathrm { V a r } ( l _ { 1 } ) = \mathrm { V a r } ( l _ { 2 } )$ and $\rho = \mathrm { C o r r } ( l _ { 1 } , l _ { 2 } ) \le 0$ . Then 

$$
\operatorname{Var} (\bar {l}) = \operatorname{Var} \left(\frac {1}{2} \left(l _ {1} + l _ {2}\right)\right) = \frac {1}{4} \left(\operatorname{Var} \left(l _ {1}\right) + \operatorname{Var} \left(l _ {2}\right) + 2 \operatorname{Cov} \left(l _ {1}, l _ {2}\right)\right) = \frac {2 \sigma^ {2}}{4} (1 + \rho) = \frac {\sigma^ {2}}{2} (1 + \rho),
$$

The effectiveness of variance reduction depends on $\rho = \mathrm { C o r r } ( l _ { 1 } , l _ { 2 } )$ . Since $\rho \in [ - 1 , 1 ] , \operatorname { V a r } ( { \bar { l } } ) \leq$ $\mathrm { V a r } ( l _ { 1 } )$ , so MIRROR is never worse than standard training. Because $\boldsymbol { x } _ { t } ^ { 1 }$ and $x _ { t } ^ { 2 }$ are constructed complementarily, $\rho$ tends to be negative, which already reduces $\textcircled{8}$ by at least a half. For $t < 0 . 5$ the two masks have no overlap, producing even stronger negative correlation between $l _ { 1 }$ and $l _ { 2 }$ and thus greater variance reduction. This explains its synergy with P-POTS (discussed later in §3.1.3). 

Comparison to MultiSample-k. We refer to as MultiSample-k the method that estimates $l _ { \theta } ( \cdot \ |$ $x _ { 0 } , t )$ by drawing k independent masked samples $x _ { t } ^ { ( 1 ) } , \ldots , x _ { t } ^ { ( k ) }$ from the same masking strategy $U _ { i } < t$ and averaging $\begin{array} { r } { \tilde { l } = \frac { 1 } { k } \sum _ { j = 1 } ^ { k } l _ { \theta } ( x _ { t } ^ { ( j ) } \mid x _ { 0 } , t ) } \end{array}$ . Our comparison focuses on: 

• Negative Correlation: MultiSample-2 uses independent masks, so $\mathrm { C o v } ( l _ { 1 } , l _ { 2 } ) = 0 .$ . MIRROR introduces natural complementarity, leading to $\mathrm { C o v } ( l _ { 1 } , l _ { 2 } ) < 0$ and better variance reduction. 

• Token Coverage: Only masked tokens contribute to gradient updates. Intuitively, masking more tokens reduces pattern noise, as each token carries less individual weight when more contribute. Standard training masks each token with probability t, giving expected coverage $t P .$ MIRROR uses complementary masks, which together maximize two-view coverage as ${ \mathrm { P r } } ( i \in M _ { 1 } \cup M _ { 2 } ) \ =$ $\operatorname* { P r } ( U _ { i } < \mathbf { \dot { \tau } } t ) + \operatorname* { P r } ( \dot { U } _ { i } > 1 - t ) \ = \ \operatorname* { m i n } ( 1 , 2 t )$ . By contrast, MultiSample2 does not have double coverage: $\operatorname* { P r } ( i \in M _ { 1 } \cup M _ { 2 } ) = 1 - \operatorname* { P r } ( i \notin M _ { 1 } \cap \dot { M } _ { 2 } ) = 1 - \left( 1 - t \right) ^ { 2 } = 2 t - t ^ { 2 } \leq 2 t$ 

## 3.1.3 SYNERGY OF P-POTS AND MIRROR

P-POTS and MIRROR are compatible because they rely on non-interacting assumptions – accurate modeling of $p ( t )$ and negative correlation between $( x _ { t } ^ { 1 } , \dot { x } _ { t } ^ { 2 } )$ . However, why their combination yields synergies beyond simple additivity $( 1 + 1 > 2 )$ worth further investigation. 

To answer this question, we return to Eq. 4, the variance in estimating the training objective $( l _ { \theta }$ in Eq. 1). With $p ( t ) \propto \sqrt { g ^ { 2 } ( t ) + v ( t ) }$ , it becomes $\begin{array} { r } { \bigg ( \int _ { 0 } ^ { 1 } \sqrt { { g ^ { * } } ^ { 2 } ( t ) + v ^ { * } ( t ) } d t \bigg ) ^ { 2 } - \bigg ( \int _ { 0 } ^ { 1 } g ^ { * } ( t ) d t \bigg ) ^ { 2 } } \end{array}$ Since this value no longer depends on $p ( t )$ , for fixed ${ \bf \boldsymbol { g } } ( t )$ and $v ( t )$ , it cannot be further reduced by altering $p ( t )$ . However, it can still be lowered by changing $g ( t )$ and $v ( t )$ during training. 

Suppose $\boldsymbol { v } ( t ) \ \ne \ \boldsymbol { 0 }$ and $\textstyle \int _ { 0 } ^ { 1 } v ( t ) d t \ = \ V \ > \ 0$ is fixed. The variance depends on how close $\begin{array} { r } { \int _ { 0 } ^ { 1 } \sqrt { g ^ { 2 } ( t ) + v ( t ) } } \end{array}$ dt and $\begin{array} { r } { \int _ { 0 } ^ { 1 } g ( t ) \ d t } \end{array}$ dt are. To bring them closer, $v ( t )$ should be concentrated where $g ( t )$ is large, as $\sqrt { g ^ { 2 } ( t ) + v ( t ) }$ is less sensitive to $v ( t )$ in this regime. MIRROR helps here by suppressing $v ( t )$ most strongly at moderate t, leaving v(t) relatively larger where $g ( t )$ is high. Thus, MIRROR naturally shifts $v ( t )$ towards the direction favored by P-POTS. Together they reinforce each other: MIRROR clears moderate-t regions, while P-POTS emphasizes low- and high-t. This synergy is remarkable, as naive method combinations rarely deliver such gains (A.6). 

## 3.2 OTHER TECHNIQUES

We introduce several other variance-reduction techniques as baseline methods in our experiments. 

## 3.2.1 ISAD: IMPORTANCE SAMPLING ON ANSWER DELIMITERS

ISAD shift masking toward answer delimiters, targeting $\textcircled{8}$ , by the sampling distribution: 

$$
q _ {j} (t) = \left\{ \begin{array}{l l} t, & \text { if   token   j\notin   rare   ids }, \\ \min (1, t + \Delta), & \text { if   token   j\in   rare   ids }, \end{array} \right.
$$

with $\Delta \in ( 0 , 1 )$ . To keep the estimator unbiased, token-level losses are reweighted by $1 / q _ { j } ( t )$ . The intuition is to guide the model on when to output answer tokens for better downstream performance. 

## 3.2.2 SYRM: SYNTAX-AND-RESPONSE MASK

SyRM is designed for structured data such as HTML tables, source code, or graphs. We partition tokens into two groups: (i) response tokens $R ,$ and (ii) syntax tokens C in the prompt. For example, in an HTML table, $\dot { C }$ includes tags such as <table>, <tr>, <th>, and <td>. SyRM modifies the eligibility set for masking to $R \cup C ,$ so that both response tokens and syntax tokens may be masked. 

Rationale. Assuming (i) tokens within each group are homogeneous in first and second moments of token-level losses, (ii) syntax tokens are predicted more stably than response tokens, and (iii) reasoning errors couple response tokens more strongly, we show that SyRM reduces masking-pattern noise $\textcircled{8}$ at the cost of a small, bounded bias (optimum shift). Overall, the mean-squared error of estimating training loss can still decrease. Intuitively, by treating easy-to-predict syntax tokens as auxiliary targets, SyRM stabilizes training and mitigates mask-pattern noise, while still preserving the primary learning signal on response tokens (see A.5.4 for the rigorous proof). 

## 3.2.3 STRATS: STRATIFIED T-SAMPLING

Instead of drawing i.i.d. $t _ { i } \sim \mathcal { U } [ 0 , 1 ]$ for batch size n, partition [0, 1] into k equal strata $[ j / k , ( j +$ $1 ) / k ]$ with $j = 0 , \ldots , k - 1$ and draw $\left\lfloor { \frac { n } { k } } \right\rfloor$ values of $t _ { i }$ per stratum, with leftovers assigned randomly. 

This approach was first introduced in Sahoo et al. (2024) with $k = n$ . In this work, we prove that stratified sampling yields an unbiased estimator and reduces $\textcircled{8}$ via inter-stratum variance. Furthermore, we recommend $k = \lceil { \sqrt { n } } \rceil$ instead of $k = n$ as a general strategy (see Appendix A.5.5). Although StraTS uses stratified sampling for the t-sampler instead of $p ^ { * } ( t ) \left( \ S 3 . 1 . 1 \right)$ and is not optimal for minimizing $A _ { x _ { 0 } } + B _ { x _ { 0 } }$ , it still outperforms baselines and is easy to implement. 

## 3.2.4 EMA: BIN-WISE EMA CONTROL VARIATE

To reduce $\textcircled{8}$ , we reduce its variance arising from heteroscedastic losses across masking rates by maintaining an exponential moving average (EMA) of losses within bins of $t \in [ 0 , 1 ]$ Each bin tracks a baseline that is updated online, yielding a piecewise-constant approximation of the expected loss against t. By using these bin-wise estimates as control variates, we cancel baseline-related noise more effectively than with fixed scaling, while preserving unbiasedness of gradient estimates. 

This approach requires choosing the number of bins $m$ and EMA rate $\eta ,$ which balances stability and adaptivity. From our MSE-minimization analysis in A.5.6, we derive m×batch size per GPU ≈ $0 . 1 \times n \times$ (train data size), which serves as a practical rule for selecting m and $\eta .$ 

## 4 EXPERIMENTS

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/f0eb91ff-cd64-48a5-ae07-430a6a4f0bcb/e10b49bd253908d9e68cafb9d5bf6f941141ef3bdecfcb22b6309575ab8aa24a.jpg)



(a)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/f0eb91ff-cd64-48a5-ae07-430a6a4f0bcb/5b69d05d9ec4a0c210d0a9125e66d27e246c042e75477077f042c0d86760d7dd.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/f0eb91ff-cd64-48a5-ae07-430a6a4f0bcb/06f3245798b22986a951b4759c901746e39a155ffa9a26acf5488e20b16cdd14.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/f0eb91ff-cd64-48a5-ae07-430a6a4f0bcb/8d3970bdfccbb7db01bf3974d4aacb440a3427d9d40d44001efc7d633ccc69d1.jpg)



(b)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/f0eb91ff-cd64-48a5-ae07-430a6a4f0bcb/4f3397da042e213e515e7148068de982643b7829697d4facfc9db541bb5671e3.jpg)



(d)



(e)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/f0eb91ff-cd64-48a5-ae07-430a6a4f0bcb/9f4639d2dd4eed2e50473a0cedbbcdefd7a89b5035cb49683d2b9411a2bfd753.jpg)



(c)



(f)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/f0eb91ff-cd64-48a5-ae07-430a6a4f0bcb/b131fc6dcf3a665a23d26682cb2fad4ef63b249f00847c5e376a7fa5b1d86059.jpg)



(g)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/f0eb91ff-cd64-48a5-ae07-430a6a4f0bcb/d712d984428d06011d3c1959313c8ddaa2d527ba686341bec16ef0c491b72d3a.jpg)



(h)



Figure 3: Images generated by MMaDA-8B-MixCoT trained with P-POTS+MIRROR (top row) and the standard method (bottom row) under the same seed. The columns correspond to the prompts: 1) A woman with blonde hair is wearing a colorful crochet headband and a black top with a floral patterned shawl. She is holding a toy sword with a green handle and a yellow ball on top; 2) A rustic wooden farmhouse with a weathered roof and a small porch stands in a field of dry grass. The sky is filled with fluffy white clouds, and a single bird is flying in the distance. The landscape is devoid of any other structures or signs of life, giving the scene a serene and isolated feel; 3) A car is on fire under a bridge with smoke billowing out. The fire is intense, with flames visible on the car’s underside. The bridge has a metal railing and is located near a forested area. There is a street lamp above the bridge, and the scene appears to be at night or during dusk; 4) A close-up of a grey fringed purse with a tassel detail on the front. The purse is placed on a surface with a soft, blurred background; more case studies can be found in A.11.


Limited by resources, and given that current MDMs perform comparably to strong ARM baselines (Nie et al., 2025; Yang et al., 2025), we focus our validation on supervised fine-tuning, though our methods naturally extend to pretraining. We evaluate our methods using the setup below: 

Setting. We train each model three times independently using seeds 42, 731 and 20231, holding model, data and all hyper-parameters fixed, and evaluate the trained models by greedy decoding. This setting ensures difference across runs arise solely from training, not inference. 

Tasks. We evaluate on three textual datasets – OpenScience (NVIDIA Corporation, 2025), GSM8K (Cobbe et al., 2021), and HiTab (Cheng et al., 2022) – as well as one image generation dataset, text-to-image-2M (Hate, 2024). OpenScience is a multi-domain, knowledge-intensive dataset spanning humanities, law, and STEM; GSM8K targets mathematical reasoning; and HiTab focuses on tabular question answering. Together with the multimodal text-to-image-2M benchmark, diverse data modalities and domains are covered. Details of data processing and training/inference hyper-parameters are provided in Appendix A.7. We report pass@1 accuracy for OpenScience, GSM8K and HiTab, and CLIP score for text-to-image-2M. 

Models. We evaluate two MDMs in total. On textual datasets, we evaluate one MDM, LLaDA-8B-Instruct (Nie et al., 2025), with three ARMs, Qwen2.5-7B-Instruct (Team, 2024), Qwen3-8B (Team, 2025) and Llama3.1-8B-Instruct (Meta AI, 2024b). For text-to-image-2M, we evaluate one multimodal MDM, MMaDA-8B-MixCoT (Yang et al., 2025). 

Baselines. We compare our methods to two baselines: (i) standard training (§3), and (ii) the clipped noise schedule of Arriola et al. (2025), which has been reported to achieve the best performance on perplexity across most standard schedules (Chang et al., 2022), including linear, logarithmic, square root, square, and cosine. We therefore adopt it as a representative baseline. 


Table 1: Per-seed performance comparison of MDMs and ARMs. The best performance within MDMs training methods is bolded. Training is repeated under three seeds, with model, data and all hyper-parameters fixed, and we evaluate trained models by greedy decoding, so difference across runs are solely from training rather than inference. Avg is the mean over available runs.


<table><tr><td rowspan="2">Method (Seed)</td><td colspan="4">OpenScience</td><td colspan="4">GSM8K</td><td colspan="4">HiTab</td></tr><tr><td>42</td><td>731</td><td>20231</td><td>Avg</td><td>42</td><td>731</td><td>20231</td><td>Avg</td><td>42</td><td>731</td><td>20231</td><td>Avg</td></tr><tr><td colspan="13">LLaDA-8B-Instruct</td></tr><tr><td>P-POTS+Mirror</td><td>50.80</td><td>53.80</td><td>53.00</td><td>52.53</td><td>62.02</td><td>58.61</td><td>60.96</td><td>60.53</td><td>66.02</td><td>66.67</td><td>68.62</td><td>67.10</td></tr><tr><td>P-POTS</td><td>45.34</td><td>47.60</td><td>47.47</td><td>46.80</td><td>59.59</td><td>59.36</td><td>56.79</td><td>58.58</td><td>60.56</td><td>64.74</td><td>58.80</td><td>61.37</td></tr><tr><td>MIRROR</td><td>45.63</td><td>47.25</td><td>46.25</td><td>46.38</td><td>56.56</td><td>51.40</td><td>53.15</td><td>53.70</td><td>64.72</td><td>65.04</td><td>63.68</td><td>64.48</td></tr><tr><td>ISAD</td><td>45.40</td><td>-</td><td>-</td><td>45.40</td><td>55.27</td><td>-</td><td>-</td><td>55.27</td><td>64.39</td><td>-</td><td>-</td><td>64.39</td></tr><tr><td>SyRM</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>59.79</td><td>-</td><td>-</td><td>59.79</td></tr><tr><td>StraTS</td><td>44.69</td><td>-</td><td>-</td><td>44.69</td><td>55.57</td><td>-</td><td>-</td><td>55.57</td><td>61.86</td><td>-</td><td>-</td><td>61.86</td></tr><tr><td>EMA</td><td>47.08</td><td>-</td><td>-</td><td>47.08</td><td>52.39</td><td>-</td><td>-</td><td>52.39</td><td>61.66</td><td>-</td><td>-</td><td>61.66</td></tr><tr><td>Clipped</td><td>47.90</td><td>49.20</td><td>50.90</td><td>49.33</td><td>55.57</td><td>54.60</td><td>58.38</td><td>56.18</td><td>58.90</td><td>60.80</td><td>62.20</td><td>60.63</td></tr><tr><td>Standard</td><td>43.57</td><td>46.50</td><td>46.50</td><td>45.52</td><td>50.64</td><td>53.22</td><td>53.68</td><td>52.51</td><td>52.96</td><td>60.43</td><td>62.64</td><td>58.68</td></tr><tr><td colspan="13">Auto-Regressive Large Language Models</td></tr><tr><td>Qwen2.5-7B-Instruct</td><td>47.85</td><td>46.30</td><td>45.10</td><td>46.42</td><td>56.20</td><td>54.66</td><td>53.60</td><td>54.82</td><td>46.55</td><td>45.74</td><td>45.20</td><td>45.83</td></tr><tr><td>Qwen3-8B-Base</td><td>53.05</td><td>51.70</td><td>50.10</td><td>51.62</td><td>72.40</td><td>70.58</td><td>69.80</td><td>70.93</td><td>49.70</td><td>48.84</td><td>48.20</td><td>48.91</td></tr><tr><td>Llama3.1-8B-Instruct</td><td>44.95</td><td>47.07</td><td>47.76</td><td>46.59</td><td>47.60</td><td>46.40</td><td>49.10</td><td>47.70</td><td>65.02</td><td>66.28</td><td>64.88</td><td>65.39</td></tr></table>


Table 2: Per-seed performance of MMaDA-8B-MixCoT on text-to-image-2M. We report CLIP scores under three training seeds with identical model, data, and hyper-parameters. Avg is the mean CLIP over runs.


<table><tr><td rowspan="2">Method (Seed)</td><td colspan="3">text-to-image-2M</td><td rowspan="2">Avg</td></tr><tr><td>42</td><td>731</td><td>20231</td></tr><tr><td>P-POTS+Mirror</td><td>35.27</td><td>34.10</td><td>34.39</td><td>34.59</td></tr><tr><td>Standard</td><td>28.61</td><td>34.28</td><td>31.19</td><td>31.36</td></tr><tr><td>P-POTS</td><td>31.51</td><td>30.02</td><td>33.41</td><td>31.65</td></tr><tr><td>Mirror</td><td>34.88</td><td>33.77</td><td>33.57</td><td>34.07</td></tr></table>

## 4.1 KEY INSIGHTS

MDMs exhibit uniquely high training variance, which makes gradient updates noisy and optimization less stable. As a result, even pretrained MDMs are shown to achieve comparable performance with strong ARMs, after task-specific training, they often under-perform by a wide margin. This instability is evident in Table 1: under standard MDM training on HiTab, accuracy varies substantially across random seeds (about 52% ∼ 63%), and the mean performance across datasets frequently remains below that of ARMs. Our method targets this root cause by reducing MDM training variance. Figure 4 shows that P-POTS+MIRROR (green) converges to a lower final loss with smoother dynamics than the Standard method (red), indicating more stable training. This improved stability shows consistent improvement and closes the training gap effectively: on all three textual benchmarks, even the best Standard/Clipped run remains below the worst run of our method (OpenScience: best Standard 46.50 vs. worst P-POTS+MIRROR 50.80; GSM8K: 53.68 vs. 58.61; HiTab: 62.64 vs. 66.02). On text-to-image-2M, the two methods are essentially comparable (best Standard 34.28 vs. worst ours 34.10). This highlights the importance of variance reduction in delivering more consistent and reliable performance across runs. Overall, MDMs trained with our method can match, and in some cases surpass, strong ARMs, while run-to-run variability drops to near ARM levels. 

Moreover, ablation studies in A.8 confirm synergy between P-POTS and MIRROR. On GSM8K and OpenScience, compared to standard training, the accuracy gains from P-POTS+MIRROR (8.02% and 7.01% respectively) surpass the sum of the individual gains from P-POTS and MIRROR ((6.07+ $1 . 1 9 ) \% = 7 . 2 \hat { 6 } \%$ and $( 1 . 2 \bar { 8 } + 0 . 8 6 ) \% = 2 . 1 4 \%$ ), while on HiTab they are nearly additive (8.42% and $( 2 . 6 9 + 5 . 8 0 ) \% = 8 . 4 9 \% )$ . We emphasize that this is nontrivial, because as discussed in A.6, naive combinations of methods do not typically yield such improvements. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/f0eb91ff-cd64-48a5-ae07-430a6a4f0bcb/de56febedd67e862ab75ee30178d5ef1b9b6b3fe9ad2cdd44c778346e25c7a53.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/f0eb91ff-cd64-48a5-ae07-430a6a4f0bcb/3f87418e863dd23501c8fc9c07f60ebbc9bc26be44685d98354db0226c2788ae.jpg)



Figure 4: Training loss comparison between P-POTS+MIRROR (green) and Standard (red) on OpenScience (left) and text-to-image-2M (right). The annotations indicate the average loss at the beginning and end of training over the first and last 5 steps. Overall, P-POTS+MIRROR achieves more stable loss trajectories with consistently lower end losses compared to the Standard baseline.


MIRROR is especially effective on long-response datasets due to its design. As shown in Tables 1 and 2, its gains on OpenScience, HiTab and text-to-image-2M are substantially higher than on GSM8K. This aligns with its design: MIRROR targets masking pattern noise A , which becomes more severe with longer responses due to the greater number of possible masking patterns. 

MDMs achieve comparable performance as ARMs with much shorter outputs. In our experiments, MDMs often generate responses as short as 5% of ARM outputs and consistently shorter than training labels (see exact values in A.7.1). Truncating ARMs to the same length severely hurts their performance because outputs are often cut off mid-reasoning. One possible explanation is that MDMs decode by repeatedly unmasking a block of [MASK] tokens whose length is predetermined by the user-passed max new tokens, giving the model an implicit sense of how many tokens it is allowed to generate. In contrast, ARMs generate token by token without an explicit stopping criterion until hitting EOS, which can lead to verbosity or redundancy. 

On text-to-image-2M, the drift in p is more severe; see A.8.2. We attribute this to larger loss scales (and thus gradients) that move θ faster. Although EPR targets a U-shape, we use it as a stable compromise because P-POTS fits p(t) only once pre-training; empirically, EPR is more robust than fitting the initial S-shape, so we recommend it as a default. In principle, periodically refitting p(t) (e.g., at each eval) could further improve P-POTS and P-POTS+MIRROR. 

## 4.2 CASE STUDY

In Figure 3, standard training often produces blurry images or misses fine details. We suspect two causes. First, rare detail-critical tokens are masked too infrequently under uniform p(t), so the model under-trains on details. Consequently, features such as the “crochet headband” and “toy sword” in case (b) are missing or distorted. Second, cross-entropy on masked image tokens can show an averaging effect. When multiple outputs are possible, the model may learn to predict their mean; at inference with low temperature, argmax decoding turns this into unstable choices, leading to blur. This may explain case (d), where fire and smoke looks unrealistic. By contrast, P-POTS+MIRROR may address these problems. P-POTS emphasizes harder regions, while MIRROR averages two negatively correlated samples and reduces the tendency to hedge. Together, they could improve the gradient signal-to-noise ratio, giving sharper textures and more faithful objects, e.g. correct headband/sword in (a) and more realistic flames in (c). 

## 5 CONCLUSION

This work derives a systematic decomposition of MDM training variance as a fundamental framework for variance reduction, and proposes six corresponding techniques. Among them, P-POTS is provably Pareto-optimal for minimizing loss variance, while MIRROR is particularly beneficial on long-response datasets; together, they yield strong synergy. Our methods reduce both run-to-run variability and within-run suboptimality, making one training run sufficient for reliable performance. Looking forward, we expect our decomposition and methods to serve as building blocks for scalable MDM training, and to inspire future work toward a unified framework for variance control in masked diffusion models. 

## REFERENCES



Marianne Arriola, Aaron Gokaslan, Justin T Chiu, Zhihan Yang, Zhixuan Qi, Jiaqi Han, Subham Sekhar Sahoo, and Volodymyr Kuleshov. Block diffusion: Interpolating between autoregressive and diffusion language models. arXiv preprint arXiv:2503.09573, 2025. 





Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne Van Den Berg. Structured denoising diffusion models in discrete state-spaces. Advances in neural information processing systems, 34:17981–17993, 2021. 





Lukas Berglund, Meg Tong, Max Kaufmann, Mikita Balesni, Asa Cooper Stickland, Tomasz Korbak, and Owain Evans. The reversal curse: Llms trained on” a is b” fail to learn” b is a”. arXiv preprint arXiv:2309.12288, 2023. 





Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T. Freeman. Maskgit: Masked generative image transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11315–11325, 2022. 





Zhoujun Cheng, Haoyu Dong, Zhiruo Wang, Ran Jia, Jiaqi Guo, Yan Gao, Shi Han, Jian-Guang Lou, and Dongmei Zhang. Hitab: A hierarchical table dataset for question answering and natural language generation. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1094–1110, 2022. 





Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021. 





Shansan Gong, Ruixiang Zhang, Huangjie Zheng, Jiatao Gu, Navdeep Jaitly, Lingpeng Kong, and Yizhe Zhang. Diffucoder: Understanding and improving masked diffusion models for code generation. arXiv preprint arXiv:2506.20639, 2025. 





Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024. 





Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 





Jacky Hate. text-to-image-2m (revision e64fca4), 2024. URL https://huggingface.co/ datasets/jackyhate/text-to-image-2M. 





Paul Jeha, Will Grathwohl, Michael Riis Andersen, Carl Henrik Ek, and Jes Frellsen. Variance reduction of diffusion model’s gradients with taylor approximation-based control variate. arXiv preprint arXiv:2408.12270, 2024. 





Herman Kahn and Theodore E. Marshall. Methods of reducing sample variance in monte carlo computations. Journal ofthe Operations Research Society ofAmerica, 1(5):263–278, 1953. 





Myunsoo Kim, Donghyeon Ki, Seong-Woong Shim, and Byung-Jun Lee. Adaptive non-uniform timestep sampling for diffusion model training. arXiv preprint arXiv:2411.09998, 2024. 





Joseph-Louis Lagrange. Mecanique Analytique´ . Courcier, 1811. 





Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion modeling by estimating the ratios of the data distribution. arXiv preprint arXiv:2310.16834, 2023. 





Chenlin Meng, Yang Song, Wenzhe Li, and Stefano Ermon. Estimating high order gradients of the data distribution by denoising. Advances in Neural Information Processing Systems, 34:25359– 25369, 2021. 





Meta AI. Introducing llama 3.1: Our most capable models to date. Meta AI Blog, 2024a. URL https://ai.meta.com/blog/meta-llama-3-1/. 





Meta AI. Llama-3.1-8b-instruct. HuggingFace model card, 2024b. URL https:// huggingface.co/meta-llama/Llama-3.1-8B-Instruct. 





Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language diffusion models. arXiv preprint arXiv:2502.09992, 2025. 





NVIDIA Corporation. Openscience: A multi-domain synthetic dataset for general-purpose reasoning. https://huggingface.co/datasets/nvidia/OpenScience, 2025. Version 1; licensed under CC BY 4.0; data creation date May 15 2025. 





Peter Peduzzi, John Concato, Elizabeth Kemper, Theodore R. Holford, and Alvan R. Feinstein. A simulation study of the number of events per variable in logistic regression analysis. Journal of Clinical Epidemiology, 49(12):1373–1379, 1996. ISSN 0895-4356. doi: https://doi.org/10. 1016/S0895-4356(96)00236-3. URL https://www.sciencedirect.com/science/ article/pii/S0895435696002363. 





Herbert Robbins and Sutton Monro. A stochastic approximation method. The Annals ofMathematical Statistics, 22(3):400–407, 1951. 





Walter Rudin. Real and Complex Analysis. McGraw-Hill, 3 edition, 1987. 





Subham Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin Chiu, Alexander Rush, and Volodymyr Kuleshov. Simple and effective masked diffusion language models. Advances in Neural Information Processing Systems, 37:130136–130184, 2024. 





John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. Highdimensional continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438, 2015. 





John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. 





Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. 





Qwen Team. Qwen2.5: A party of foundation models, September 2024. URL https://qwenlm. github.io/blog/qwen2.5/. 





Qwen Team. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388. 





Yilun Xu, Shangyuan Tong, and Tommi Jaakkola. Stable target field for reduced variance score estimation in diffusion models. arXiv preprint arXiv:2302.00670, 2023. 





An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zhihao Fan. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024. 





Ling Yang, Ye Tian, Bowen Li, Xinchen Zhang, Ke Shen, Yunhai Tong, and Mengdi Wang. Mmada: Multimodal large diffusion language models. arXiv preprint arXiv:2505.15809, 2025. 





Jiacheng Ye, Zhihui Xie, Lin Zheng, Jiahui Gao, Zirui Wu, Xin Jiang, Zhenguo Li, and Lingpeng Kong. Dream 7b: Diffusion large language models. arXiv preprint arXiv:2508.15487, 2025. 





Runpeng Yu, Xinyin Ma, and Xinchao Wang. Dimple: Discrete diffusion multimodal large language model with parallel decoding. arXiv preprint arXiv:2505.16990, 2025. 





Fengqi Zhu, Rongzhen Wang, Shen Nie, Xiaolu Zhang, Chunwei Wu, Jun Hu, Jun Zhou, Jianfei Chen, Yankai Lin, Ji-Rong Wen, et al. Llada 1.5: Variance-reduced preference optimization for large language diffusion models, 2025. 



## A APPENDIX

A.1 Statement ..... 14
A.2 Preliminaries ..... 15
A.2.1 Stochastic Gradient Descent on Convex Objectives ..... 15
A.2.2 Discrete Diffusion Models ..... 15
A.2.3 Proximal Policy Optimization ..... 17
A.2.4 Group Relative Policy Optimization ..... 17
A.3 Training Variance Decomposition ..... 18
A.4 Algorithms ..... 18
A.4.1 Bin-Wise EMA ..... 18
A.5 Proofs ..... 19
A.5.1 Training Variance Decomposition ..... 19
A.5.2 P-POTS: Pareto-Optimal Property ..... 20
A.5.3 Importance Sampling ..... 21
A.5.4 SyRM ..... 22
A.5.5 Stratified Sampler ..... 26
A.5.6 Bin-Wise EMA: Variance Reduction & Hyperparameter Selection ..... 27
A.6 Method Combination ..... 28
A.6.1 Stratified Sampling VS EMA ..... 29
A.6.2 ISAD VS SyRM ..... 30
A.7 Experimental Details ..... 30
A.7.1 Hyperparameter Settings ..... 30
A.7.2 How Training Variance is Estimated ..... 31
A.8 Complete Experimental Results ..... 32
A.8.1 Per-Run Experimental Results on LLaDA ..... 32
A.8.2 Per-Run Experimental Results on MMaDA ..... 33
A.9 Extension to Reinforcement Learning ..... 33
A.10 Future Work ..... 34
A.11 Case Study ..... 34
A.11.1 Case Study: Image Generation ..... 34 

## A.1 STATEMENT

Reproducibility. All models and datasets used in this work are publicly available, as listed in 4. Details of dataset splits and hyperparameter configurations are provided in Appendix A.7.1, while the evaluation protocols for inference results are described in Appendix 4. All theoretical proofs and assumptions are presented in Appendix A.5. 

LLM Usgae. We used large language models (LLMs) as a general-purpose assist tool in three ways: (i) polishing the writing for grammar and clarity, (ii) helping to identify potentially relevant related work (with all citations manually verified by the authors), and (iii) providing code assistance for implementing ideas developed by the authors. The research ideas, methodology, experiments, and final paper content remain the responsibility of the authors. 

## A.2 PRELIMINARIES

## A.2.1 STOCHASTIC GRADIENT DESCENT ON CONVEX OBJECTIVES

We consider the optimization problem 

$$
\min _ {\beta \in C} f (\beta),\tag{6}
$$

where $C \subseteq \mathbb { R } ^ { p }$ is a closed convex set and $f : C \to \mathbb { R }$ is convex. In machine learning, $f$ typically represents the population risk, i.e., the expected loss with respect to the true but unknown data distribution. A common example is the expected cross-entropy loss of the model under parameter $\beta$ relative to the true data distribution. 

Gradient-based optimization methods generally rely on estimates of the gradient of the expected loss. Using a larger batch size reduces the variance of this estimate but increases the computational cost per iteration. Stochastic Gradient Descent (SGD) strikes a balance by sampling a batch of fixed size at random and using the average loss gradient over this batch as an unbiased estimator of the population gradient. Convergence of SGD can be guaranteed in principle, provided that certain standard assumptions are satisfied: 

Theorem 1 Suppose $\hat { \beta }$ is a minimiser of f over a closed convex set $C \subseteq \mathbb { R } ^ { p }$ . Suppose $\begin{array} { r } { \operatorname* { s u p } _ { \beta \in C } \| \beta \| _ { 2 } \leq R < } \end{array}$ ∞ and 

$$
\sup _ {\beta \in C} \mathbb {E} \left(\sup _ {\tilde {g} \in \partial \tilde {f} (\beta ; U)} \| \tilde {g} \| _ {2} ^ {2}\right) \leq L ^ {2} <   \infty .
$$

Then if $\eta _ { s } \equiv \eta = 2 R / ( L \sqrt { k } )$ , the output $\bar { \beta }$ of the stochastic gradient descent algorithm satisfies 

$$
\mathbb {E} f (\bar {\beta}) - f (\hat {\beta}) \leq \frac {2 L R}{\sqrt {k}}.
$$

Note. From a statistic point of view, the batch mean is merely an estimator of the population gradient, and thus inevitably subject to estimation variance. In MDM training, where the expected loss is subject to three sources of variance, this issue becomes particularly severe. It is therefore important to develop estimators that are more variance-efficient. 

## A.2.2 DISCRETE DIFFUSION MODELS

Let $\mathbf { x } _ { 0 } \in \{ 1 , \ldots , K \} ^ { n }$ denote an initial discrete data vector, such as a length n token sequence, where each element $x _ { 0 } ^ { i }$ takes values from a finite vocabulary of size K. A discrete diffusion model defines a forward corruption process that transforms $\mathbf { x } _ { \mathrm { 0 } }$ into a sequence of progressively noisier variables $\mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { T }$ over T timesteps. 

At each timestep $t = 1 , \dots , T$ , let $x _ { t } ( k )$ represents the kth token of $\mathbf { x } _ { t }$ for $k \in \{ 1 , . . . , n \}$ , then each token $x _ { t - 1 } ( k )$ is corrupted into $x _ { t } ( k )$ using a predefined transition matrix $Q _ { t } \in \mathbb { R } ^ { K \times K }$ . Formally, the transition probabilities are defined as: 

$$
q (x _ {t} (k) = j \mid x _ {t - 1} (k) = i) = [ Q _ {t} ] _ {i, j}.\tag{7}
$$

where $Q _ { t }$ is a stochastic matrix with each column sum to 1. Let $\mathbf { v } ( x ) \in R ^ { K }$ be a one-hot vector with 1 at position x. For simplicity, we abbreviate token position information and use $x _ { t }$ for ${ \bf x } _ { t } ( k )$ for any $k \in \{ 1 , . . . , K \}$ . Then we can write the transition equation as 

$$
q (x _ {t} \mid x _ {t - 1}) = \mathbf {v} ^ {T} (x _ {t}) Q _ {t} \mathbf {v} (x _ {t - 1}).\tag{8}
$$

We can then marginalize intermediate steps and compute the transition probability from $x _ { 0 }$ to $x _ { t } \colon$ 

$$
q (x _ {t} \mid x _ {0}) = \mathbf {v} ^ {T} (x _ {t}) \tilde {Q} \mathbf {v} (x _ {0}) = \mathbf {v} ^ {T} (x _ {t}) Q _ {t} \dots Q _ {1} \mathbf {v} (x _ {0}).\tag{9}
$$

The posterior distribution conditioned on x<sub>0</sub> is then: 

$$
q (x _ {t - 1} | x _ {t}, x _ {0}) = \frac {q (x _ {t} | x _ {t - 1} , x _ {0}) q (x _ {t - 1} | x _ {0})}{q (x _ {t} | x _ {0})} = \frac {(\mathbf {v} ^ {T} (x _ {t}) Q _ {t} \mathbf {v} (x _ {t - 1})) (\mathbf {v} ^ {T} (x _ {t - 1}) \hat {Q} _ {t - 1} \mathbf {v} (x _ {0})))}{\mathbf {v} ^ {T} (x _ {t}) \hat {Q} _ {t} \mathbf {v} (x _ {0})}\tag{10}
$$

For the reverse (denoising) process, a neural network $p _ { \theta } ( x _ { t - 1 } \mid x _ { t } ) = { \mathrm { C a t e g o r i c a l } } { \bigl ( } x _ { t - 1 } ; P _ { \theta } ( x _ { t } ) { \bigr ) }$ is trained to undo this corruption, where $P _ { \theta } ( x _ { t } ) \in \mathbb { R } ^ { K }$ denotes the logits produced by the model. $p _ { \theta } ( x _ { t - 1 } \ | \ x _ { t } )$ is trained to estimate this posterior. Since this posterior is defined on predefined forward processes only (the transition matrix $Q _ { T }$ that controls the noisy level at each time step $t ) ,$ the design of transition matrix is crucial in the success of discrete diffusion models. A well-designed noise schedule ensures the model encounters a smooth curriculum of denoising tasks, leading to better final performance. 

Early works propose a uniform noise design: 

$$
Q _ {t} = \alpha_ {t} I + (1 - \beta_ {t}) \mathbf {1 1} ^ {T}, \beta_ {t} = \frac {1 - \alpha_ {t}}{K}
$$

This design means at each time step, a token has probability $\alpha _ { t } + \beta _ { t }$ to remain and $\beta _ { t }$ probability to transfer to other $K - 1$ tokens. This matrix has one eigenvalue λ = 1(with eigenvector 1) and all other $K - 1$ eigenvalues equal to $\alpha _ { t }$ . Thus, the signal of $x _ { 0 }$ decays like $\alpha _ { t } . . . \alpha _ { 0 }$ . If this goes to 0 too quickly, you lose all information. 

Among the general discrete diffusion model framework, masked diffusion models consistently achieves the best performance (Austin et al., 2021)(Lou et al., 2023). In masked diffusion, a specific design is employed in which the Kth category serves as an absorbing state, commonly corresponding to a special token such as [MASK]. Under this configuration, the transition matrix $Q _ { t }$ is defined as: 

$$
Q _ {t} = \left[ \begin{array}{c c c c c} \alpha_ {t} & 0 & \dots & 0 & 1 - \alpha_ {t} \\ 0 & \alpha_ {t} & \dots & 0 & 1 - \alpha_ {t} \\ \vdots & \vdots & \ddots & \vdots & 1 - \alpha_ {t} \\ 0 & 0 & \dots & \alpha_ {t} & 1 - \alpha_ {t} \end{array} \right].
$$

where $\alpha _ { t } \in [ 0 , 1 ]$ denotes the masking ratio at timestep $t .$ This design ensures that each token is independently replaced with the [MASK] token with probability $1 - \alpha _ { t }$ , and retained with probability $\alpha _ { t }$ . Note the [MASK] token is an absorbing state. By isolating noise into a dedicated [MASK] token, the model preserves a clean, low-dimensional signal subspace, which it can focus on when reconstructing $\mathbf { x } _ { t - 1 }$ 

The training objective maximizes a variational lower bound (ELBO) on log $p _ { \theta } ( x _ { 0 } )$ , as introduced in Eq. 1. Importantly, Sahoo et al. (2024) has shown that Eq. 1: 

1. is invariant to the specific choice of the noise schedule $\alpha _ { t }$ , and 

2. serves as an upper bound of 

$$
- \mathbb {E} _ {p _ {\mathrm{data}} (x)} \left[ p _ {\theta} (x) \right],
$$

thus enabling a principled and simplified training framework for masked diffusion models. 

Restricting attention to interpolating masked diffusion models—i.e., those whose forward process q interpolates between clean data $x \in \mathcal { V }$ and a target distribution $\operatorname { C a t } ( \cdot ; \pi )$ , Sahoo et al. (2024) derived a principled training objective Eq. 1 that is invariant to noisy schedules. Recent models such as LLaDA and MMaDA base their training on this training objective and adopt a log-linear noise schedule defined by: 

$$
\alpha_ {t} = \exp (\log (1 - t)) = 1 - t,
$$

which means that at each timestep $t ,$ every token is independently replaced with a [MASK] token with probability t. 

## A.2.3 PROXIMAL POLICY OPTIMIZATION

Proximal Policy Optimization (PPO) can be derived as a first-order approximation to the Trust Region Policy Optimization (TRPO) problem by replacing the hard KL-constraint with a KL-penalty. We begin with the constrained optimization 

$$
\max _ {\theta} \mathbb {E} _ {t} \big [ r _ {t} (\theta) \hat {A} _ {t} \big ] \mathrm{s.t.} \mathbb {E} _ {t} \big [ D _ {\mathrm{KL}} \big [ \pi_ {\theta_ {\mathrm{old}}} (\cdot | s _ {t}) \| \pi_ {\theta} (\cdot | s _ {t}) \big ] \big ] \leq \delta ,\tag{11}
$$

where 

$$
r _ {t} (\theta) = \frac {\pi_ {\theta} (a _ {t} \mid s _ {t})}{\pi_ {\theta_ {\text { old }}} (a _ {t} \mid s _ {t})} \quad \text { and } \quad \hat {A} _ {t} \text {   is   an   estimator   of   the   advantage   at   time   } t.
$$

Introducing a Lagrange multiplier $\beta$ for the KL-constraint yields the penalized surrogate objective 

$$
L ^ {\mathrm{KLPEN}} (\theta) = \mathbb {E} _ {t} \left[ r _ {t} (\theta) \hat {A} _ {t} - \beta D _ {\mathrm{KL}} \left[ \pi_ {\theta_ {\text { old}}} (\cdot | s _ {t}) \| \pi_ {\theta} (\cdot | s _ {t}) \right] \right].\tag{12}
$$

In practice, PPO proceeds by: 

1. Collect trajectories $\left\{ \left( s _ { t } , a _ { t } , r _ { t } \right) \right\}$ under the old policy $\pi _ { \theta _ { \mathrm { o l d } } }$ 

2. Compute advantage estimates $\hat { A } _ { t }$ (e.g., via GAE). 

3. For K epochs, perform minibatch gradient ascent on $L ^ { \mathrm { K L P E N } } ( \theta )$ 

$$
\theta \leftarrow \theta + \eta \nabla_ {\theta} L ^ {\mathrm{KLPEN}} (\theta).
$$

4. Optionally adjust $\beta$ to keep the KL close to a target δ: 

$$
\beta   \leftarrow   \left\{ \begin{array}{l l} \alpha_ {+}   \beta , & \bar {D} _ {D _ {\mathrm{KL}}} > 1. 5   \delta , \\ \alpha_ {-}   \beta , & \bar {D} _ {D _ {\mathrm{KL}}} <   \delta / 1. 5, \\ \beta , & \text { otherwise }, \end{array} \right.
$$

where $\bar { D } _ { D _ { \mathrm { K L } } } = \mathbb { E } _ { t } [ D _ { \mathrm { K L } } ( \pi _ { \theta _ { \mathrm { o l d } } } \| \pi _ { \theta } ) ]$ 

5. Set $\theta _ { \mathrm { o l d } }  \theta$ and repeat. 

## A.2.4 GROUP RELATIVE POLICY OPTIMIZATION

Group Relative Policy Optimization (GRPO) extends PPO to settings where we have a collection of $G$ groups (or tasks), each potentially requiring its own policy behavior while sharing parameters. Let $\bar { \mathcal { G } } = \{ 1 , \ldots , G \}$ denote the set of groups, with weightings $\{ \omega _ { g } \} _ { g \in \mathcal { G } }$ such that $\begin{array} { r } { \sum _ { g } \bar { \omega } _ { g } = 1 } \end{array}$ . For each group g, define 

$$
r _ {t} ^ {g} (\theta) = \frac {\pi_ {\theta} (a _ {t} \mid s _ {t} , g)}{\pi_ {\theta_ {\text { old }}} (a _ {t} \mid s _ {t} , g)}, \quad \hat {A} _ {t} ^ {g} = \text { advantage   estimate   for   group } g \text { at   time } t,
$$

and the group-specific KL divergence 

$$
D _ {D _ {\mathrm{KL}}} ^ {g} (\theta) = \mathbb {E} _ {t \sim g} \Big [ D _ {\mathrm{KL}} \big [ \pi_ {\theta_ {\mathrm{old}}} (\cdot | s _ {t}, g) \| \pi_ {\theta} (\cdot | s _ {t}, g) \big ] \Big ].
$$

Introducing a common penalty coefficient $\beta ,$ the GRPO surrogate objective is 

$$
L ^ {\mathrm{GRPO}} (\theta) = \sum_ {g \in \mathcal {G}} \omega_ {g} \mathbb {E} _ {t \sim g} \Big [ r _ {t} ^ {g} (\theta) \hat {A} _ {t} ^ {g} - \beta D _ {\mathrm{KL}} \big [ \pi_ {\theta_ {\mathrm{old}}} (\cdot | s _ {t}, g) \| \pi_ {\theta} (\cdot | s _ {t}, g) \big ] \Big ].\tag{13}
$$

In some cases, one may also include an inter-group regularizer to encourage policies for different groups to remain similar: 

$$
-\gamma \sum_{\substack{g,h\in \mathcal{G}\\ g\neq h}}\omega_{g,h}\mathbb{E}_{s\sim \mathcal{D}}\Bigl [D_{\mathrm{KL}}\bigl [\pi_{\theta}(\cdot \mid s,g)\parallel \pi_{\theta}(\cdot \mid s,h)\bigr ]\Bigr ],
$$

where $\gamma \geq 0$ and $\{ \omega _ { g , h } \}$ are pairwise weights. 

The GRPO algorithm proceeds analogously to PPO: 

1. For each $g \in { \mathcal { G } }$ , collect trajectories under $\pi _ { \theta _ { \mathrm { o l d } } } ( \cdot \ | \cdot , g )$ 

2. Compute group-specific advantages $\{ \hat { A } _ { t } ^ { g } \}$ 

3. For $K$ epochs, perform minibatch gradient ascent on $L ^ { \mathrm { G R P O } } ( \theta )$ (θ): 

$$
\theta \leftarrow \theta + \eta \nabla_ {\theta} L ^ {\mathrm{GRPO}} (\theta).
$$

4. Optionally adjust $\beta$ (and $\gamma ,$ , if used) to keep each $D _ { D _ { \mathrm { K I } } } ^ { g }$ (and inter-group KLs) near desired targets. 

5. Update $\theta _ { \mathrm { o l d } }  \theta$ and repeat. 

## A.3 TRAINING VARIANCE DECOMPOSITION

The proof of Eq. 2 is deferred to A.5.1. Here we clarify why this decomposition is well-motivated and how it supports the P-POTS derivation (§3.1.1). 

Effect of Alternative decompositions. Our decomposition is obtained by iteratively conditioning on the random variables $( x _ { 0 } , t , x _ { t } )$ , so different conditioning orders give different decompositions. The order adopted in this paper $x _ { 0 }  t  x _ { y }$ is the only interpretable one, because it aligns with the MDM training procedure and matches $\mathbf { A R } \mathbf { \bar { M } s } ^ { \prime }$ variance decomposition. Importantly, the resulting form of P-POTS in $\ S 3 . 1 . 1$ is invariant to the conditioning order. This is because the optimal sampler $p ^ { * } ( t )$ depends only on $\mathbb { E } [ l ^ { 2 } \mid t ]$ , not on how the total variance is partitioned. 

Loss Variance vs. Gradient Variance. While gradient variance is most directly tied to optimization stability, we focus on loss variance for three reasons. (1) it is is optimizer-agnostic: adaptive optimizers rescale gradients, making “raw” gradient variance dependent on the optimizer, whereas $\mathrm { V a r } ( l )$ directly improves the SNR of the Monte Carlo estimate of the training objective. (2) Gradients are high-dimensional (e.g., a 8B-model has a gradient of dimension 8B), so any notion of “gradient variance” requires a scalarization $( e . g . , \mathbb { E } \| \mathbf { g } \| _ { 2 } ^ { 2 } )$ . Yet this is not theoretically neutral: different parameters may prefer different sampling distributions. Any scalarization therefore introduce extra design choices and breaks the clean optimality claim. In contrast, the loss is already a unified scalar with a clear interpretation as the objective whose expectation we estimate, so its variance is unambiguous. (3) A t-sampler can reliably control only the conditional variance of the loss: by first-order linearization, ${ \mathrm { V a r } } ( \mathbf { g } \mid t ) \approx J _ { t } { \mathrm { V a r } } ( l \mid t ) J _ { t } ^ { \top } \ +$ (terms from model sensitivity), where $J _ { t }$ is the Jacobian w.r.t. parameters. This highlights that part of gradient variance comes from model sensitivity (the Jacobian), which cannot be affected by t-sampler. The only reliably controllable component is $\mathrm { V a r } ( l \mid t )$ (holding fixed over short horizons). 

## A.4 ALGORITHMS

## A.4.1 BIN-WISE EMA

We partition the interval $t \in [ 0 , 1 ]$ into m equal bins and fix an EMA learning rate η. For each bin $j ,$ we maintain four EMA-tracked quantities: 

$$
\mu_ {L} (j) \approx \mathbb {E} [ \ell_ {i} \mid t \in \mathrm{bin} j ], \quad \mu_ {H} (j) \approx \mathbb {E} [ \tilde {b} _ {j} ], \quad M _ {L H} (j) \approx \mathbb {E} [ \ell_ {i} \tilde {b} _ {j} ], \quad M _ {H H} (j) \approx \mathbb {E} [ \tilde {b} _ {j} ^ {2} ],
$$

and a per-bin baseline estimate $\tilde { b } _ { j }$ , initialized to zero. On each training step: 

• Draw a batch $\{ x _ { i } \} _ { i = 1 } ^ { B }$ , sample mask ratios $t _ { i } \sim \mathcal { U } [ 0 , 1 ]$ , run the forward diffusion, and compute per-sample losses $\ell _ { i } .$ 

• Assign each t<sub>i</sub> to bin j via 

$$
j = f (t _ {i}) \quad \text { if } t _ {i} \in \left(\frac {j - 1}{m}, \frac {j}{m} \right].
$$

• For each sample, form an online estimate of the optimal control-variate coefficient 

$$
c _ {j} = \frac {M _ {L H} (j) - \mu_ {L} (j) \mu_ {H} (j)}{M _ {H H} (j) - \mu_ {H} (j) ^ {2} + \epsilon},
$$

where ϵ is a small damping constant. 

• Compute the adjusted loss 

$$
\ell_ {i} ^ {\mathrm{adj}} = \ell_ {i} - c _ {j} \cdot \tilde {b} _ {j},
$$

detaching $\tilde { b } _ { j }$ so that gradients flow only through $\ell _ { i } .$ , and backpropagate on $\ell _ { i } ^ { \mathrm { a d j } }$ 

• Update all four EMA statistics for bin $j \colon$ 

$$
\mu_ {L} (j) \leftarrow (1 - \eta) \mu_ {L} (j) + \eta \ell_ {i},
$$

$$
\mu_ {H} (j) \leftarrow (1 - \eta) \mu_ {H} (j) + \eta \tilde {b} _ {j},
$$

$$
M _ {L H} (j) \leftarrow (1 - \eta) M _ {L H} (j) + \eta (\ell_ {i} \tilde {b} _ {j}),
$$

$$
M _ {H H} (j) \leftarrow (1 - \eta) M _ {H H} (j) + \eta (\tilde {b} _ {j} ^ {2}).
$$

• Finally, update the baseline itself via 

$$
\tilde {b} _ {j} \leftarrow (1 - \eta) \tilde {b} _ {j} + \eta \ell_ {i}.
$$

This procedure produces a piecewise-constant approximation $\tilde { \mathbf { b } }$ of the true expected loss $b ( t )$ while dynamically choosing, per bin, the coefficient $c _ { j }$ that maximally cancels out baseline-related noise– yielding tighter variance reduction than a fixed scaling $e . g . , c = 1$ . Additionally, this “per-bin” EMA is updated per GPU batch and is analogous to standard moving-average updates used elsewhere in deep learning. 

## A.5 PROOFS

## A.5.1 TRAINING VARIANCE DECOMPOSITION

The law of total variance states that 

$$
\operatorname{Var} (Y) = \mathbb {E} \bigl [ \operatorname{Var} (Y \mid X) \bigr ] + \operatorname{Var} \bigl (\mathbb {E} [ Y \mid X ] \bigr).\tag{14}
$$

Let $Y = l ( x _ { 0 } , t , x _ { t } )$ where $x _ { 0 } \sim p _ { \mathrm { d a t a } } ( . ) , t \sim U [ 0 , 1 ] , x _ { t } \sim q ( x _ { t } \mid x _ { 0 } , t )$ are three random variables. We apply the law of total variance twice: 

## 1. First Decomposition over x<sub>0</sub>

View $x _ { 0 }$ as the outer random variable. By the law of total variance, 

$$
\operatorname{Var} _ {x _ {0}, t, x _ {t}} (Y) = \mathbb {E} _ {x _ {0}} \left[ \operatorname{Var} _ {t, x _ {t}} (Y \mid x _ {0}) \right] + \operatorname{Var} _ {x _ {0}} \left[ \mathbb {E} _ {t, x _ {t}} (Y \mid x _ {0}) \right].
$$

Define 

$$
g _ {\theta} (x _ {0}, t) = \mathbb {E} _ {x _ {t}} [ \ell_ {\theta} (x _ {0}, t, x _ {t}) \mid x _ {0}, t ], \quad h _ {\theta} (x _ {0}) = \mathbb {E} _ {t, x _ {t}} [ \ell_ {\theta} (x _ {0}, t, x _ {t}) \mid x _ {0} ] = \mathbb {E} _ {t} [ g _ {\theta} (x _ {0}, t) \mid x _ {0} ].
$$

Then the second term becomes 

$$
\operatorname{Var} _ {x _ {0}} \left[ h _ {\theta} \left(x _ {0}\right) \right] = \operatorname{Var} _ {x _ {0}} \left(\mathbb {E} _ {t} \left[ g _ {\theta} \left(x _ {0}, t\right) \right]\right) \quad \text {© data variance.}
$$

## 2. Second decomposition inside $\mathbb { E } _ { x _ { 0 } } [ \mathrm { V a r } _ { t , x _ { t } } ( Y \mid x _ { 0 } ) ]$

We apply the law of total variance again, this time conditioning on t: 

$$
\operatorname{Var} _ {t, x _ {t}} (Y \mid x _ {0}) = \mathbb {E} _ {t} \left[ \operatorname{Var} _ {x _ {t}} (Y \mid x _ {0}, t) \right] + \operatorname{Var} _ {t} \left[ \mathbb {E} _ {x _ {t}} (Y \mid x _ {0}, t) \right].
$$

Note that $\mathbb { E } _ { x _ { t } } ( Y \mid x _ { 0 } , t ) = g _ { \theta } ( x _ { 0 } , t )$ , so: 

$$
\begin{array}{l} \mathbb {E} _ {x _ {0}} [ \operatorname{Var} _ {t, x _ {t}} (Y \mid x _ {0}) ] = \mathbb {E} _ {x _ {0}} \left[ \mathbb {E} _ {t} \left[ \operatorname{Var} _ {x _ {t}} (\ell_ {\theta} \mid x _ {0}, t) \right] \right] + \mathbb {E} _ {x _ {0}} \left[ \operatorname{Var} _ {t} (g _ {\theta} (x _ {0}, t) \mid x _ {0}) \right], \\ = \underbrace {\mathbb {E} _ {x _ {0} , t} \left[ \operatorname{Var} _ {x _ {t}} (\ell_ {\theta} \mid x _ {0} , t) \right]} _ {\texttt {A} \texttt {m a s k p a t t e r n n o i s e}} + \underbrace {\mathbb {E} _ {x _ {0}} \left[ \operatorname{Var} _ {t} (g _ {\theta} (x _ {0} , t) \mid x _ {0}) \right]} _ {\texttt {B} \texttt {m a s k r a t e n o i s e}}. \end{array}
$$

## 3. Combine Everything

Putting steps 1 and 2 together, we have: 

$$
\operatorname{Var} _ {x _ {0}, t, x _ {t}} (\ell_ {\theta}) = \underbrace {\mathbb {E} _ {x _ {0} , t} \left[ \operatorname{Var} _ {x _ {t}} (\ell_ {\theta} \mid x _ {0} , t) \right]} _ {\text {A}} + \underbrace {\mathbb {E} _ {x _ {0}} \left[ \operatorname{Var} _ {t} (g _ {\theta} (x _ {0} , t) \mid x _ {0}) \right]} _ {\text {B}} + \underbrace {\operatorname{Var} _ {x _ {0}} \left(\mathbb {E} _ {t} [ g _ {\theta} (x _ {0} , t) ]\right)} _ {\text {C}},
$$

which is exactly the claimed decomposition Eq. 2. 

## A.5.2 P-POTS: PARETO-OPTIMAL PROPERTY

Consider the reweighted estimator 

$$
\frac {1}{p (t)} l _ {\theta} (x _ {0}, t, x _ {t}), \quad t \sim p (t).
$$

As shown in Eq. 3, this estimator is unbiased for the training objective in Eq. 1. We now give a detailed proof that 

$$
p (t) = \frac {\sqrt {g ^ {2} (t) + v (t)}}{\int_ {0} ^ {1} \sqrt {g ^ {2} (s) + v (s)} d s}
$$

is the theoretically optimal distribution that minimizes $\textcircled { A } + \textcircled { B } + \textcircled { C }$ 

Recall $\operatorname { V a r } [ X ] = \mathbb { E } [ X ^ { 2 } ] - \mathbb { E } [ X ] ^ { 2 }$ . Then 

$$
Ⓐ + ⓐ + Ⓒ = \operatorname{Var} _ {x _ {0}, t, x _ {t}} \left(\frac {1}{p (t)} l _ {\theta} (x _ {0}, t, x _ {t})\right).
$$

Expanding yields 

$$
\mathbb {E} _ {x _ {0}, t, x _ {t}} \Big [ \frac {1}{p ^ {2} (t)} l _ {\theta} ^ {2} (x _ {0}, t, x _ {t}) \Big ] - \mathbb {E} _ {x _ {0}, t, x _ {t}} ^ {2} \Big [ \frac {1}{p (t)} l _ {\theta} (x _ {0}, t, x _ {t}) \Big ].
$$

Assuming absolute convergence and finite expectations, Fubini’s theorem allows us to exchange the order of integration, giving 

$$
\int_ {0} ^ {1} \frac {1}{p (t)} \int_ {x _ {0} \sim D} \int_ {X _ {t}} l _ {\theta} ^ {2} \left(x _ {0}, t, x _ {t}\right) d x _ {t} d x _ {0} d t - \left(\int_ {0} ^ {1} \int_ {x _ {0} \sim D} \int_ {X _ {t}} l _ {\theta} \left(x _ {0}, t, x _ {t}\right) d x _ {t} d x _ {0} d t\right) ^ {2}.
$$

Define 

$$
g (t) := \mathbb {E} _ {x _ {0}, x _ {t}} [ l _ {\theta} \mid t ], \qquad v (t) := \operatorname{Var} _ {x _ {0}, x _ {t}} [ l _ {\theta} \mid t ].
$$

Substituting yields 

$$
\int_ {0} ^ {1} \frac {g ^ {2} (t) + v (t)}{p (t)} d t - \left(\int_ {0} ^ {1} g (t) d t\right) ^ {2}.
$$

The second term is independent of $p ( t )$ . Thus the optimization problem reduces to 

$$
\min _ {p} J [ p ] := \int_ {0} ^ {1} \frac {g ^ {2} (t) + v (t)}{p (t)} d t, \quad \text { s.t. } p (t) \geq 0, \int_ {0} ^ {1} p (t) d t = 1.
$$

Lagrangian method. Form the Lagrangian: 

$$
\mathcal {L} [ p, \lambda ] = \int_ {0} ^ {1} \Bigl (\frac {g ^ {2} (t) + v (t)}{p (t)} + \lambda p (t) \Bigr) d t - \lambda ,
$$

where λ enforces normalization. Taking the variation w.r.t. $p ( t )$ gives 

$$
- \frac {g ^ {2} (t) + v (t)}{p (t) ^ {2}} + \lambda = 0 \quad \Longrightarrow \quad p ^ {*} (t) = \sqrt {\frac {g ^ {2} (t) + v (t)}{\lambda}}.
$$

Normalization requires 

$$
\sqrt {\lambda} = \int_ {0} ^ {1} \sqrt {g ^ {2} (s) + v (s)} d s.
$$

Final result. Therefore, the unique minimizer is 

$$
\boxed {p ^ {*} (t) = \frac {\sqrt {g ^ {2} (t) + v (t)}}{\int_ {0} ^ {1} \sqrt {g ^ {2} (s) + v (s)} d s}}
$$

which achieves the Pareto-optimal balance by minimizing $\textcircled{4} + \textcircled { B } + \textcircled { C }$ 

## A.5.3 IMPORTANCE SAMPLING

## Part 1: Unbiasedness of the Importance-Sampling Estimator

Suppose we wish to estimate 

$$
I = \int f (x) p (x) d x,
$$

but drawing samples from $p ( x )$ directly is difficult. Instead, we draw $x _ { 1 } , \ldots , x _ { N }$ i.i.d. from a proposal distribution $q ( x )$ , which satisfies $q ( x ) > 0$ whenever $f ( x ) p ( x ) \neq 0$ , and construct the estimator 

$$
\hat {I} _ {N} = \frac {1}{N} \sum_ {i = 1} ^ {N} w (x _ {i}) f (x _ {i}), \quad \text { where } w (x) = \frac {p (x)}{q (x)}.
$$

We check its expectation under the sampling law $q \colon$ 

$$
\begin{array}{l} \mathbb {E} _ {q} [ \hat {I} _ {N} ] = \frac {1}{N} \sum_ {i = 1} ^ {N} \mathbb {E} _ {q} [ w (x _ {i}) f (x _ {i}) ] \quad (\text { by   linearity   and   identical   distribution }) \\ \quad = \mathbb {E} _ {q} [ w (X) f (X) ] \quad (\text { drop   index }) \\ \quad = \int \frac {p (x)}{q (x)} f (x) q (x) d x = \int f (x) p (x) d x = I. \end{array}
$$

Hence, $\hat { I } _ { N }$ is an unbiased estimator of I. 

## Part 2: Choice of $q$ to Minimize Variance

The variance of $\hat { I } _ { N }$ under q is 

$$
\mathrm{Var} _ {q} (\hat {I} _ {N}) = \frac {1}{N} \mathrm{Var} _ {q} [ w (X) f (X) ] = \frac {1}{N} \left(\mathbb {E} _ {q} [ w (X) ^ {2} f (X) ^ {2} ] - I ^ {2}\right).
$$

Since I is fixed, minimizing $\mathrm { V a r } _ { q } ( \hat { \cal I } _ { N } )$ amounts to minimizing 

$$
\mathbb {E} _ {q} [ w (X) ^ {2} f (X) ^ {2} ] = \int {\frac {p (x) ^ {2}}{q (x)}} f (x) ^ {2} d x,
$$

subject to the constraint $\textstyle \int q ( x ) d x = 1$ 

Introduce a Lagrange multiplier $\lambda ,$ and consider the functional 

$$
\mathcal {L} [ q ] = \int \frac {p (x) ^ {2} f (x) ^ {2}}{q (x)} d x + \lambda \left(\int q (x) d x - 1\right).
$$

Taking the functional derivative with respect to $q ( x )$ and setting it to zero: 

$$
- \frac {p (x) ^ {2} f (x) ^ {2}}{q (x) ^ {2}} + \lambda = 0 \Rightarrow q (x) \propto p (x) | f (x) |.
$$

Thus, the optimal proposal distribution is 

$$
q ^ {*} (x) = \frac {| f (x) | p (x)}{\int | f (u) | p (u) d u},
$$

which, for nonnegative $f ,$ minimizes the variance to its theoretical minimum–zero in the ideal continuous-sampling limit. 

## A.5.4 SYRM

SyRM is designed for structured data such as HTML tables, source code, or graphs. We categorize tokens into two groups: 1) response tokens R, and 2) syntax tokens $C$ in the prompt. For example, in an html table, the syntax tokens are <table>, <tr>, <th>, <td>, and similar tags. SyRM modifies the eligibility set for masking as follows: 

$$
\text { eligible } := \text { response   tokens } \cup \text { syntax   tokens   in   prompt }
$$

## Definition of Eligible Tokens

We define a token position as eligible if it can be masked during the forward process. According to LLaDA(Nie et al., 2025), during pretraining, all tokens in $\mathbf { x } _ { \mathrm { 0 } }$ are eligible. During supervised finetuning, where $\mathbf { x } _ { \mathrm { 0 } }$ is composed of concatenated prompt and response tokens, only response tokens are eligible. Thus, in this stage, prompt tokens remain unmasked while response tokens are masked independently. 

## Under the following assumptions,

• Tokens within each group are homogeneous in first and second moments of token-level losses. 

• Syntax tokens are stabler predicted than response tokens. 

• Reasoning errors couple response tokens more strongly than syntax tokens. 

we show below that SyRM rigorously reduces the masking pattern noise A . Although this introduces some bias (optimum shift), we show this bias is bounded, and the total MSE of estimating training loss is reduced due to significant variance reduction. Intuitively, by including syntax tokens in prompts that are both easy to predict and helpful for guiding the model’s reasoning as an auxiliary loss, SyRM reduces A mask pattern noise. 

## Setup

• Number of eligible positions: $P \in \mathbb { N } .$ 

• Mask variables: $M _ { i } \stackrel { \mathrm { i . i . d . } } { \sim }$ Bernoulli(t) with $0 < t < 1$ 

• Per-token loss: random variables $Y _ { i }$ satisfying 

$$
\mathbb {E} [ Y _ {i} ] = \mu_ {i}, \qquad \operatorname{Var} (Y _ {i}) = \sigma_ {i} ^ {2}, \qquad \operatorname{Cov} (Y _ {i}, Y _ {j}) = \rho_ {i j} (i \neq j).
$$

• Batch loss: 

$$
L = \frac {1}{P t} \sum_ {i = 1} ^ {P} M _ {i} Y _ {i}.
$$

• The $M _ { i } { ' } \mathrm { s }$ are independent of all $Y _ { j }$ , and the $M _ { i } { ' } \mathrm { s }$ are mutually independent. 

Theorem 1 

$$
\operatorname{Var} [ L \mid t ] = \frac {1}{P ^ {2} t} \sum_ {i = 1} ^ {P} \sigma_ {i} ^ {2} + \frac {1 - t}{P ^ {2} t} \sum_ {i = 1} ^ {P} \mu_ {i} ^ {2} + \frac {2}{P ^ {2}} \sum_ {1 \leq i <   j \leq P} \rho_ {i j}.
$$

Proof Let $\begin{array} { r } { S : = \sum _ { i = 1 } ^ { P } M _ { i } Y _ { i } } \end{array}$ , so $L = S / ( P t )$ . 

Then 

$$
\operatorname{Var} (S) = \sum_ {i = 1} ^ {P} \operatorname{Var} \left(M _ {i} Y _ {i}\right) + 2 \sum_ {1 \leq i <   j \leq P} \operatorname{Cov} \left(M _ {i} Y _ {i}, M _ {j} Y _ {j}\right).\tag{15}
$$

where 

$$
\mathrm{Var} (M _ {i} Y _ {i}) = \mathbb {E} [ M _ {i} Y _ {i} ^ {2} ] - (\mathbb {E} [ M _ {i} Y _ {i} ]) ^ {2} = t \mathbb {E} [ Y _ {i} ^ {2} ] - t ^ {2} \mu_ {i} ^ {2} = t \sigma_ {i} ^ {2} + t (1 - t) \mu_ {i} ^ {2}.\tag{16}
$$

and 

$$
\operatorname{Cov} \left(M _ {i} Y _ {i}, M _ {j} Y _ {j}\right) = \mathbb {E} \left[ M _ {i} M _ {j} Y _ {i} Y _ {j} \right] - \mathbb {E} \left[ M _ {i} Y _ {i} \right] \mathbb {E} \left[ M _ {j} Y _ {j} \right] = t ^ {2} \left(\rho_ {i j} + \mu_ {i} \mu_ {j}\right) - t ^ {2} \mu_ {i} \mu_ {j} = t ^ {2} \rho_ {i j}.\tag{17}
$$

Combine Eqs. 15, 16 and 17: 

$$
\operatorname{Var} (S) = t \sum_ {i = 1} ^ {P} \sigma_ {i} ^ {2} + t (1 - t) \sum_ {i = 1} ^ {P} \mu_ {i} ^ {2} + 2 t ^ {2} \sum_ {1 \leq i <   j \leq P} \rho_ {i j}.\tag{18}
$$

Convert Eq. 18 back to $\begin{array} { r } { L = { \frac { S } { P t } } } \end{array}$ 

$$
\begin{array}{l} \operatorname{Var} [ L \mid t ] = \frac {1}{(P t) ^ {2}} \operatorname{Var} (S) \\ \qquad = \frac {1}{P ^ {2} t} \sum_ {i = 1} ^ {P} \sigma_ {i} ^ {2} + \frac {1 - t}{P ^ {2} t} \sum_ {i = 1} ^ {P} \mu_ {i} ^ {2} + \frac {2}{P ^ {2}} \sum_ {1 \leq i <   j \leq P} \rho_ {i j}. \end{array}
$$

Theorem 2 Let $S _ { \mathrm { C } }$ be the SyRM strategy and $S _ { \mathrm { R } }$ the baseline strategy that masks only response tokens. Define the mask-pattern noise of a strategy $S$ as $A ( S ) : = \mathrm { V a r } [ \bar { L } \mid S ]$ . For $m \in \{ \dot { \mathrm { C } } , \mathrm { R } \}$ with corresponding eligible-set size $P _ { m }$ and token statistics $\mu _ { i , m } , \sigma _ { i , m } ^ { 2 } , \rho _ { i j , m } ,$ 

$$
A (S _ {m}) = \frac {\mathbb {E} [ 1 / t ]}{P _ {m} ^ {2}} \sum_ {i = 1} ^ {P _ {m}} \sigma_ {i, m} ^ {2} + \frac {\mathbb {E} [ (1 - t) / t ]}{P _ {m} ^ {2}} \sum_ {i = 1} ^ {P _ {m}} \mu_ {i, m} ^ {2} + \frac {2}{P _ {m} ^ {2}} \sum_ {1 \leq i <   j \leq P _ {m}} \rho_ {i j, m}.
$$

Proof Take the conditional variance from Theorem 1 and integrate over t. Note that token-wise parameters naturally depend on the chosen strategy, so we let $P , \sigma _ { i } , \mu _ { i } , \rho _ { i j }$ depend on $m \in \{ \mathrm { C } , \mathrm { R } \}$ 

Theorem 3 Partition the eligible tokens into 

1. response tokens R (count $P _ { R } )$ 

2. syntax prompt tokens C (count $P _ { C } )$ 

so $P = P _ { R } + P _ { C }$ . Under the assumptions 

(A1) All tokens in R share $\mu _ { R } , \sigma _ { R } ^ { 2 }$ ; tokens in C share $\mu _ { C } , \sigma _ { C } ^ { 2 }$ 

(A2) $\sigma _ { C } ^ { 2 } \ll \sigma _ { R } ^ { 2 }$ 

(A3) $\rho _ { C C } \ll \rho _ { R R } .$ 

(A4) $| \rho _ { R C } | \leq { \sqrt { \rho _ { R R } \rho _ { C C } } } .$ 

(A5) 2 $w _ { R } \beta \leq \frac { ( 1 - \alpha ) ( P _ { R } - 1 ) \rho _ { R R } } { \sigma _ { _ B } ^ { 2 } B } + ( 1 - \beta )$ , where $\alpha : = \sigma _ { C } ^ { 2 } / \sigma _ { R } ^ { 2 } , \beta : = \rho _ { C C } / \rho _ { R R } .$ $w _ { R } : = P _ { R } / P _ { \mathrm { ~ : ~ } }$ , and $B : = \dot { \mathbb { E } } [ 1 / t ] > 0$ 

we have 

$$
A (S _ {\mathrm{C}}) <   A (S _ {\mathrm{R}}).
$$

Proof Denote $B : = \mathbb { E } \big [ \frac { 1 } { t } \big ] > 0$ . By Theorem 2, 

$$
A \big (S _ {\mathrm{R}} \big) = \frac {B}{P _ {R}} \sigma_ {R} ^ {2} + \frac {1}{P _ {R}} (B - 1) \mu_ {m} ^ {2} + \frac {P _ {R} - 1}{P _ {R}} \rho_ {R R}\tag{19}
$$

In the SyRM strategy, both R and C are eligible. Denote $\begin{array} { r } { w _ { R } = \frac { P _ { R } } { P } , w _ { C } = \frac { P _ { C } } { P } } \end{array}$ where $P = P _ { R } +$ $P _ { C } > \operatorname* { m a x } ( P _ { R } , P _ { C } )$ . We group terms accordingly, yielding 

$$
A (S _ {\mathrm{C}}) = \frac {\sigma_ {R} ^ {2}}{P _ {R} B} + \frac {\sigma_ {C} ^ {2}}{P _ {C} B} + \frac {P _ {R} - 1}{P} \rho_ {R R} + \frac {P _ {C} - 1}{P} \rho_ {C C} + \frac {2 P _ {R} P _ {C}}{P ^ {2}} \rho_ {R C} = w _ {R} A _ {R} + w _ {C} A _ {C} + 2 w _ {R} w _ {C} \rho_ {R C},\tag{20}
$$

where 

$$
A _ {R} := \frac {\sigma_ {R} ^ {2}}{P _ {R} B} + \frac {P _ {R} - 1}{P _ {R}} \rho_ {R R}, A _ {C} := \frac {\sigma_ {C} ^ {2}}{P _ {C} B} + \frac {P _ {C} - 1}{P _ {C}} \rho_ {C C}.
$$

Thus, 

$$
A (S _ {\mathrm{C}}) - A (S _ {\mathrm{R}}) = w _ {C} (A _ {C} - A _ {R}) + 2 w _ {R} w _ {C} \rho_ {R C}.
$$

Claim 3.1 

$$
A _ {R} > A _ {C}
$$

Proof By the Cauchy–Schwarz inequality, 

$$
| \rho_ {R C} | \leq \sqrt {\rho_ {R R} \rho_ {C C}}.
$$

From Eqs. 19 and 20, we further obtain: 

$$
A _ {R} - A _ {C} = \left(\frac {\sigma_ {R} ^ {2}}{P _ {R}} - \frac {\sigma_ {C} ^ {2}}{P _ {C}}\right) B + \left(\frac {P _ {R} - 1}{P _ {R}} \rho_ {R R} - \frac {P _ {C} - 1}{P _ {C}} \rho_ {C C}\right).\tag{21}
$$

Define: 

$$
\alpha := \frac {\sigma_ {C} ^ {2}}{\sigma_ {R} ^ {2}}, \quad \beta := \frac {\rho_ {C C}}{\rho_ {R R}} \quad (\text {   as   in   Assumption   A,   assume   } \alpha , \beta \ll 1),
$$

then: 

$$
A _ {R} - A _ {C} = \sigma_ {R} ^ {2} B \left(\frac {1}{P _ {R}} - \frac {\alpha}{P _ {C}}\right) + \rho_ {R R} \left(\frac {P _ {R} - 1}{P _ {R}} - \beta \cdot \frac {P _ {C} - 1}{P _ {C}}\right) \geq \sigma_ {R} ^ {2} B \cdot \frac {1 - \alpha}{P _ {R}} + \rho_ {R R} \cdot \frac {1 - \beta}{P _ {R}} \cdot \frac {P _ {R} - 1}{1}.\tag{22}
$$

Therefore, 

$$
A _ {R} > A _ {C}.
$$

Using the inequality 

$$
| \rho_ {R C} | \leq \sqrt {\rho_ {R R} \rho_ {C C}} \leq \beta \rho_ {R R},
$$

we obtain: 

$$
A \big (S _ {\mathrm{C}} \big) - A \big (S _ {\mathrm{R}} \big) \leq w _ {C} \underbrace {\left[ - (1 - \alpha) \frac {\sigma_ {R} ^ {2} B}{P _ {R}} - (1 - \beta) \frac {P _ {R} - 1}{P _ {R}} \rho_ {R R} \right]} _ {=: D} + 2 w _ {R} w _ {C} \beta \rho_ {R R}.
$$

Since $D > 0$ and $\beta \ll 1$ , setting 

$$
2 w _ {R} \beta \leq \frac {D}{\rho_ {R R}},
$$

ensures that 

$$
A (S _ {\mathrm{C}}) - A (S _ {\mathrm{R}}) <   0.
$$

## Discussion of Assumptions

• A1–A2. Syntax tokens C are homogeneous and more stably predicted, hence lower mean and variance. 

• A3. Reasoning errors couple response tokens more strongly than syntax tokens, implying ρ<sub>RR</sub> ≫ ρ<sub>CC</sub> . 

• A4. Cauchy–Schwarz bounds the cross-covariance. 

• A5. Given A2–A3, this technical condition is typically satisfied in practice and ensures the crossterm cannot offset the variance reduction provided by including C. 

Theorem 4 

$$
L _ {\mathrm{SyRM}} = \alpha L _ {\text { resp }} + (1 - \alpha) L _ {\text { coord }}
$$

Proof 4 Consider a single forward pass with $P _ { R }$ response tokens and $P _ { C }$ syntax tokens (disjoint sets, so $P = P _ { R } + P _ { C } )$ . All eligible positions are masked with the same rate $t \sim$ Uniform(0, 1). Define the per-step loss 

$$
L = \frac {1}{P t} \sum_ {i = 1} ^ {P} M _ {i} Y _ {i}, \quad M _ {i} \sim \operatorname{Bernoulli} (t),
$$

where $P$ is the number of eligible tokens under the chosen strategy. 

For the SyRM strategy, $P _ { \mathrm { S y R M } } = P _ { R } + P _ { C }$ , and 

$$
\begin{array}{l} L _ {\text {SyRM}} = \frac {1}{P _ {\text {SyRM}} t} \sum_ {i \in R \cup C} M _ {i} Y _ {i} \\ \qquad = \frac {P _ {R}}{P _ {\text {SyRM}}} \Big [ \frac {1}{P _ {R} t} \sum_ {i \in R} M _ {i} Y _ {i} \Big ] + \frac {P _ {C}}{P _ {\text {SyRM}}} \Big [ \frac {1}{P _ {C} t} \sum_ {j \in C} M _ {j} Y _ {j} \Big ] \\ \qquad := \alpha   L _ {\text {resp}} + (1 - \alpha)   L _ {\text {coord}}, \end{array}
$$

where 

Hence, 

$$
\alpha := \frac {P _ {R}}{P _ {R} + P _ {C}}, \qquad 1 - \alpha := \frac {P _ {C}}{P _ {R} + P _ {C}}.
$$

$$
L _ {\mathrm{SyRM}} = \alpha L _ {\text { resp }} + (1 - \alpha) L _ {\text { coord }},
$$

which is a purely algebraic identity requiring no additional assumptions. 

## Theorem 5 Optimum-shift bias is inevitable.

Proof Define the objectives $J _ { \mathrm { r e s p } } ( \theta ) \ = \ \mathbb { E } [ L _ { \mathrm { r e s p } } ] , \quad J _ { \mathrm { c o o r d } } ( \theta ) \ = \ \mathbb { E } [ L _ { \mathrm { c o o r d } } ] , \quad J _ { \mathrm { S y R M } } ( \theta ) \ =$ α $J _ { \mathrm { r e s p } } ( \theta ) + ( 1 - \alpha ) \bar { J _ { \mathrm { c o o r d } } } ( \theta )$ . Let the response-only optimum $\theta _ { \mathrm { r e s p } } ^ { \star }$ satisfy $\nabla _ { \theta } J _ { \mathrm { r e s p } } ( \theta _ { \mathrm { r e s p } } ^ { \star } ) = 0$ . If at the same point $\nabla _ { \theta } J _ { \mathrm { c o o r d } } ( \theta _ { \mathrm { r e s p } } ^ { \star } ) \neq 0$ , then 

$$
\nabla_ {\theta} J _ {\mathrm{SyRM}} (\boldsymbol {\theta} _ {\mathrm{resp}} ^ {\star}) = (1 - \alpha) \nabla_ {\theta} J _ {\mathrm{coord}} (\boldsymbol {\theta} _ {\mathrm{resp}} ^ {\star}) \neq 0,
$$

so $\theta _ { \mathrm { r e s p } } ^ { \star }$ is not a stationary point of $J _ { \mathrm { S y R M } }$ . Under gradient-based optimization, the parameters will continue to move and eventually converge to a different optimum $\theta _ { \mathrm { S y R M } } ^ { \bar { \star } } \neq \theta _ { \mathrm { r e s p } } ^ { \star }$ . Hence, whenever $\nabla _ { \theta } J _ { \mathrm { c o o r d } } ( \theta _ { \mathrm { r e s p } } ^ { \star } ) \neq 0$ , the two optima necessarily diverge; coincidence occurs iff $\nabla _ { \theta } J _ { \mathrm { c o o r d } } ( \theta _ { \mathrm { r e s p } } ^ { \star } ) =$ 0. 

Theorem 6 If the Hessian of $J _ { \mathrm { S y R M } }$ is positive definite, 

$$
\left\| \theta_ {\mathrm{SyRM}} ^ {\star} - \theta_ {\mathrm{resp}} ^ {\star} \right\| \leq \frac {1 - \alpha}{\lambda_ {\min} (H _ {\mathrm{SyRM}})} \left\| \nabla J _ {\mathrm{coord}} (\theta_ {\mathrm{resp}} ^ {\star}) \right\|.
$$

Proof Perform a first-order Taylor expansion of $\theta _ { \mathrm { S y R M } } ^ { \star }$ around $\theta _ { \mathrm { r e s p } } ^ { \star } .$ 

$$
0 = \nabla J _ {\mathrm{SyRM}} (\theta_ {\mathrm{SyRM}} ^ {\star}) \simeq \nabla J _ {\mathrm{SyRM}} (\theta_ {\text {resp}} ^ {\star}) + H _ {\mathrm{SyRM}} (\xi) (\theta_ {\mathrm{SyRM}} ^ {\star} - \theta_ {\text {resp}} ^ {\star}),
$$

where $H _ { \mathrm { S y R M } } ( \xi )$ is the Hessian evaluated at some point ξ between $\theta _ { \mathrm { S y R M } } ^ { \star }$ and $\theta _ { \mathrm { r e s p } } ^ { \star }$ . Rearranging $\mathrm { g i v e s } { \mathrm { : } }$ 

$$
\theta_ {\mathrm{SyRM}} ^ {\star} - \theta_ {\mathrm{resp}} ^ {\star} \simeq - (1 - \alpha) H _ {\mathrm{SyRM}} ^ {- 1} (\xi) \nabla J _ {\mathrm{coord}} (\theta_ {\mathrm{resp}} ^ {\star}).
$$

If $H _ { \mathrm { S y R M } } ( \xi )$ is positive definite, we can bound the Euclidean norm using its smallest eigenvalue $\lambda _ { \operatorname* { m i n } } ( H _ { \operatorname { S y R M } } )$ : 

$$
\left\| \theta_ {\mathrm{SyRM}} ^ {\star} - \theta_ {\mathrm{resp}} ^ {\star} \right\| \leq \frac {1 - \alpha}{\lambda_ {\min} (H _ {\mathrm{SyRM}})} \left\| \nabla J _ {\mathrm{coord}} (\theta_ {\mathrm{resp}} ^ {\star}) \right\|.
$$

## Explanation:

• Weight factor $1 - \alpha \mathrm { : }$ The smaller the proportion of syntax tokens, the tighter the bound. 

• Curvature $\lambda _ { \operatorname* { m i n } } ( H _ { \operatorname { S y R M } } )$ : A steeper loss (larger curvature) suppresses the shift. 

• Task conflict $\left\| \nabla J _ { \mathrm { c o o r d } } ( \theta _ { \mathrm { r e s p } } ^ { \star } ) \right\|$ : If the syntax gradient is already small at the resp-optimal point, the shift will naturally be small. 

## A.5.5 STRATIFIED SAMPLER

Specifically, when processing a batch of n data points, instead of independently drawing mask ratios $t _ { i } \sim \mathcal { U } [ 0 , 1 ]$ ], one partitions the interval [0, 1] into k uniform strata: $[ 0 , 1 / k ] , \dot { [ 1 / k , 2 / k ] } , \dots , [ ( k -$ $1 ) / k , 1 \big |$ , and samples one masking ratio from each stratum. This approach, known as the stratified sampler, is known to produce low discrepancy. Here, k is the number of strata, which must be chosen before training. 

This technique is discussed in Sahoo et al. (2024), where they use $n = k ( i . e .$ , one sample per stratum). Here, we provide a rigorous analysis and offer a practical guideline for selecting k. 

## A principled Method: Unbiased and Reduce Variance

Let 

$$
G (t) = g _ {\theta} (x _ {0}, t) = \mathbb {E} _ {x _ {t} \sim q (\cdot | x _ {0}, t)} [ \ell_ {\theta} (x _ {0}, t, x _ {t}) \mid x _ {0}, t ].
$$

For a fixed $x _ { 0 } ,$ , we are interested in the expected value of $G ( t )$ under $t \sim \mathrm { U n i f } [ 0 , 1 ] , i . e .$ 

$$
\mu = \mathbb {E} _ {t} [ G (t) ].
$$

During training, this is estimated by the empirical mean over a batch: 

$$
\hat {\mu} = \frac {1}{n} \sum_ {i = 1} ^ {n} G (t _ {i}).
$$

If $t _ { i } \sim \mathrm { i . i . d }$ . Unif[0, 1] for $i = 1 , \ldots , n$ , then by standard results, 

$$
\mathrm{Var} [ \hat {\mu} \mid x _ {0} ] = \frac {1}{n} \sigma_ {t} ^ {2}, \quad \mathrm{where} \sigma_ {t} ^ {2} = \mathrm{Var} _ {t} [ G (t) ].
$$

This corresponds to the $\textcircled{8}$ term in the variance decomposition: 

$$
\mathbb {E} _ {x _ {0}} [ \operatorname{Var} _ {t} (G) ].
$$

For stratified sampling, partition [0, 1] into k equal-length intervals: 

$$
I _ {j} = \left[ \frac {j - 1}{k}, \frac {j}{k}\right), \quad \text { for } j = 1, \dots , k,
$$

and sample $t _ { j } \sim \mathrm { U n i f } ( I _ { j } )$ independently within each stratum. Then the estimator becomes 

$$
\hat {\mu} = \frac {1}{k} \sum_ {j = 1} ^ {k} G (t _ {j}).
$$

This remains an unbiased estimator of $\mu ,$ and its conditional variance is: 

$$
\mathrm{Var} [ \hat {\mu} \mid x _ {0} ] = \frac {1}{k ^ {2}} \sum_ {j = 1} ^ {k} \mathrm{Var} _ {t \in I _ {j}} [ G (t) ] = \frac {1}{k} \sigma_ {w} ^ {2},
$$

where we define the within-stratum variance: 

$$
\sigma_ {w} ^ {2} = \frac {1}{k} \sum_ {j = 1} ^ {k} \mathrm{Var} _ {t \in I _ {j}} [ G (t) ].
$$

Decomposing the total variance $\sigma _ { t } ^ { 2 }$ into within- and between-stratum variance gives: 

$$
\sigma_ {t} ^ {2} = \frac {n}{k} (\sigma_ {w} ^ {2} + \sigma_ {b} ^ {2}), \quad \text { where } \sigma_ {b} ^ {2} = \operatorname{Var} _ {k} [ \mu_ {k} ], \quad \mu_ {k} = \mathbb {E} _ {t \in I _ {k}} [ G (t) ].
$$

Therefore, the variance of the stratified estimator satisfies: 

$$
\mathrm{Var} _ {\mathrm{strat}} [ \hat {\mu} \mid x _ {0} ] = \frac {1}{n} \sigma_ {t} ^ {2} - \frac {1}{k} \sigma_ {b} ^ {2},
$$

which is smaller than the SRS variance: 

$$
\mathrm{Var} _ {\mathrm{SRS}} = \frac {1}{n} \sigma_ {t} ^ {2}.
$$

The reduction is exactly 

$$
\frac {1}{k} \sigma_ {b} ^ {2}.
$$

## Practical Suggestion for Choosing n

To choose an appropriate number of strata, we aim to balance two objectives: 

1. Reduce the overall estimation variance (which favors using a larger number of strata n); 

–denoted by $\hat { \sigma } _ { k } ^ { 2 } { - } \mathrm { i s }$ sufficiently stable (which favors having more samples per stratum, $i . e . , m = \operatorname* { i n t } ( B / n ) )$ . 

While setting $n = B$ results in highly fine-grained strata, it may lead to unstable estimates within each stratum due to too few samples. To balance this trade-off, we propose the following optimal number of strata: 

$$
n _ {\mathrm{opt}} \propto \sqrt {B},
$$

where B is the total sampling budget, and $m = \operatorname { i n t } ( B / n )$ denotes the number of samples per stratum. 

## A.5.6 BIN-WISE EMA: VARIANCE REDUCTION & HYPERPARAMETER SELECTION

Bin-Wise EMA Reduces $\textcircled{8}$ as a Control-Variate Strategy To estimate a random variable X with mean $\mu _ { X } = \operatorname { \mathbb { E } } [ X ]$ , we can use a control variate Y with known mean $\mu _ { Y } = \mathbb { E } [ Y ]$ . Given n i.i.d. samples $\{ ( X _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { \bar { n } }$ , the control variate estimator 

$$
\mu_ {C V} = \bar {X} - \frac {\operatorname{Cov} (X , Y)}{\sigma_ {Y} ^ {2}} \bar {Y}
$$

reduces the variance from $\frac { \sigma _ { X } ^ { 2 } } { n }$ to $\textstyle { \frac { \sigma _ { X } ^ { 2 } } { n } } ( 1 - \rho ^ { 2 } )$ , where $\begin{array} { r } { \rho = \frac { \mathrm { C o v } ( X , Y ) } { \sigma _ { X } \sigma _ { Y } } } \end{array}$ . Importantly, this form of estimator gives an unbiased estimate of loss gradients. 

In our context, X represents the loss gradient for the current batch, while $Y$ is the exponential moving average $b _ { k }$ of past gradients within the same t-bin. Assuming strong heteroscedasticity over $t \in [ 0 , \bar { 1 } ]$ , partitioning t into bins and applying EMA within each bin maximizes $| \rho |$ , thus achieving significant variance reduction. Several alternative control strategies are listed in Table 3. Among these, we adopt bin-wise EMA for its strong variance reduction under heteroscedastic conditions and minimal implementation burden, which also helps prevent overfitting. 

<table><tr><td>Variant</td><td>Assumptions</td><td>Tuning and Resources</td><td>Characteristics</td></tr><tr><td>Spline</td><td>Smooth <eq>\ell(t)</eq>, no sharp changes</td><td>Requires degree/node selection; low memory usage</td><td>Captures trends well but sensitive to under-/over-fitting</td></tr><tr><td>Kernel</td><td>Flexible; handles arbitrary heteroscedasticity</td><td>Only bandwidth <eq>h</eq>; stores <eq>O(1/h)</eq> samples; higher memory cost than bin-wise EMA</td><td>Most robust; no shape assumptions; tuning and implementation are more complex</td></tr></table>


Table 3: Comparison of spline and kernel variants


Choose the Number of Bins in EMA: A Mathematical Perspective Suppose we have m data points and choose b bins; then each bin will contain approximately $m / b$ samples to estimate the piecewise-constant value of $b ( t )$ over an interval of width $1 / b$ 

We use MSE optimization analysis to determine a principled choice of b given m. Specifically, we aim to estimate the function 

$$
g (t) = \mathbb {E} _ {x _ {t}} [ \ell_ {\theta} (x _ {0}, t, x _ {t}) \mid x _ {0}, t ]
$$

on [0, 1] by a piecewise-constant approximation $g _ { b } ( t )$ with b equal-width bins of size $\Delta = 1 / b ,$ using the sample average within each bin. 

The total mean squared error is decomposed as 

$$
\mathrm{MSE} \textcircled {B} = \underbrace {\mathbb {E} [ (g _ {b} (t) - g (t)) ^ {2} ]} _ {\mathrm{Bias} ^ {2}} + \underbrace {\mathrm{Var} (g _ {b} (t))} _ {\mathrm{Variance}}.
$$

Bias Term Assuming $g ( t )$ is twice differentiable within each bin, we can apply Taylor approximation: 

$$
\mathrm{Bias} ^ {2} \approx C _ {1} \Delta^ {4} = \frac {C _ {1}}{b ^ {4}},
$$

where $C _ { 1 } \propto$ max $| g ^ { \prime \prime } ( t ) |$ 

Variance Term Each bin contains $n = m / b$ samples, hence: 

$$
\mathrm{Var} \approx \frac {C _ {2}}{n} = \frac {C _ {2} b}{m},
$$

where $C _ { 2 } \propto \mathrm { V a r } _ { \mathrm { m a s k } } [ \ell ]$ 

Total MSE Combining both terms: 

$$
\mathrm{MSE} Ⓑ \approx \frac {C _ {1}}{b ^ {4}} + \frac {C _ {2} b}{m}.
$$

Optimal b Taking the derivative w.r.t. b and setting it to zero: 

$$
\frac {d}{d b} \operatorname{MSE} \widehat {\mathcal {B}} = - 4 C _ {1} b ^ {- 5} + \frac {C _ {2}}{m} = 0 \quad \Rightarrow \quad b ^ {*} = \left(\frac {4 C _ {1} m}{C _ {2}}\right) ^ {1 / 5} \propto m ^ {1 / 5}.
$$

Selection for $\eta$ For $\eta ,$ it’s useful to think in terms of the e-folding time $\begin{array} { r } { k = - \frac { 1 } { \ln ( 1 - \eta ) } \approx \frac { 1 } { \eta } . } \end{array}$ , so that each EMA bin effectively averages over the past $k \times$ (batch size per GPU) losses. Larger η adapts faster but use fewer past samples to estimate $b ( t )$ ; smaller η is stabler but can cause the EMA to lag (and suffer from domain-shift issues). In our experiments, we set: 

$$
m \times \left(\frac {1}{\eta} \times \text { batch   size   per   GPU }\right) \approx 0. 1 \times (\text { train   data   size }).
$$

## A.6 METHOD COMBINATION

One might wonder whether all methods proposed in this work could be combined into a single ultra-strong approach. While such a combination may sound attractive, it is not a straightforward task: different methods rely on distinct assumptions, and naive aggregation can easily introduce redundancy or bias: 

1. Different Assumptions. Each method relies on specific assumptions. For example, the effectiveness of P-POTS depends on accurate modeling of $g ( t )$ and v(t), while Bin-Wise EMA requires heteroskedasticity of the loss l with respect to t. Blindly combining them risks violating these assumptions. 

2. Redundancy. Each method in this work is already effective on its own. If two methods target the same source, the first may remove most of the variance (e.g., 80%), while the second may introduce estimation bias that offset any residual variance reduction, resulting in worse performance than using a single method. 

Instead, we emphasize the synergy between P-POTS and MIRROR, which our experiments confirm to be stable. In practice, we recommend P-POTS+MIRROR for maximal performance gains, and P-POTS alone for cost efficiency. Other techniques may still be preferable in specific contexts due to their simplicity or ease of implementation. Rather than combining everything, we provide guidance below on how to select among them. 

## A.6.1 STRATIFIED SAMPLING VS EMA

Both methods rely on the assumption that the loss l is heteroskedastic in t: after averaging over x<sub>0</sub> and $x _ { t } ,$ , different values of t yield losses of different magnitudes. This generally holds, since larger t masks more tokens, making reconstruction harder and the loss higher, as shown in Figure 5a. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/f0eb91ff-cd64-48a5-ae07-430a6a4f0bcb/2acdf237800dbeee27a2c3e60bab3547eb6cf0b908d3c55f802255aa6d9d32a9.jpg)



(a)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/f0eb91ff-cd64-48a5-ae07-430a6a4f0bcb/0b4b55288394f463a0339e5d59c092a33e8ea8a63d8932bb4656ca085bcfff21.jpg)



(b)



Figure 5: The left panel shows the heteroscedasticity of the t-loss, and the right panel illustrates the shrinking difference between IID and stratified sampling as batch size B increases, where we use the KS statistic to measure their maximum deviation.


However, the two methods reduce $\textcircled{8}$ in fundamentally different ways: 

• Batch Size: Stratified sampling enforces fixed proportions of t-values in each batch, which is particularly beneficial when the batch size is small. As the batch size grows, its advantage over simple independent sampling diminishes, as shown in Figure 5b. By contrast, EMA accumulates information continuously across training; its effectiveness depends on the choice of m and η, but not on batch size. 

• Theoretical Limit: Stratified sampling cannot fully eliminate variance, as within-stratum variance always remains. In contrast, EMA can, in principle, reduce B to zero if the control variate is sufficiently correlated. 

Note. We do not recommend combining EMA with stratified sampling. Each method is effective on its own, and their joint use may introduce estimation bias that offset any residual variance reduction. 

In practice, we suggest using stratified sampling as the default strategy for reducing $\textcircled{8} ,$ and switching to EMA only when the batch size is sufficiently large. 

## A.6.2 ISAD VS SYRM

These methods are dataset-specific. We recommend choosing based on the dataset type: ISAD is designed for QA datasets with explicit answer delimiters, while SyRM is designed for datasets containing syntax tokens in prompts. 

## A.7 EXPERIMENTAL DETAILS

## A.7.1 HYPERPARAMETER SETTINGS

We detail here the hyperparameters used for training and inference of masked diffusion models (MDMs) and auto-regressive models (ARMs). 

## Chain-of-Thought (CoT) Reasoning.

<table><tr><td>Dataset</td><td>CoT Handling</td></tr><tr><td>All except text-to-image-2M &amp; HiTab</td><td>Responses include chain-of-thoughts</td></tr><tr><td>HiTab</td><td>No CoT provided; generated via rejection sampling</td></tr></table>

## Train/Validation/Test Splits.

<table><tr><td>Dataset</td><td>Train/Validation Split</td><td>Test Split</td></tr><tr><td>OpenScience</td><td>5,000 samples, 9:1 (4,500/500)</td><td>5001st–6000th examples</td></tr><tr><td>GSM8K</td><td>Official train split</td><td>Official test split</td></tr><tr><td>HiTab</td><td>Official train split</td><td>Official test split</td></tr><tr><td>text-to-image-2M</td><td>5,000 samples, 9:1 (4,500/500)</td><td>5001st–5100th examples</td></tr></table>

text-to-image-2M consists of two subsets, 1024 10k and 512 2M. We sample 2,500 examples from each subset to construct the training and validation splits, and 500 examples from each subset to form the test split. 

## General Hyperparameter.

<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Learning rate</td><td><eq>5 \times 10^{-5}</eq></td></tr><tr><td>Scheduler</td><td>Linear</td></tr><tr><td>Global batch size</td><td>32</td></tr><tr><td>Epochs</td><td>5</td></tr><tr><td>Max sequence length</td><td>4096</td></tr><tr><td>Warmup steps</td><td>0</td></tr></table>

Note that after constructing the training data as described in Train/Validation/Test Splits., we further filter out samples with a total sequence length (prompt + response) exceeding 4096, matching the fixed context length used in the pretraining stage of LLaDA. As a result, the final dataset may be smaller than the original. 

## Method-Specific Hyperparameters (MDMs Only).

<table><tr><td>Technique</td><td>Setting</td></tr><tr><td>Importance sampling</td><td><eq>\Delta = 0.2</eq></td></tr><tr><td>Stratified sampler</td><td><eq>\# \text{strata} = \lceil \sqrt{\text{batch size}} \rceil = 6</eq></td></tr><tr><td>EMA bins</td><td>10</td></tr><tr><td>EMA learning rate</td><td>0.01</td></tr></table>

## Checkpoint Selection.

<table><tr><td>Criterion</td><td>Setting</td></tr><tr><td>Evaluation strategy</td><td>Early stopping (patience = 3)</td></tr></table>

Random Seeds (MDMs). 

Seeds used 42, 731, 20231 

## How Data Cleaned in HiTab

Raw answers are extracted from \boxed{} delimiters, then cleaned by removing surrounding whitespace, unwrapping \text{...} blocks, converting escaped percent signs \% to %, deleting backslashes and trimming a trailing period 

## Inference Settings for LLaDA.

We set the decoding temperature to 0 for all ARMs and MDMs, ensuring that variance in downstream task performance originates solely from training 

<table><tr><td>Dataset</td><td>Train Data Response Max Length</td><td>ARMs</td><td>MDMs</td></tr><tr><td>GSM8K</td><td>444</td><td>512</td><td>128-128-32</td></tr><tr><td>HiTab</td><td>3571</td><td>3968</td><td>512-256-16</td></tr><tr><td>Open</td><td>4605</td><td>5120</td><td>256-256-256</td></tr></table>

The ARM column indicates the maximum number of new tokens, while the MDM column specifies generation length-steps-block length, as we adopt block diffusion for inference (Arriola et al., 2025). 

For ARMs, we determine the maximum new tokens as: 

max new tokens = ⌈response max length × (1 + s)⌉<sub>64</sub>, 

where s is a safety factor (set to 0.1), and the result is upscaled to the nearest multiple of 64. 

For MDMs, hyperparameters are selected according to dataset type. 

Note. As shown in the table, ARMs require significantly longer generation lengths at inference. This arises from their auto-regressive nature: given a budget of max new tokens, they auto-regressively generate token by token, without explicit awareness of how many tokens are ultimately needed. In contrast, MDMs use substantially fewer tokens-even shorter than the training responses-yet still produce accurate reasoning without truncation. This efficiency arises because MDMs predict all masked tokens in parallel and are constrained by the generation length. For example, with gen length = 512, the model predicts all 512 tokens for the masked input and iteratively applies the remasking–masking scheduler until completion. By design, MDMs avoid unnecessarily verbose outputs, which we argue constitutes another advantage of MDMs over ARMs. 

## Inference Settings for MMaDA.

<table><tr><td>Dataset</td><td>generation length</td><td>steps</td><td>block_length</td><td>temperature</td><td>mask schedule</td></tr><tr><td>text-to-image-2M</td><td>1024</td><td>15</td><td>1024</td><td>0</td><td>cosine</td></tr></table>

## A.7.2 HOW TRAINING VARIANCE IS ESTIMATED

During training, we maintain an online estimator of the per-sample loss variance using importanceweighted statistics. Since losses decrease during training, this metric captures both fluctuations and the overall downward trend, it is an approximate empirical proxy for Eq. 2 

For each batch, the per-token (or per-sample) losses L are detached, flattened, and reweighted by the corresponding importance weights w. We then update three running sums: the weighted mean of losses $( S _ { 1 } ) .$ the weighted mean of squared losses $( S _ { 2 } )$ , and the weighted mean of squared weighted losses $( S _ { 1 2 } )$ . These sufficient statistics allow us to compute unbiased estimates of the second moment $m _ { 2 } .$ , the squared mean $\mu ^ { 2 }$ , and finally the unbiased variance estimator 

$$
\widehat {\mathrm{Var}} [ L ] = m _ {2} - \mu^ {2},\tag{23}
$$

where $\begin{array} { r } { m _ { 2 } = \frac { S _ { 2 } } { n } } \end{array}$ and $\begin{array} { r } { \mu ^ { 2 } = \frac { S _ { 1 } ^ { 2 } - S _ { 1 2 } } { n ( n - 1 ) } } \end{array}$ . Here, n denotes the total number of accumulated samples across updates. If importance sampling on t is enabled, the weights w are drawn from the fitted distribution $p ( t )$ to ensure unbiased estimation; otherwise, unit weights are used. 

At the end of training, the accumulated statistics are queried once to produce the final unbiased variance estimate. These values are then substituted into Eq. 23 to obtain the final estimate of training variance. 

Intuitively, the use of an unbiased estimator is necessary because the naive sample variance can be downward biased when computed incrementally with importance weights. By correcting for this bias, our estimate better reflects the true variance of the training process, independent of batch size or logging frequency. 

## A.8 COMPLETE EXPERIMENTAL RESULTS


A.8.1 PER-RUN EXPERIMENTAL RESULTS ON LLADA


<table><tr><td rowspan="2">Dataset / Method</td><td colspan="3">Seed 42</td><td colspan="3">Seed 731</td><td colspan="3">Seed 20231</td></tr><tr><td>Perf</td><td>Time (h)</td><td>Var</td><td>Perf</td><td>Time (h)</td><td>Var</td><td>Perf</td><td>Time (h)</td><td>Var</td></tr><tr><td colspan="10">OpenScience</td></tr><tr><td>Baseline</td><td>43.57</td><td>11.29</td><td>2.1404</td><td>46.50</td><td>11.32</td><td>2.0614</td><td>46.50</td><td>11.40</td><td>2.0743</td></tr><tr><td>P-POTS+Mirror</td><td>50.80</td><td>14.70</td><td>1.5550</td><td>53.80</td><td>14.07</td><td>1.5672</td><td>53.00</td><td>14.00</td><td>1.5776</td></tr><tr><td>P-POTS</td><td>45.34</td><td>13.63</td><td>1.6319</td><td>47.60</td><td>12.98</td><td>1.6417</td><td>47.47</td><td>12.88</td><td>1.6622</td></tr><tr><td>Mirror</td><td>45.63</td><td>12.46</td><td>2.0800</td><td>47.25</td><td>12.47</td><td>2.0392</td><td>46.25</td><td>12.47</td><td>2.0205</td></tr><tr><td colspan="10">GSM8K</td></tr><tr><td>Baseline</td><td>50.64</td><td>1.55</td><td>0.2526</td><td>53.22</td><td>1.62</td><td>0.2524</td><td>53.68</td><td>1.61</td><td>0.2673</td></tr><tr><td>P-POTS+Mirror</td><td>62.02</td><td>3.64</td><td>0.1906</td><td>58.61</td><td>3.73</td><td>0.1949</td><td>60.96</td><td>3.88</td><td>0.1922</td></tr><tr><td>P-POTS</td><td>59.59</td><td>1.67</td><td>0.1726</td><td>59.36</td><td>1.77</td><td>0.1715</td><td>56.79</td><td>1.67</td><td>0.1722</td></tr><tr><td>Mirror</td><td>56.56</td><td>3.69</td><td>0.3167</td><td>51.40</td><td>3.37</td><td>0.3042</td><td>53.15</td><td>3.38</td><td>0.3162</td></tr><tr><td colspan="10">HiTab</td></tr><tr><td>Baseline</td><td>52.96</td><td>8.22</td><td>1.0063</td><td>60.43</td><td>8.22</td><td>1.0116</td><td>62.64</td><td>8.26</td><td>1.0173</td></tr><tr><td>P-POTS+Mirror</td><td>66.02</td><td>17.80</td><td>0.9903</td><td>66.67</td><td>16.78</td><td>0.9892</td><td>68.62</td><td>16.83</td><td>0.9847</td></tr><tr><td>P-POTS</td><td>60.56</td><td>10.19</td><td>0.7498</td><td>64.74</td><td>10.66</td><td>0.7536</td><td>58.80</td><td>10.99</td><td>0.7503</td></tr><tr><td>Mirror</td><td>64.72</td><td>16.08</td><td>1.3052</td><td>65.04</td><td>16.27</td><td>1.3282</td><td>63.68</td><td>16.29</td><td>1.3480</td></tr></table>


Table 4: Complete results –performance, training time (in hours), and variance – across datasets (OpenScience, GSM8K, HiTab). 


Compared to only reporting mean and standard deviations across runs, this complete results table provides a more direct view of training instability: under standard training, the final performance can vary substantially, $e . g .$ , from 52.96% to 62.64% on HiTab. In contrast, our method achieves more consistent outcomes across runs, demonstrating greater reliability. 

Table 4 and 5 show that methods with lower reported training variance are generally associated with better downstream performance. Interestingly, MIRROR sometimes reports higher variance than the baseline, yet achieves more stable results in evaluation. Since our variance measure serves only as a proxy that captures both the downward trend and short-term fluctuations of training losses, we conjecture that the stronger overall loss reduction under MIRROR may interact with this proxy in a way that inflates the reported variance, even while true training stability improves. 


Table 5: Average training variance with training hours across datasets (All experiments were trained on 2 H100 GPUs; details on training variance approximation provided in A.7.2)


<table><tr><td></td><td>OpenScience</td><td>GSM8K</td><td>HiTab</td></tr><tr><td>Baseline</td><td>2.0920 (11.33)</td><td>0.7723 (1.59)</td><td>3.0352 (8.23)</td></tr><tr><td>P-POTS+MIRROR</td><td>1.5666 (14.26)</td><td>0.5777 (3.75)</td><td>2.9642 (17.13)</td></tr><tr><td>P-POTS</td><td>1.6453 (13.17)</td><td>0.5163 (1.71)</td><td>2.2537 (10.61)</td></tr><tr><td>MIRROR</td><td>2.0466 (12.47)</td><td>0.9371 (3.48)</td><td>3.9814 (16.21)</td></tr><tr><td>ISAD</td><td>2.1480 (11.48)</td><td>0.2525 (1.61)</td><td>1.0016 (10.51)</td></tr><tr><td>SyRM</td><td>-</td><td>-</td><td>0.9233 (8.23)</td></tr><tr><td>StraTS</td><td>2.0851 (11.46)</td><td>0.2568 (1.59)</td><td>1.0083 (10.19)</td></tr><tr><td>EMA</td><td>2.1450 (11.46)</td><td>0.2535 (1.62)</td><td>1.0018 (8.40)</td></tr></table>


A.8.2 PER-RUN EXPERIMENTAL RESULTS ON MMADA


<table><tr><td rowspan="2">Dataset / Method</td><td colspan="3">Seed 42</td><td colspan="3">Seed 731</td><td colspan="3">Seed 20231</td></tr><tr><td>Perf</td><td>Time (h)</td><td>Var</td><td>Perf</td><td>Time (h)</td><td>Var</td><td>Perf</td><td>Time (h)</td><td>Var</td></tr><tr><td>Baseline</td><td>28.61</td><td>3.21</td><td>1.4149</td><td>34.28</td><td>3.09</td><td>1.4245</td><td>31.19</td><td>3.31</td><td>1.3848</td></tr><tr><td>MIRROR</td><td>34.88</td><td>6.88</td><td>1.0308</td><td>33.77</td><td>6.13</td><td>1.0808</td><td>33.57</td><td>6.74</td><td>1.0458</td></tr><tr><td colspan="10">text-to-image-2M EPR</td></tr><tr><td>P-POTS</td><td>31.51</td><td>3.28</td><td>2.6203</td><td>30.02</td><td>3.33</td><td>2.6150</td><td>33.41</td><td>3.37</td><td>2.6236</td></tr><tr><td>P-POTS+MIRROR</td><td>35.27</td><td>6.97</td><td>2.4044</td><td>34.10</td><td>6.79</td><td>2.3504</td><td>34.39</td><td>6.87</td><td>2.3370</td></tr><tr><td colspan="10">text-to-image-2M polynomial (degree 7)</td></tr><tr><td>P-POTS</td><td>30.00</td><td>3.28</td><td>2.8395</td><td>33.35</td><td>3.27</td><td>2.8398</td><td>28.66</td><td>3.56</td><td>2.6187</td></tr><tr><td>P-POTS+MIRROR</td><td>34.51</td><td>5.96</td><td>2.3538</td><td>33.78</td><td>6.43</td><td>2.7319</td><td>34.78</td><td>6.71</td><td>2.3199</td></tr></table>


Table 6: Performance (Perf), training time (in hours), and variance (Var) on text-to-image-2M-EPR and textto-image-2M-poly7 across methods and random seeds (42, 731, 20231). 


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/f0eb91ff-cd64-48a5-ae07-430a6a4f0bcb/4485e9ec4da65f0fac20bb0fa452a995637742af8b2f8f84cb083d696bcca523.jpg)



(a)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/f0eb91ff-cd64-48a5-ae07-430a6a4f0bcb/c3e46f0922778236de5ca6a863f03074576107df45e15fbf259091f6c350b11f.jpg)



(b)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/f0eb91ff-cd64-48a5-ae07-430a6a4f0bcb/ebc6d586fe789ab0fc238e39e54df6f953f06021609cc84059d31e87e20bcf50.jpg)



(c)



Figure 6: Shape of $p ( t )$ under different situations: (a) shape of empirical and fitted $p ( t )$ by EPR before training; (b) by polynomial of degree 7 before training; (c) fitted by EPR after training for one epoch using polynomial (degree 7). These functions are fitted by minimizing the KL divergence, so only their shapes are matched; they are later renormalized in training to yield valid PDFs.


As shown in Figure 6, even when the model is trained with the S-shaped distribution, after one epoch it quickly reverts back to the U-shape. This suggests that the EPR form is a stable and persistent structure across datasets. 

## A.9 EXTENSION TO REINFORCEMENT LEARNING

Applying standard RL algorithms such as PPO and GRPO to MDMs is challenging, because it requires computing log-likelihoods over transitions between consecutive states. In ARMs, states are token sequences. Since tokens are generated one by one, by the chain rule $\begin{array} { r } { P ( x _ { 1 : n } ) = \prod _ { i = 1 } ^ { n } P ( x _ { i } \mid } \end{array}$ $x _ { < i } )$ , the transition probability for each token $x _ { i }$ is naturally $p _ { \theta } ( x _ { i } \mid x _ { < i } )$ . In contrast, MDMs generate tokens in parallel, which induces complex marginalization between intermediate states $x _ { t - 1 }$ and $x _ { t }$ . Training stability is further challenged by token-level rewards derived from highvariance sequence returns. 

Algorithm 1 UniGRPO Policy Gradient Optimization
Require: Reference model $\pi_{\text{ref}}$ , prompt distribution $\mathcal{D}$ , number of completions per prompt $G$ , number of inner updates $\mu$ , diffusion steps $T$ 1: Initialize policy $\pi_{\theta} \leftarrow \pi_{\text{ref}}$ 2: while not converged do
3: $\pi_{\text{old}} \leftarrow \pi_{\theta}$ 4: Sample a prompt $q \sim \mathcal{D}$ 5: Sample $G$ completions $o_i \sim \pi_{\text{old}}(\cdot | q)$ for $i \in [G]$ 6: For each $o_i$ , compute reward $r_i$ and advantage $A_i^{(k)}(\pi_{\text{old}})$ 7: Sample a starting timestep $t_0 \sim \mathcal{U}(0, T-1)$ 8: Generate $\mu-1$ uniformly spaced timesteps $t_1, \ldots, t_{\mu-1}$ from $[t_0, T]$ 9: for gradient update iterations $n=1$ to $\mu$ do
10: if $n=1$ then
11: Sample a starting mask ratio $r_1 \sim \mathcal{U}(0, 1)$ and compute initial timestep $t_1 = \lfloor r_1 T \rfloor$ 12: else
13: Compute $t_n = \left\lfloor \frac{n-1}{\mu-1}(T - t_1) + t_1 \right\rfloor$ 14: end if
15: Construct input ( $q$ , masked $o_i$ ) using timestep $t_n$ (with $q$ always unmasked)
16: For $\pi_{\theta}, \pi_{\text{old}}, \pi_{\text{ref}}$ , estimate log-probabilities of masked tokens in $o_i$ at $t_n$ 17: Compute UniGRPO objective 24 and update $\pi_{\theta}$ via gradient descent
18: end for
19: end while
20: return $\pi_{\theta}$ 

Our methods can be naturally extended to these problems. As an example, consider UniGRPO (Yang et al., 2025) (Algorithm 1), whose training objective is defined as 

$\mathcal{J}_{\mathrm{UniGRPO}}(\theta) = \mathbb{E}_{(q,a)\sim \mathcal{D},\{o_i\}_{i = 1}^G\sim \pi_{\theta_{\mathrm{old}}}(\cdot |q),\{p_i\in [0,1]\}_{i = 1}^G}\left[\frac{1}{G}\sum_{i = 1}^{G}\frac{1}{|o_i|}\sum_{t = 1}^{|o_i|}\Big(\min \big(r_{i,t}^{\prime}(\theta)\hat{A}_{i,t},\operatorname {clip}(r_{i,t}^{\prime}(\theta),1 - \varepsilon ,1 + \varepsilon)\hat{A}_{i,t}\big) - \beta D_{\mathrm{KL}}\big(\pi_\theta^\prime \parallel \pi_\mathrm{ref}^\prime \big)\Big)\right].$ 

(24) 

We can incorporate our core methods through the following modifications: 

• P-POTS: Replace lines 7–8 with sampling µ timesteps from $p ^ { * } ( t )$ , sorted as $t _ { 0 } , \dots , t _ { \mu - 1 }$ 

• MIRROR: At line 15, construct complementary masks $o _ { i } ^ { ( 1 ) } , o _ { i } ^ { ( 2 ) }$ for each $t _ { n }$ , and average their log-probabilities before updating π<sub>θ</sub>. 

At present, this integration is a conjecture; confirming its effectiveness requires empirical validation, which we leave as future work. 

## A.10 FUTURE WORK

While our current results highlight both the importance of variance reduction and the effectiveness of our proposed methods, several open directions remain. First, we aim to deepen our understanding of the p-drift problem and to validate the effectiveness of periodically re-estimating or gradually down-weighting p(t), as suggested in §3.1.1, which so far remains conjectural. Second, since reinforcement learning on MDMs face the difficulty of likelihood evaluation, it is worthwhile to explore whether our methods can also reduce estimation variance in this setting (see A.9). 

## A.11 CASE STUDY

## A.11.1 CASE STUDY: IMAGE GENERATION

Below are images generated by MMaDA-8B-MixCoT trained with P-POTS+MIRROR (left) and standard method (right). 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/f0eb91ff-cd64-48a5-ae07-430a6a4f0bcb/ac9dbafa37469c25f12e2bbb08e7355cf199045516fa9d1879318631ccf4b2c8.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/f0eb91ff-cd64-48a5-ae07-430a6a4f0bcb/38716010c7f309ae5ea1a1bd4acc91632439e09c93cce8f64dbebb20353091d8.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/f0eb91ff-cd64-48a5-ae07-430a6a4f0bcb/633771bebcd09e16ad6c3e04d5c4c7d356ac798e963f6986e816d74d81ecf8ee.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/f0eb91ff-cd64-48a5-ae07-430a6a4f0bcb/8e2368f0c64134f2d96cd2a54ecda30c3ec494573bd9ae4112d6ddf79b01fce9.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/f0eb91ff-cd64-48a5-ae07-430a6a4f0bcb/504fb620df4b7818c5169213c5f553946e8f7999b8d7dc51beb72370eab91343.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/f0eb91ff-cd64-48a5-ae07-430a6a4f0bcb/73b12a877462e0cc3565b43b353bd692e6ac68117b316690b6a468a1521958ca.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/f0eb91ff-cd64-48a5-ae07-430a6a4f0bcb/74b6f9e034f0b7066ce8f939f3097eaf24b9a64dda378e680fb593472cd7dd8e.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/f0eb91ff-cd64-48a5-ae07-430a6a4f0bcb/0109ec5f2a13aca9e48806e8025b2bc033ce31ce2369cb8fe082989555ee1029.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/f0eb91ff-cd64-48a5-ae07-430a6a4f0bcb/8b1892ed267a4d64037e07cba98778c378f74cefee3c36ea256431b333051951.jpg)



Dhettas bo1p


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/f0eb91ff-cd64-48a5-ae07-430a6a4f0bcb/28dbe7957057489a2b6fd5bf85c0745a14fcf8e10b503f8606957667c070d673.jpg)


ocu 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/f0eb91ff-cd64-48a5-ae07-430a6a4f0bcb/b4ac2da36d45ed0d0d25ec9529fa0fee6a38890e63e6d839627396d4e0858c0b.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/f0eb91ff-cd64-48a5-ae07-430a6a4f0bcb/5296e6a9ca84f112709cdf58c718a6e149f6c8eb847ed3cefe76936dbe120595.jpg)
