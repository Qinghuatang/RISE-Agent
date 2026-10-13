# BAPO: STABILIZING OFF-POLICY REINFORCEMENT LEARNING FOR LLMS VIA BALANCED POLICY OPTIMIZATION WITH ADAPTIVE CLIPPING

Zhiheng Xi<sup>1∗†</sup>, Xin Guo<sup>1∗</sup>, Yang Nan<sup>1</sup>, Enyu Zhou<sup>1</sup>, Junrui Shen<sup>1</sup>, Wenxiang Chen<sup>1</sup>, Jiaqi Liu<sup>1</sup>, Jixuan Huang<sup>1</sup>, Zhihao Zhang<sup>1</sup>, Honglin Guo<sup>1</sup>, Xun Deng<sup>2</sup>, Zhikai Lei<sup>2</sup>, Miao Zheng<sup>2</sup>, Guoteng Wang<sup>2</sup>, Shuo Zhang<sup>2</sup>, Peng Sun<sup>2</sup>, Rui Zheng<sup>2</sup>, Hang Yan<sup>2</sup>, Tao Gui<sup>1,3†</sup>, Qi Zhang<sup>1†</sup>, Xuanjing Huang<sup>1</sup> <sup>1</sup>Fudan University <sup>2</sup>Shanghai Qiji Zhifeng Co., Ltd. <sup>3</sup>Shanghai Innovation Institute zhxi22@m.fudan.edu.cn, {tgui,qz}@fudan.edu.cn 

## ABSTRACT

Reinforcement learning (RL) has recently become the core paradigm for aligning and strengthening large language models (LLMs). Yet, applying RL in off-policy settings—where stale data from past policies are used for training—improves sample efficiency, but remains challenging: policy entropy declines sharply, optimization often becomes unstable and may even collapse. Through theoretical and empirical analysis, we identify two key insights: (i) an imbalance in optimization, where negative-advantage samples dominate the policy gradient, suppressing useful behaviors and risking gradient explosions; and (ii) the derived Entropy-Clip Rule, which reveals that the fixed clipping mechanism in PPO-like objectives systematically blocks entropy-increasing updates, thereby driving the policy toward over-exploitation at the expense of exploration. Building on these insights, we propose BAlanced Policy Optimization with Adaptive Clipping (BAPO), a simple yet effective method that dynamically adjusts clipping bounds to adaptively re-balance positive and negative contributions, preserve entropy, and stabilize RL optimization. Across diverse off-policy scenarios—including sample replay and partial rollout—BAPO achieves fast, stable, and data-efficient training. On AIME 2024 and AIME 2025 benchmarks, our 7B BAPO model surpasses open-source counterparts such as SkyWork-OR1-7B, while our 32B BAPO model not only achieves state-of-the-art results among models of the same scale but also outperforms leading proprietary systems like o3-mini and Gemini-2.5-Flash-Thinking. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/b0c03305-03db-4992-947b-779a4fb581f3/3654c02a1481043ca5689dbd29012f71d1e3993fe3e7311b1482205e4ebcf3a7.jpg)



Figure 1: Performance of BAlanced Policy Optimization with Adaptive Clipping (BAPO).


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/b0c03305-03db-4992-947b-779a4fb581f3/bf86596764a5787d2c1fbd500d7cee4366cb1c12befb27a8944734616748a3a7.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/b0c03305-03db-4992-947b-779a4fb581f3/658fc11106e32dd4687947cd6d4f3f7069fc0c3ab34839f25f18902bb1b35e66.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/b0c03305-03db-4992-947b-779a4fb581f3/d5fcdc774fbcc5dc25d9ce8684ba021d7b99ab64e000fea9115f9e79c3c422c0.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/b0c03305-03db-4992-947b-779a4fb581f3/f26af5e93e38928dc4b2f051e6eb875720762f10accfbcad12f057d7f9699471.jpg)



Figure 2: Preliminary results with different data staleness. As the staleness increases, the model suffers from unstable optimization, decreasing entropy, and even a sudden collapse in training.


## 1 INTRODUCTION

Reinforcement learning (RL) has become a pivotal paradigm for optimizing large language models (LLMs) (Zhang et al., 2025), delivering significant improvements in complex tasks such as reasoning (Jaech et al., 2024; Guo et al., 2025a), coding (Anthropic, 2025), and agentic decision-making (Bai et al., 2025; Xi et al., 2025). Among RL methods, off-policy RL—where the rollout policy (behavior policy) differs from the training policy (target policy)—emerges as particularly promising (Roux et al., 2025b; Arnal et al., 2025). It offers high sample efficiency and tolerance to data staleness, making it well-suited for extremely long-horizon and challenging scenarios, while also aligning more naturally with features in modern AI infrastructures such as partial rollout (Team et al., 2025; Fu et al., 2025). 

However, applying off-policy RL to LLMs introduces substantial challenges (Yu et al., 2025; Arnal et al., 2025). As shown in Figure 2, increasing data staleness leads to unstable optimization, exploding gradient and even collapse. Meanwhile, policy entropy declines sharply, reflecting reduced exploratory capacity and a bias toward over-exploitation. By contrast, on-policy training—where rollout and target policies coincide—remains stable across metrics, consistent with prior studies (Tang et al., 2024; Roux et al., 2025b; Arnal et al., 2025). 

To understand the instability of off-policy training, we conduct a comprehensive theoretical and empirical analysis to reveal two key insights. We first demonstrate an imbalance in optimization: policy updates are often dominated by negative-advantage samples, producing excessive penalization signals that suppress even neutral or correct actions and may cause gradient explosions (Gulc¸ehre¨ et al., 2023). We then derive and empirically validate the Entropy-Clip Rule in the widely-used PPO (Schulman et al., 2017) and GRPO (Shao et al., 2024), showing that the clipping mechanism in PPO-like objectives blocks many low-probability positive tokens while over-penalizing low-probability negatives. This systematically excludes entropy-increasing updates, sharpens the output distribution, and drives policies toward over-exploitation at the cost of exploration. 

Based on these insights, we propose BAlanced Policy Optimization with Adaptive Clipping (BAPO), a new method for stable and effective off-policy RL. BAPO dynamically adjusts the clipping bounds to re-balance positive and negative contributions for each update step, incorporate lowprobability positives while filtering excessive negatives, and preserve policy entropy—achieving a better balance between exploration and exploitation. An overview of our approach is illustrated on the right side of Figure 3. 

Experiments across diverse off-policy scenarios—including sample replay, partial rollout, and varying degrees of staleness—on base models such as DeepSeek-R1-Distill-Qwen-7B (Guo et al., 2025a) and OctoThinker-Llama3.2-3B-Long-Zero (Wang et al., 2025b) show that BAPO consistently yields significant improvements. Our 7B model achieves scores of 70.8 on AIME24 and 62.5 on AIME25, surpassing open-source counterparts such as SkyWork-OR1-7B (He et al., 2025). Moreover, our 32B model reaches 87.1 on AIME24 and 80.0 on AIME25, outperforming both comparably scaled open-source models like Qwen3-32B (Yang et al., 2025a) and leading proprietary systems including o3-mini-medium (OpenAI, 2025) and Gemini-2.5-Flash-Thinking (Comanici et al., 2025). 

Our contributions are summarized as follows: 

• We identify and analyze two key insights behind instability in off-policy RL for LLMs: the imbalanced optimization and the Entropy-Clip Rule. (§3) 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/b0c03305-03db-4992-947b-779a4fb581f3/ec26a6904add15ca105f03af2ace962f43aa37ee0a997a1a7b8bef42fa84367c.jpg)



Figure 3: An illustration of our proposed BAPO. (Left) Baseline methods like GRPO use symmetric fixed clipping bounds, reinforcing high-probability positive tokens while penalizing excessive low-probability negatives, leading to sharp distributions and entropy collapse. (Right) BAPO dynamically adjusts the clipping bounds $c _ { \mathrm { l o w } }$ and $c _ { \mathrm { h i g h } }$ based on the loss contributions from positive tokens. It excludes overly negative tokens $\bigstar$ to maintain a smoother distribution and incorporates previously clipped positive tokens $\triangle$ to preserve entropy balance.


• We propose BAPO, a new RL algorithm that dynamically adjusts clipping bounds to balance positive and negative signals, preserving entropy for exploration, and stabilizing training. (§4) 

• We validate BAPO across multiple backbones, model scales, and off-policy settings, showing that it achieves stable optimization and competitive results with proprietary systems.<sup>1</sup> (§5) 

## 2 PRELIMINARIES

Policy gradient. In the field of LLM RL (Trung et al., 2024; Jaech et al., 2024; Xi et al., 2024b), policy gradient-based (PG) algorithms (Williams, 1992) are widely used. Specifically, given an input prompt x, an LLM $\pi _ { \theta }$ sequentially generates a $T \cdot$ -token response $\pmb { y } = ( y _ { 1 } , . . . , y _ { T } )$ 

$$
\pi_ {\theta} (\pmb {y} | \pmb {x}) = \prod_ {t = 1} ^ {T} \pi_ {\theta} (y _ {t} | \pmb {x}, \pmb {y} _ {<   t}).\tag{1}
$$

Given a training dataset $\mathcal { D } = \{ \pmb { x } _ { 1 } , . . . , \pmb { x } _ { N } \}$ and reward function R, the RL objective is to maximize the expected reward: 

$$
J (\theta) = \mathbb {E} _ {\boldsymbol {x} \sim \mathcal {D}, \boldsymbol {y} \sim \pi_ {\theta} (\cdot | \boldsymbol {x})} [ R (\boldsymbol {x}, \boldsymbol {y}) ].\tag{2}
$$

PG algorithms then leverage gradient ascent to optimize the policy with the following gradient: 

$$
\nabla_ {\theta} J (\theta) = \mathbb {E} _ {\pmb {x} \sim \mathcal {D}, \pmb {y} \sim \pi_ {\theta} (\cdot | \pmb {x})} \left[ \sum_ {t = 1} ^ {T} \nabla_ {\theta} \log \pi_ {\theta} (y _ {t} | \pmb {x}, \pmb {y} _ {<   t}) \cdot A _ {t} \right],\tag{3}
$$

where $A _ { t }$ denotes the estimated advantage at time step $t ,$ i.e., how much better action $y _ { t }$ is than the expected action under the current policy. 

Importance sampling and PPO objective. To improve sample efficiency and adapt to modern infrastructure, mainstream RL algorithms for LLMs typically adopt a PPO-like surrogate objective (Schulman et al., 2017): 

$$
J ^ {\mathrm{PPO}} (\theta) = \mathbb {E} _ {\boldsymbol {x} \sim \mathcal {D}, \boldsymbol {y} \sim \pi_ {\theta_ {\mathrm{rollout}}} (\cdot | \boldsymbol {x})} \sum_ {t = 1} ^ {T} \left[ \min (r _ {t} \cdot A _ {t}, \operatorname{clip} (r _ {t}, 1 - \varepsilon , 1 + \varepsilon) \cdot A _ {t}) \right],\tag{4}
$$

where $\begin{array} { r } { r _ { t } = \frac { \pi _ { \boldsymbol { \theta } } \left( y _ { t } | \mathbf { x } , \pmb { y } _ { < t } \right) } { \pi _ { \boldsymbol { \theta } _ { \mathrm { r o l l o u t } } } \left( y _ { t } | \mathbf { x } , \pmb { y } _ { < t } \right) } } \end{array}$ is the importance weight that corrects for the distribution mismatch, estimating the expected advantage of tokens generated by the behavior policy $\pi _ { \theta _ { \mathrm { r o l l o u t } } }$ under the target policy $\pi _ { \theta } .$ . The clipping mechanism in PPO serves to implicitly enforce a trust region between the behavior and target policies, preventing overly large policy updates that could destabilize training. The hyperparameter $\varepsilon \in ( 0 , 1 )$ determines the width of this clipping interval. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/b0c03305-03db-4992-947b-779a4fb581f3/9dc2db9439fb7475dca55a74307e7e98ead2c512c3b966f03dfeadebcbbcbe6a.jpg)



Figure 4: Contribution of positive and negative tokens to the policy-gradient loss and their proportion of tokens during training.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/b0c03305-03db-4992-947b-779a4fb581f3/846388004f61fdc878c8d7681cbfcf2fe7fb3549edf53fd59811eb4fdc4c7d3d.jpg)



Figure 5: Relationship between token probability and importance sampling weight.


We then analyze data with positive and negative advantages respectively. The policy gradient can then be expressed as: 

$$
\begin{array}{c} \nabla J ^ {\text {PPO}} = \sum_ {\boldsymbol {x}, y _ {t}} [ \underbrace {\mathbb {I} (A _ {t} > 0) \cdot \mathbb {I} \{r _ {t} <   1 + \varepsilon \} \cdot A _ {t} \cdot \pi_ {\theta} (y _ {t}) \cdot \nabla \log \pi_ {\theta} (y _ {t})} _ {\text {positive tokens}} \\ + \underbrace {\mathbb {I} (A _ {t} <   0) \cdot \mathbb {I} \{r _ {t} > 1 - \varepsilon \} \cdot A _ {t} \cdot \pi_ {\theta} (y _ {t}) \cdot \nabla \log \pi_ {\theta} (y _ {t}) ]} _ {\text {negative tokens}}, \end{array}\tag{5}
$$

where I represents the indicator function. 

## 3 MOTIVATION: IMBALANCED OPTIMIZATION AND ENTROPY-CLIP RULE

In this section, we first conduct preliminary experiments to show the influence of data staleness on the RL optimization process. Next, we perform in-depth empirical and theoretical analysis to reveal the underlying mechanisms and provide new insights. 

Training instability with data staleness. We perform experiments under different levels of data staleness using the popular GRPO algorithm. Results in Figure 2 show that, compared to on-policy training, off-policy RL typically suffers from instability, and entropy decreases rapidly, reflecting reduced exploratory capacity (He et al., 2025). As staleness increases, the entropy decline becomes more severe and a larger number of tokens are clipped; meanwhile, training becomes more unstable. In the following paragraphs, we attempt to explain this phenomenon from different perspectives and summarize the motivation behind our method. 

Excessive negative samples lead to imbalanced optimization. Within the PPO-like objective for policy updates, we analyze tokens with positive and negative advantages separately, as shown in Equation 5. Empirical results in Figure 4 reveal a pronounced imbalance: positive samples constitute a minority both in number and in their contribution to the policy-gradient loss. We attribute this skew to two main factors: (i) the model tends to generate longer trajectories on difficult queries, thereby producing more tokens in negative samples (Figure 6); and (ii) in early stages of training, the model has not yet acquired sufficient capability, resulting in a higher proportion of negative samples. This observation may help explain the effectiveness of certain curriculum-based approaches (Xi et al., 2024a; Yuan et al., 2025). 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/b0c03305-03db-4992-947b-779a4fb581f3/ce6e5dac16ca17e1cd16cad837e6e71c94b78ee2aa9d22ca5676b048099e1bf9.jpg)



Figure 6: Average model response length during training.


In the RL training of LLMs, reinforcing positive samples is often more efficient for driving performance gains than attempting to “suppress” the vast number of negative samples (Gulc¸ehre et al.,¨ 2023; Zhu et al., 2025). To this end, prior work has proposed amplifying positive signals through the clip-higher technique (Yu et al., 2025). However, merely enlarging the clipping upper bound does not mitigate the influence of negative data, thus failing to prevent them from dominating the optimization process. Moreover, as shown in Equation 5, the accumulation of low-probability negative tokens $( \mathrm { i . e . , } \pi _ { \theta } ( y _ { t } )  0$ , driving the log term toward −∞) may trigger gradient explosion, further destabilizing training (Yang et al., 2025c). 

The Entropy-Clip Rule exposes insufficient entropy promotion in optimization, leading to entropy collapse. Theoretically, we derives Equation 6 (see Appendix G for detailed derivations) for PPO surrogate objective to reveal the factors that influence the policy entropy (Roux et al., 2025b): 

$$
\Delta \mathcal {H} (\pi_ {\theta}) \approx - \eta \cdot \mathrm{Cov} _ {\pmb {y} \sim \pi_ {\theta} (\cdot | \pmb {x})} \left[ \log \pi_ {\theta} (y _ {t} | \pmb {x}, \pmb {y} _ {<   t}), A _ {t} \cdot \mathcal {X} (y _ {t}) + C \right],\tag{6}
$$

where C is a constant, and 

$$
\mathcal {X} (y _ {t}) = \left\{ \begin{array}{l l} 1, & \text { if } A _ {t} > 0 \& r _ {t} <   1 + \epsilon \\ & \text { or } A _ {t} <   0 \& r _ {t} > 1 - \epsilon \\ 0, & \text { otherwise. } \end{array} \right.\tag{7}
$$

We observe that changes in policy entropy are driven by the influence of unclipped tokens, which is determined by the covariance between their log probabilities and advantages. We term this as the Entropy-Clip Rule. The left side of Figure 3 illustrates how the optimization of different tokens influences the probability distribution, thereby affecting entropy. The Entropy-Clip Rule theoretically explains the following statement: Specifically, updating the policy with positive high-probability tokens (high advantage, high probability) and negative low-probability tokens (low advantage, low probability) sharpens the distribution and consequently reduces entropy. Conversely, updating the policy with negative high-probability tokens and positive low-probability tokens smooths the distribution, resulting in an increase in entropy (detailed proofs are available in Appendix G.4.2). 

Empirically, our statistical analysis on token probabilities and their importance sampling (IS) weights further clarifies this dynamic. As shown in Figure 5, we find that tokens with either very high or very low IS weights tend to have low probabilities. However, in standard algorithms with symmetric clipping bounds (e.g., [0.8,1.2]), a majority of positive, low-probability tokens are prevented from contributing to the optimization. This systematic exclusion of entropy-increasing updates causes a continuous decline in entropy, ultimately crippling the model’s exploratory capacity and resulting in a performance bottleneck. 

Summary of motivation. Based on the above analysis, we can summarize two main motivations: (1) to balance the contributions of positive and negative tokens while preventing gradient explosion, and (2) to preserve policy entropy for sustaining exploration and preventing collapse. 

## 4 METHODOLOGY

## 4.1 VALIDATION EXPERIMENT: ASYMMETRIC CLIPPING

The main idea of our method is to stabilize the training and maintain exploration ability of the policy by asymmetrically adjusting the trust region for positive and negative tokens, i.e., adjusting $c _ { \mathrm { l o w } }$ and $c _ { \mathrm { h i g h } }$ 

We then conduct preliminary validation experiments to examine whether asymmetrically adjusting the clipping range could influence the training dynamics. The results, shown in Figure 7, together with Figure 5, reveal that increasing the upper bound $c _ { \mathrm { h i g h } }$ (which introduces more low-probability positive tokens to policy updates) improves performance while counteract-

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/b0c03305-03db-4992-947b-779a4fb581f3/e316318216fdfffc88fdc1bf2d6dc9f00b67001172173babc55f6beb6d4bf8ab.jpg)



Figure 7: Training dynamics of asymmetric clipping experiments.


ing the downward trend of entropy, albeit at a rapid pace. In contrast, relaxing the lower bound $c _ { \mathrm { l o w } }$ (which introduces more low-probability negative tokens to policy updates) not only degrades performance but also accelerates entropy collapse. These findings confirm the effectiveness of entropy control through asymmetric clipping. Nevertheless, this approach remains relatively rigid and manually specified, providing limited flexibility and adaptation. 

Algorithm 1: BAPO

Input: Initialized LLM policy $\pi_{\theta}$ , training dataset D, reward function R, staleness E, movable range of clipping bounds $[a^{-}, b^{-}]$ and $[a^{+}, b^{+}]$ , step size of upper bound $\delta_{1}$ , step size of lower bound $\delta_{2}$ , positive token contribution threshold $\rho_{0}$ for step s = 1...S do

Procedure Sample and filter out responses

Update the old LLM policy $\pi_{\theta_{rollout}} \leftarrow \pi_{\theta}$ ;

Sample the s-th batch $D_{s}$ from D ;

Sample G responses $\{y_{i}\}_{i=1}^{G} \sim \pi_{\theta_{rollout}}(\cdot | x)$ , where $x \in D_{s}$ ;

Compute reward and advantage for each $y_{i}$ based on reward function R ;

for staleness = 0...E do

Procedure Dynamically adjusting the clipping bounds $c_{high}$ and $c_{low}$ Initialize clipping bounds $c_{low} = a^{-}$ and $c_{high} = a^{+}$ ;

while the positive token contribution $\rho < \rho_{0}$ and $c_{low} + \delta_{2} \leq b^{-}$ do

if $c_{high} + \delta_{1} \leq b^{+}$ then $c_{high} \leftarrow c_{high} + \delta_{1}$ else $c_{low} \leftarrow c_{low} + \delta_{2}$ end

end

Procedure Update the LLM policy $\pi_{\theta}$ Update the LLM policy $\pi_{\theta}$ by maximizing the following objective: $J^{\mathrm{BAPO}}(\theta) = \mathbb{E}_{\boldsymbol{y} \sim \pi_{\theta_{rollout}}(\cdot | x)} \sum_{t=1}^{T} [\min(r_t \cdot A_t), \operatorname{clip}(r_t, c_{\operatorname{low}}, c_{\operatorname{high}}) \cdot A_t)]$ end

end 

## 4.2 BAPO: BALANCED POLICY OPTIMIZATION WITH ADAPTIVE CLIPPING

To this end, we propose BAlanced Policy Optimization with Adaptive Clipping (BAPO), a new method to achieve stable, fast RL optimization for LLMs. The core insight of ${ \mathrm { B A P O } }$ lies in its adaptive clipping mechanism, which dynamically adjusts the clipping bounds $c _ { \mathrm { h i g h } }$ and $c _ { \mathrm { l o w } } .$ , to regulate the positive contribution to the policy loss and maintain a balance in entropy throughout RL training. Formally, for each update with a batch, our goal is to find a pair of $c _ { \mathrm { h i g h } }$ and $c _ { \mathrm { l o w } }$ that satisfy: 

$$
\frac {\sum_ {A _ {t} > 0} \pi_ {\theta_ {\text {rollout}}} (y _ {t}) \cdot | \min (r _ {t} \cdot A _ {t} , \operatorname{clip} (r _ {t} , 0 , c _ {\text {high}}) \cdot A _ {t}) |}{\sum_ {A _ {t}} \pi_ {\theta_ {\text {rollout}}} (y _ {t}) \cdot | \min (r _ {t} \cdot A _ {t} , \operatorname{clip} (r _ {t} , c _ {\text {low}} , c _ {\text {high}}) \cdot A _ {t}) |} \geq \rho_ {0}  ,\tag{8}
$$

where $\rho _ { 0 }$ is the target contribution of positive signals to the policy gradient loss. Specifically, BAPO gradually increases $c _ { \mathrm { h i g h } }$ and $c _ { \mathrm { l o w } }$ with step sizes of $\delta _ { 1 }$ and $\delta _ { 2 } ,$ respectively, until the condition in Equation 8 is met. We present an overview of BAPO in Figure 3 and summarize it in Algorithm 1. 

Overall, BAPO offers several significant benefits. First, by dynamically adjusting $c _ { \mathrm { h i g h } }$ and $\mathcal { C } _ { \mathrm { l o w } }$ for each step, we can increase the contribution of positive tokens to the policy-gradient loss while preventing negative tokens from excessively dominating the optimization objective. Second, based on our earlier analysis of the relationship between IS weights and token probabilities in Figure 5, BAPO incorporates more low-probability positive tokens and filters out more low-probability negative tokens, both of which contribute to maintaining entropy. Third, by setting the target contribution from positive tokens, BAPO prevents uncontrolled entropy growth, avoids situations where positive tokens overwhelm the loss, and mitigates tail degradation—where the model overfits to easy problems but fails to handle more challenging ones (Ding et al., 2025b). 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/b0c03305-03db-4992-947b-779a4fb581f3/38c5c5b3463b8e11081ce0bb25e10f38a1fab040ba8167f5dfc5dddf9de21bda.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/b0c03305-03db-4992-947b-779a4fb581f3/a844b1828aa40d0bd821f2ebadcc3ff292cc3df942e0498e22628deb5954ab98.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/b0c03305-03db-4992-947b-779a4fb581f3/b4e4bb608f7a23c978e000f74978877b21fe3f56757c63387b2c1552a33d12da.jpg)



Figure 8: Training dynamics of BAPO.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/b0c03305-03db-4992-947b-779a4fb581f3/689b01d8e131f54c5edf4e54477deddc9dbd4a809ca6c0ed763f777610d0a355.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/b0c03305-03db-4992-947b-779a4fb581f3/783d494c4ebd97984b26f96bf460e1357998c469cfb808e05cff71910d7e48ff.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/b0c03305-03db-4992-947b-779a4fb581f3/67e175b60b3a28d5592ef0144332a44971bd09df07d3deff6170e485f8b28d95.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/b0c03305-03db-4992-947b-779a4fb581f3/64ac32cc4492e58ec5b9d573f90eba35764022d89c6da917c45081dbab2c766c.jpg)



Figure 9: Relationship among token probabilities, importance sampling weights, and entropy.


## 4.3 ANALYSIS

Stable and fast training of BAPO. As shown in Figure 8, BAPO enables a more stable optimization process, characterized by rapidly increasing training rewards, greater contributions from positive tokens, steady gradient normalization, and stable policy entropy—resulting in an improved balance between exploration and exploitation. 

We further visualize the adjustment process of the clipping bounds in BAPO. As shown in Figure 10, the averaged upper and lower clipping bounds both fluctuate during training, confirming that BAPO dynamically adjusts the clipping for both types of data and adaptively balances their contributions to the loss. In contrast to approaches such as DAPO (Yu et al., 2025) or the asymmetric clipping in Section 4.1, which rely on empirical tuning, BAPO eliminates the need for complex manual hyperparameter tuning, making it simple yet effective. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/b0c03305-03db-4992-947b-779a4fb581f3/956025bfb1ede2e8b494e242cbb93327a9dac728dc774fe5f5aab553b3343f17.jpg)



Figure 10: Clipping bounds.


Effectiveness of BAPO across different staleness. We conduct experiments using the R1- Distill model (Guo et al., 2025a) on the SkyWork-OR1-RL dataset (He et al., 2025), with a maximum sequence length of 32k. The results in Figure 11 show that under different data staleness, our method consistently outperforms both the baseline and the clip-higher approach, demonstrating its superiority. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/b0c03305-03db-4992-947b-779a4fb581f3/b59edde055687ef44c340f3057973dfe91936f5445f58833a7ea93900c3cca7f.jpg)



Figure 11: Results with different data staleness.


The working mechanism of BAPO and its connection to prior work. To better understand the working mechanism of BAPO, we present the relationship among token probabilities, IS weights, and entropy during training in Figure 9. We find that as IS weights deviate further from 1, the corresponding token probabilities decrease, and such low-probability tokens often exhibit higher entropy. Based on this observation, we explain how BAPO relates to prior work. For example, Clip-Higher in Yu et al. (2025) sets the clipping upper bound to 1.28, thereby including more lowprobability positive tokens in training, which stabilizes entropy while balancing the contributions of positive and negative tokens. Similarly, Wang et al. (2025a) retain only the top 20% highest-entropy tokens for training, ensuring stable entropy throughout optimization and preserving the model’s exploratory capability, and the target entropy technique in He et al. (2025) plays a similar role, which aligns with our motivation. 


Table 1: Main evaluation results.


<table><tr><td>Model</td><td>Model Size</td><td>AIME 2024</td><td>AIME 2025</td><td>Average</td></tr><tr><td colspan="5"><eq>\geq 100B</eq> Models and Proprietary Models</td></tr><tr><td>Qwen3-235B-A22B (Yang et al., 2025a)</td><td>235B</td><td>85.7</td><td>81.5</td><td>83.6</td></tr><tr><td>DeepSeek-R1 (Guo et al., 2025a)</td><td>671B</td><td>79.8</td><td>70.0</td><td>74.9</td></tr><tr><td>DeepSeek-R1-0528 (Guo et al., 2025a)</td><td>671B</td><td>91.4</td><td>87.5</td><td>89.5</td></tr><tr><td><eq>o1_{medium}</eq> (Jaech et al., 2024)</td><td>-</td><td>83.3</td><td>79.0</td><td>81.2</td></tr><tr><td><eq>o3-mini_{medium}</eq> (OpenAI, 2025)</td><td>-</td><td>79.6</td><td>76.7</td><td>78.2</td></tr><tr><td><eq>o3-mini_{high}</eq> (OpenAI, 2025)</td><td>-</td><td>87.3</td><td>86.5</td><td>86.9</td></tr><tr><td>Gemini-2.0<eq>_{Flash-Thinking}</eq> (Google, 2024)</td><td>-</td><td>73.3</td><td>53.5</td><td>63.4</td></tr><tr><td>Gemini-2.5<eq>_{Flash-Thinking-0520}</eq> (Comanici et al., 2025)</td><td>-</td><td>82.3</td><td>72.0</td><td>77.2</td></tr><tr><td colspan="5">10B - 100B Scale Models</td></tr><tr><td>Qwen3-30B-A3B (Yang et al., 2025a)</td><td>30B</td><td>-</td><td>61.3</td><td>-</td></tr><tr><td>R1-Distill-Qwen-32B (Guo et al., 2025a)</td><td>32B</td><td>72.6</td><td>54.9</td><td>63.8</td></tr><tr><td>QwQ-32B (Qwen, 2025)</td><td>32B</td><td>79.5</td><td>65.3</td><td>72.4</td></tr><tr><td>Qwen3-32B (Yang et al., 2025a)</td><td>32B</td><td>81.4</td><td>72.9</td><td>77.2</td></tr><tr><td>SkyWork-OR1-32B (He et al., 2025)</td><td>32B</td><td>82.2</td><td>73.3</td><td>77.8</td></tr><tr><td>BP-Math-32BSFT</td><td>32B</td><td>84.4</td><td>78.1</td><td>81.3</td></tr><tr><td>BP-Math-32BGRPO</td><td>32B</td><td>84.6</td><td>78.8</td><td>81.7</td></tr><tr><td>BP-Math-32BBAPO</td><td>32B</td><td>87.1</td><td>80.0</td><td>83.5</td></tr><tr><td colspan="5"><eq>\leq 10B</eq> Models</td></tr><tr><td>R1-Distill-Qwen-7B (Guo et al., 2025a)</td><td>7B</td><td>54.2</td><td>38.4</td><td>46.3</td></tr><tr><td>Light-R1-7B-DS (Wen et al., 2025)</td><td>7B</td><td>59.1</td><td>44.2</td><td>51.7</td></tr><tr><td>AReaL-boba-RL-7B (Fu et al., 2025)</td><td>7B</td><td>61.9</td><td>48.3</td><td>55.1</td></tr><tr><td>AceReason-Nemotron-7B (Chen et al., 2025c)</td><td>7B</td><td>69.0</td><td>53.6</td><td>61.3</td></tr><tr><td>SkyWork-OR1-7B (He et al., 2025)</td><td>7B</td><td>70.2</td><td>54.6</td><td>62.4</td></tr><tr><td>BP-Math-7BSFT</td><td>7B</td><td>66.9</td><td>59.0</td><td>62.9</td></tr><tr><td>BP-Math-7BGRPO</td><td>7B</td><td>69.2</td><td>59.2</td><td>64.2</td></tr><tr><td>BP-Math-7BBAPO</td><td>7B</td><td>70.8</td><td>62.5</td><td>66.7</td></tr></table>

## 5 EXPERIMENTS AND DISCUSSION

## 5.1 EXPERIMENTAL SETUPS

Datasets and Models. We use SkyWork-OR1-RL-Data (He et al., 2025) as our RL dataset, as it is widely adopted and of high quality. For evaluation, we employ both the AIME 2024 and the newly released AIME 2025 (AIME, 2025) benchmarks. Our experiments cover a range of backbone models, including DeepSeek-R1-Distill-Qwen-7B, DeepSeek-R1-Distill-Qwen-32B (Guo et al., 2025a), and OctoThinker-Llama3.2-3B-Long-Zero (Wang et al., 2025b). In addition, we incorporate two our own supervised fine-tuning (SFT) models, BP-Math-7B and BP-Math-32B, which are derived from Qwen2.5-Math (Yang et al., 2024) through fine-tuning. 

Implementation details. We leverage GRPO as the basis for BAPO. Both our preliminary and validation experiments are conducted using DeepSeek-R1-Distill-Qwen-7B, with the maximum response length set to 8k, learning rate to $2 \times 1 0 ^ { - 6 }$ , and temperature to 0.6. For main results on BP-Math models, we set the maximum response length to 64k to align with the SFT setting. To introduce staleness, we adopt multiple strategies, including experience reuse through ppo epoch (Schulman et al., 2017) and the modern partial rollouts (Team et al., 2025; Fu et al., 2025). For BAPO, we set the target contribution $\rho _ { 0 } = 0 . 4$ , the movable range $a ^ { - } = 0 . 6 , b ^ { - } = 0 . 9 , a ^ { + } = 1 . 2 .$ $b ^ { + } = 3 . 0 ,$ and the step size $\delta _ { 1 } = 0 . 0 5 , \delta _ { 2 } = 0 . 0 2$ . These hyperparameters are not finely tuned, as they already demonstrate strong empirical performance. For evaluation, we report results averaged over 16 rollouts. 

Baselines. We include a variety of commercial and open-source models of different scales as baselines, as shown in Table 1, and report their performance as extracted from prior work. In addition, we compare different training approaches, including SFT and GRPO. 

## 5.2 MAIN RESULTS

The main results are shown in Figure 1 and Table 1. 

Significant performance improvements across models of varying sizes. For strong SFT models, GRPO provides only marginal benefits—for instance, it improves performance by just 0.2 and 0.7 points on AIME24 and AIME25 with the BP-Math-32B model. In contrast, BAPO delivers substantial gains across models of different scales. Specifically, with the BP-Math-32B model, BAPO outperforms SFT by 2.7 and 1.9 points on AIME24 and AIME25, respectively; with the BP-Math-7B model, it achieves even larger improvements of 3.9 and 3.5 points. 

SOTA performance over open-source models of comparable sizes and competitive results against proprietary models. Compared to open-source models of similar sizes, our BAPOtrained models achieve state-of-the-art (SOTA) performance. For instance, among 32B models, BP-Math-32B<sub>BAPO</sub> outperforms Qwen3-32B by 5.7 and 7.1 points on AIME24 and AIME25, respectively, and surpasses SkyWork-OR1-32B by 4.9 and 6.7 points. Among 7B models, BP-Math-$7 \mathtt { B } _ { \mathtt { B A P O } }$ also delivers a notable 7.9-point improvement over SkyWork-OR1-7B on AIME25. 

Moreover, BP-Math-32B even outperforms some larger-scale models—for example, it surpasses DeepSeek-R1 by 7.3 and 10.0 points on AIME24 and AIME25, respectively—while achieving performance comparable to o3-mini. Notably, even the smaller $\mathsf { B P - M a t h - 7 B _ { B A P O } }$ yields results on par with Gemini-2.0-Flash-Thinking, underscoring the competitiveness of our approach against commercial models. 

## 5.3 DISCUSSION

Partial rollout. To speed up rollouts in LLM reinforcement learning, modern AI infrastructures have introduced several techniques, with partial rollout being particularly noteworthy (Team et al., 2025; Fu et al., 2025). In this approach, long trajectories are split into segments: when a rollout exceeds a fixed token budget, the unfinished portion is stored in a replay buffer and resumed in later iterations instead of being regenerated from scratch. While this improves training efficiency, it also introduces off-policy 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/b0c03305-03db-4992-947b-779a4fb581f3/bd34ecedaced9af420dda231ec44b57e4b4441a275ac317c26fde23a0460730f.jpg)



Figure 12: Training dynamics with partial rollout.


learning, since different parts of the same trajectory may come from multiple outdated policies. We evaluate BAPO under this setting, as shown in Figure 12. Compared to the baseline GRPO, BAPO exhibits greater robustness to such off-policy infrastructures and achieves more stable optimization. 

Results on OctoThinker-Llama3.2-3B-Long-Zero. In addition to the DeepSeek-R1-Distill-Qwen, we also conducted experiments on Llama-based models (Wang et al., 2025b). As shown in Table 2 and Figure 13 in Appendix B, our method achieves more competitive results and exhibits greater stability in training dynamics. 


Table 2: Performance of Llama-based models.


<table><tr><td>Method</td><td>AIME 2024</td><td>AIME 2025</td><td>MATH</td></tr><tr><td>GRPO</td><td>2.5%</td><td>2.9%</td><td>58.4%</td></tr><tr><td>BAPO</td><td>5.4%</td><td>5.8%</td><td>66.0%</td></tr></table>

Generalization across diverse reasoning domains. To further validate the universality and robustness of BAPO beyond standard mathematical benchmarks, we expanded our evaluation scope to cover a broader spectrum of reasoning tasks. Experiments were conducted using DeepSeek-R1- Distill-Qwen-7B. We configured the maximum sequence length to 16k for the logical reasoning task to accommodate long-context few-shot examples, while maintaining an 8k limit for mathematical and scientific tasks. As shown in Table 3 and Figure 14 in Appendix C, BAPO consistently delivers substantial performance gains across all tested domains. 

Comparison with other algorithms. Beyond the standard GRPO baseline, we extended our evaluation to systematically compare BAPO with advanced variants sharing similar motivations, including TOPR Roux et al. (2025a), DAPO Yu et al. (2025), and DCPO Yang et al. (2025b). All methods were implemented and evaluated using the identical BP-Math-7B backbone and BAPO consistently outperforms these strong baselines as presented in Table 4. This superior performance validates that, among current advanced variants of GRPO, our adaptive clipping strategy offers the most effective solution for stabilizing training and maximizing reasoning capabilities. 


Table 3: Comparison of performance across different domains. Results includes AMC 2023 and OlympiadBench He et al. (2024) for advanced mathematical reasoning, ARC-AGI Chollet (2019) for logical reasoning, and GPQA-Diamond Rein et al. (2024) for scientific question answering.


<table><tr><td>Domain</td><td>Training Set</td><td>Test Set</td><td>Original</td><td>GRPO</td><td>BAPO</td><td>Improvement</td></tr><tr><td rowspan="2">Math</td><td rowspan="2">Skywork Math</td><td>AMC 2023</td><td>73.3</td><td>82.5</td><td>92.5</td><td>+19.2</td></tr><tr><td>OlympiadBench</td><td>33.6</td><td>54.2</td><td>58.4</td><td>+25.4</td></tr><tr><td>Logical</td><td>Enigmata Chen et al. (2025a)</td><td>ARC-AGI</td><td>0</td><td>1.4</td><td>3.2</td><td>+3.2</td></tr><tr><td>General</td><td>General-Reasoner Ma et al. (2025)</td><td>GPQA-Diamond</td><td>42.4</td><td>45.7</td><td>51.3</td><td>+8.9</td></tr></table>

Robustness to hyperparamteters To demonstrate the robustness of BAPO and assess its sensitivity to hyperparameter settings, we conducted comprehensive ablation studies on the BP-Math-7B model. We specifically examined three key components of our algorithm: (1) the positive token contribution threshold $\rho _ { 0 } , ( 2 )$ the movable ranges of clipping bounds $[ a _ { - } , b _ { - } ]$ and $[ \bar { a } _ { + } , b _ { + } ]$ , and (3) the adjustment step sizes $\delta _ { 1 }$ and $\delta _ { 2 }$ . The results are detailed in Appendix E. Overall these experiments demonstrate that BAPO does not require intricate hyperparameter tuning. Its key advantage lies in the ability to automatically adapt clipping values within broad search spaces, achieving superior performance using simple heuristic settings. 

## 6 RELATED WORK

Recent landmark models, like OpenAI o1 (Jaech et al., 2024), DeepSeek-R1 (Guo et al., 2025a), Gemini 2.5 (Comanici et al., 2025), QwQ (Qwen, 2025), have demonstrated that reinforcement learning can effectively enable long chain-of-thought reasoning in LLMs (Shao et al., 2024; Zhang et al., 2025; Cai et al., 2025; Chen et al., 2025b). Mainstream algorithms include PPO (Schulman et al., 2017) and GRPO (Shao et al., 2024): PPO constrains updates via a clipping-based surrogate objective, while GRPO enhances long-horizon reasoning through group-based rewards. 

Despite the remarkable success of RL for LLMs, ensuring stability and efficiency in optimization remains a major challenge (Yu et al., 2025; Cui et al., 2025). Recent studies have sought to better understand the underlying mechanisms of RL and proposed new methods to achieve a balance (Zheng et al., 2025; Wang et al., 2025a; Yang et al., 2025b; 2026; Ding et al., 2025a; Guo et al., 2025b; Xi et al., 2024c). For example, DAPO (Yu et al., 2025) introduces techniques such as Clip-Higher and dynamic sampling to raise the performance ceiling; Wang et al. (2025a) explore optimizing only a small subset of high-entropy tokens for improved efficiency. He et al. (2025), Cui et al. (2025), and other works (Zheng et al., 2025; Cheng et al., 2025; Liu et al., 2025) systematically investigate how to maintain entropy stability during training, thereby preserving the model’s exploration ability. For off-policy RL, Roux et al. (2025b) and Arnal et al. (2025) introduce asymmetric clipping mechanisms. The most similar to our work is DCPO (Yang et al., 2025b), which adjusts token-level clipping based on token prior probabilities. However, our approach takes a holistic optimization perspective: we observe the imbalance in loss contributions and derive the Entropy-Clip Rule for the PPO objective, enabling dynamic control over global clipping bounds. We further validate the effectiveness of our method through larger-scale experiments. 

## 7 CONCLUSION

In this paper, we begin by analyzing the impact of data staleness on model training through both empirical and theoretical studies. We reveal the imbalance between positive and negative samples in RL optimization, and derive as well as empirically validate the Entropy-Clip Rule for PPO-like objectives. Building on these insights, we propose BAPO, which dynamically adjusts the clipping bounds to balance positive and negative samples while preserving the model’s exploratory capability during training. We conduct extensive experiments across different models and settings to validate our method. We hope our work provides key insights for the LLM RL community. 

## ACKNOWLEDGMENTS

The authors wish to thank the anonymous reviewers for their helpful comments. This work was partially funded by Shanghai Municipal Science and Technology Major Project 2025SHZDZX025G07, National Natural Science Foundation of China (No.62476061, 62576106, 62376061), Shanghai Rising-Star Program (23QA1400200), and Natural Science Foundation of Shanghai (23ZR1403500). 

## ETHICS STATEMENT

This research introduces an RL methodology designed to augment reasoning capabilities. However, we recognize that it may inadvertently strengthen other capabilities, including those with potential for malicious use. We firmly state that this work is intended for ethical and constructive purposes. Users of this method bear the full responsibility for ensuring it is applied in a safe, fair, and harmless manner. Any misuse of this method is strictly against the intent of the authors. 

## REPRODUCIBILITY STATEMENT

We have describe our method and the hyperparameters in §4 and §5. To support reproducibility, we will open-source our code. The datasets used for RL experiments are already publicly available. 

## REFERENCES



AIME. Aime problems and solution, 2025. URL https://artofproblemsolving.com/ wiki/index.php/AIME_Problems_and_Solutions. 





Anthropic. Claude code, 2025. URL https://docs.anthropic.com/en/docs/ claude-code. 





Charles Arnal, Gaetan Narozniak, Vivien Cabannes, Yunhao Tang, Julia Kempe, and R¨ emi Munos.´ Asymmetric REINFORCE for off-policy reinforcement learning: Balancing positive and negative rewards. CoRR, abs/2506.20520, 2025. doi: 10.48550/ARXIV.2506.20520. URL https:// doi.org/10.48550/arXiv.2506.20520. 





Yifan Bai, Yiping Bao, Guanduo Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen, Yanru Chen, Yuankun Chen, Yutian Chen, Zhuofu Chen, Jialei Cui, Hao Ding, Mengnan Dong, Angang Du, Chenzhuang Du, Dikang Du, Yulun Du, Yu Fan, Yichen Feng, Kelin Fu, Bofei Gao, Hongcheng Gao, Peizhong Gao, Tong Gao, Xinran Gu, Longyu Guan, Haiqing Guo, Jianhang Guo, Hao Hu, Xiaoru Hao, Tianhong He, Weiran He, Wenyang He, Chao Hong, Yangyang Hu, Zhenxing Hu, Weixiao Huang, Zhiqi Huang, Zihao Huang, Tao Jiang, Zhejun Jiang, Xinyi Jin, Yongsheng Kang, Guokun Lai, Cheng Li, Fang Li, Haoyang Li, Ming Li, Wentao Li, Yanhao Li, Yiwei Li, Zhaowei Li, Zheming Li, Hongzhan Lin, Xiaohan Lin, Zongyu Lin, Chengyin Liu, Chenyu Liu, Hongzhang Liu, Jingyuan Liu, Junqi Liu, Liang Liu, Shaowei Liu, T. Y. Liu, Tianwei Liu, Weizhou Liu, Yangyang Liu, Yibo Liu, Yiping Liu, Yue Liu, Zhengying Liu, Enzhe Lu, Lijun Lu, Shengling Ma, Xinyu Ma, Yingwei Ma, Shaoguang Mao, Jie Mei, Xin Men, Yibo Miao, Siyuan Pan, Yebo Peng, Ruoyu Qin, Bowen Qu, Zeyu Shang, Lidong Shi, Shengyuan Shi, Feifan Song, Jianlin Su, Zhengyuan Su, Xinjie Sun, Flood Sung, Heyi Tang, Jiawen Tao, Qifeng Teng, Chensi Wang, Dinglu Wang, Feng Wang, and Haiming Wang. Kimi K2: open agentic intelligence. CoRR, abs/2507.20534, 2025. doi: 10.48550/ARXIV.2507.20534. URL https://doi.org/ 10.48550/arXiv.2507.20534. 





Yuxuan Cai, Lu Chen, Qiaoling Chen, Yuyang Ding, Liwen Fan, Wenjie Fu, Yufei Gao, Honglin Guo, Pinxue Guo, Zhenhua Han, et al. Nex-n1: Agentic models trained via a unified ecosystem for large-scale environment construction. arXiv preprint arXiv:2512.04987, 2025. 





Jiangjie Chen, Qianyu He, Siyu Yuan, Aili Chen, Zhicheng Cai, Weinan Dai, Hongli Yu, Qiying Yu, Xuefeng Li, Jiaze Chen, et al. Enigmata: Scaling logical reasoning in large language models with synthetic verifiable puzzles. arXiv preprint arXiv:2505.19914, 2025a. 





Wenxiang Chen, Wei He, Zhiheng Xi, Honglin Guo, Boyang Hong, Jiazheng Zhang, Nijun Li, Tao Gui, Yun Li, Qi Zhang, et al. Better process supervision with bi-directional rewarding signals. In Findings ofthe Associationfor Computational Linguistics: ACL 2025, pp. 14471–14485, 2025b. 





Yang Chen, Zhuolin Yang, Zihan Liu, Chankyu Lee, Peng Xu, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. Acereason-nemotron: Advancing math and code reasoning through reinforcement learning. CoRR, abs/2505.16400, 2025c. doi: 10.48550/ARXIV.2505.16400. URL https://doi.org/10.48550/arXiv.2505.16400. 





Daixuan Cheng, Shaohan Huang, Xuekai Zhu, Bo Dai, Wayne Xin Zhao, Zhenliang Zhang, and Furu Wei. Reasoning with exploration: An entropy perspective. CoRR, abs/2506.14758, 2025. doi: 10.48550/ARXIV.2506.14758. URL https://doi.org/10.48550/arXiv.2506. 14758. 





Franc¸ois Chollet. On the measure of intelligence. arXiv preprint arXiv:1911.01547, 2019. 





Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit S. Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, Luke Marris, Sam Petulla, Colin Gaffney, Asaf Aharoni, Nathan Lintz, Tiago Cardal Pais, Henrik Jacobsson, Idan Szpektor, Nan-Jiang Jiang, Krishna Haridasan, Ahmed Omran, Nikunj Saunshi, Dara Bahri, Gaurav Mishra, Eric Chu, Toby Boyd, Brad Hekman, Aaron Parisi, Chaoyi Zhang, Kornraphop Kawintiranon, Tania Bedrax-Weiss, Oliver Wang, Ya Xu, Ollie Purkiss, Uri Mendlovic, Ila¨ı Deutel, Nam Nguyen, Adam Langley, Flip Korn, Lucia Rossazza, Alexandre Rame, Sagar Wagh-´ mare, Helen Miller, Nathan Byrd, Ashrith Sheshan, Raia Hadsell Sangnie Bhardwaj, Pawel Janus, Tero Rissa, Dan Horgan, Sharon Silver, Ayzaan Wahid, Sergey Brin, Yves Raimond, Klemen Kloboves, Cindy Wang, Nitesh Bharadwaj Gundavarapu, Ilia Shumailov, Bo Wang, Mantas Pajarskas, Joe Heyward, Martin Nikoltchev, Maciej Kula, Hao Zhou, Zachary Garrett, Sushant Kafle, Sercan Arik, Ankita Goel, Mingyao Yang, Jiho Park, Koji Kojima, Parsa Mahmoudieh, Koray Kavukcuoglu, Grace Chen, Doug Fritz, Anton Bulyenov, Sudeshna Roy, Dimitris Paparas, Hadar Shemtov, Bo-Juen Chen, Robin Strudel, David Reitter, Aurko Roy, Andrey Vlasov, Changwan Ryu, Chas Leichner, Haichuan Yang, Zelda Mariet, Denis Vnukov, Tim Sohn, Amy Stuart, Wei Liang, Minmin Chen, Praynaa Rawlani, Christy Koh, JD Co-Reyes, Guangda Lai, Praseem Banzal, Dimitrios Vytiniotis, Jieru Mei, and Mu Cai. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. CoRR, abs/2507.06261, 2025. doi: 10.48550/ARXIV.2507.06261. URL https://doi.org/10.48550/arXiv.2507.06261. 





Ganqu Cui, Yuchen Zhang, Jiacheng Chen, Lifan Yuan, Zhi Wang, Yuxin Zuo, Haozhan Li, Yuchen Fan, Huayu Chen, Weize Chen, Zhiyuan Liu, Hao Peng, Lei Bai, Wanli Ouyang, Yu Cheng, Bowen Zhou, and Ning Ding. The entropy mechanism of reinforcement learning for reasoning language models. CoRR, abs/2505.22617, 2025. doi: 10.48550/ARXIV.2505.22617. URL https://doi.org/10.48550/arXiv.2505.22617. 





Yiwen Ding, Zhiheng Xi, Wei He, Lizhuoyuan Lizhuoyuan, Yitao Zhai, Shi Xiaowei, Xunliang Cai, Tao Gui, Qi Zhang, and Xuan-Jing Huang. Mitigating tail narrowing in llm self-improvement via socratic-guided sampling. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 10627–10646, 2025a. 





Yiwen Ding, Zhiheng Xi, Wei He, Lizhuoyuan Lizhuoyuan, Yitao Zhai, Shi Xiaowei, Xunliang Cai, Tao Gui, Qi Zhang, and Xuanjing Huang. Mitigating tail narrowing in LLM self-improvement via socratic-guided sampling. In Luis Chiruzzo, Alan Ritter, and Lu Wang (eds.), Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL 2025 - Volume 1: Long Papers, Albuquerque, New Mexico, USA, April 29 - May 4, 2025, pp. 10627–10646. Association for Computational Linguistics, 2025b. doi: 10.18653/V1/2025.NAACL-LONG.533. URL https://doi.org/10.18653/v1/2025.naacl-long.533. 





Wei Fu, Jiaxuan Gao, Xujie Shen, Chen Zhu, Zhiyu Mei, Chuyi He, Shusheng Xu, Guo Wei, Jun Mei, Jiashu Wang, Tongkai Yang, Binhang Yuan, and Yi Wu. Areal: A large-scale asynchronous reinforcement learning system for language reasoning. CoRR, abs/2505.24298, 2025. doi: 10. 48550/ARXIV.2505.24298. URL https://doi.org/10.48550/arXiv.2505.24298. 



Google. Introducing gemini 2.0: our new ai model for the agentic era, December 2024. URL https://blog.google/technology/google-deepmind/ google-gemini-ai-update-december-2024/. 

C¸ aglar Gulc¸ehre, Tom Le Paine, Srivatsan Srinivasan, Ksenia Konyushkova, Lotte Weerts, Abhishek¨ Sharma, Aditya Siddhant, Alex Ahern, Miaosen Wang, Chenjie Gu, Wolfgang Macherey, Arnaud Doucet, Orhan Firat, and Nando de Freitas. Reinforced self-training (rest) for language modeling. CoRR, abs/2308.08998, 2023. doi: 10.48550/ARXIV.2308.08998. URL https://doi.org/ 10.48550/arXiv.2308.08998. 



Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Hanwei Xu, Honghui Ding, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jingchang Chen, Jingyang Yuan, Jinhao Tu, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaichao You, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingxu Zhou, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shaoqing Wu, Tao Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun Zha, Yuting Yan, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, and Zhen Zhang. Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nature, 645(8081):633–638, 2025a. doi: 10.1038/s41586-025-09422-z. URL https://doi.org/10.1038/s41586-025-09422-z. 



Xin Guo, Zhiheng Xi, Yiwen Ding, Yitao Zhai, Xiaowei Shi, Xunliang Cai, Tao Gui, Qi Zhang, and Xuanjing Huang. Counteracting matthew effect in self-improvement of lvlms through head-tail re-balancing. arXiv preprint arXiv:2510.26474, 2025b. 



Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, et al. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3828–3850, 2024. 



Jujie He, Jiacai Liu, Chris Yuhao Liu, Rui Yan, Chaojie Wang, Peng Cheng, Xiaoyu Zhang, Fuxiang Zhang, Jiacheng Xu, Wei Shen, Siyuan Li, Liang Zeng, Tianwen Wei, Cheng Cheng, Bo An, Yang Liu, and Yahui Zhou. Skywork open reasoner 1 technical report. CoRR, abs/2505.22312, 2025. doi: 10.48550/ARXIV.2505.22312. URL https://doi.org/10.48550/arXiv. 2505.22312. 

Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, Alex Iftimie, Alex Karpenko, Alex Tachard Passos, Alexander Neitz, Alexander Prokofiev, Alexander Wei, Allison Tam, Ally Bennett, 

Ananya Kumar, Andre Saraiva, Andrea Vallone, Andrew Duberstein, Andrew Kondrich, Andrey Mishchenko, Andy Applebaum, Angela Jiang, Ashvin Nair, Barret Zoph, Behrooz Ghorbani, Ben Rossen, Benjamin Sokolowsky, Boaz Barak, Bob McGrew, Borys Minaiev, Botao Hao, Bowen Baker, Brandon Houghton, Brandon McKinzie, Brydon Eastman, Camillo Lugaresi, Cary Bassin, Cary Hudson, Chak Ming Li, Charles de Bourcy, Chelsea Voss, Chen Shen, Chong Zhang, Chris Koch, Chris Orsinger, Christopher Hesse, Claudia Fischer, Clive Chan, Dan Roberts, Daniel Kappler, Daniel Levy, Daniel Selsam, David Dohan, David Farhi, David Mely, David Robinson, Dimitris Tsipras, Doug Li, Dragos Oprica, Eben Freeman, Eddie Zhang, Edmund Wong, Elizabeth Proehl, Enoch Cheung, Eric Mitchell, Eric Wallace, Erik Ritter, Evan Mays, Fan Wang, Felipe Petroski Such, Filippo Raso, Florencia Leoni, Foivos Tsimpourlas, Francis Song, Fred von Lohmann, Freddie Sulit, Geoff Salmon, Giambattista Parascandolo, Gildas Chabot, Grace Zhao, Greg Brockman, Guillaume Leclerc, Hadi Salman, Haiming Bao, Hao Sheng, Hart Andrin, Hessam Bagherinezhad, Hongyu Ren, Hunter Lightman, Hyung Won Chung, Ian Kivlichan, Ian O’Connell, Ian Osband, Ignasi Clavera Gilaberte, and Ilge Akkaya. Openai o1 system card. CoRR, abs/2412.16720, 2024. doi: 10.48550/ARXIV.2412.16720. URL https://doi.org/10.48550/arXiv.2412.16720. 

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. CoRR, abs/2503.20783, 2025. doi: 10.48550/ARXIV.2503.20783. URL https://doi.org/10.48550/arXiv.2503. 20783. 



Xueguang Ma, Qian Liu, Dongfu Jiang, Ge Zhang, Zejun Ma, and Wenhu Chen. General-reasoner: Advancing llm reasoning across all domains. arXiv preprint arXiv:2505.14652, 2025. 



OpenAI. Openai o3-mini system card, 2025. URL https://cdn.openai.com/ o3-mini-system-card-feb10.pdf. 



Qwen. Qwq-32b: Embracing the power of reinforcement learning, March 2025. URL https: //qwenlm.github.io/blog/qwq-32b/. 





David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling, 2024. 





Nicolas Le Roux, Marc G Bellemare, Jonathan Lebensold, Arnaud Bergeron, Joshua Greaves, Alex Frechette, Carolyne Pelletier, Eric Thibodeau-Laufer, S´ andor Toth, and Sam Work. Ta-´ pered off-policy reinforce: Stable and efficient reinforcement learning for llms. arXiv preprint arXiv:2503.14286, 2025a. 



Nicolas Le Roux, Marc G. Bellemare, Jonathan Lebensold, Arnaud Bergeron, Joshua Greaves, Alexandre Frechette, Carolyne Pelletier, Eric Thibodeau-Laufer, S´ andor T´ oth, and Sam Work.´ Tapered off-policy REINFORCE: stable and efficient reinforcement learning for llms. CoRR, abs/2503.14286, 2025b. doi: 10.48550/ARXIV.2503.14286. URL https://doi.org/10. 48550/arXiv.2503.14286. 



John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. CoRR, abs/1707.06347, 2017. URL http://arxiv.org/abs/ 1707.06347. 





Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. CoRR, abs/2402.03300, 2024. doi: 10.48550/ARXIV.2402.03300. URL https://doi.org/10.48550/arXiv.2402.03300. 





Yunhao Tang, Zhaohan Daniel Guo, Zeyu Zheng, Daniele Calandriello, Yuan Cao, Eugene Tarassov, Remi Munos, Bernardo´ Avila Pires, Michal Valko, Yong Cheng, and Will Dabney. Understanding<sup>´</sup> the performance gap between online and offline alignment algorithms. CoRR, abs/2405.08448, 2024. doi: 10.48550/ARXIV.2405.08448. URL https://doi.org/10.48550/arXiv. 2405.08448. 





Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, Chuning Tang, Congcong Wang, Dehao Zhang, Enming Yuan, Enzhe Lu, Fengxiang Tang, Flood Sung, Guangda Wei, Guokun Lai, Haiqing Guo, Han Zhu, Hao Ding, Hao Hu, Hao Yang, Hao Zhang, Haotian Yao, Haotian Zhao, Haoyu Lu, Haoze Li, Haozhen Yu, Hongcheng Gao, Huabin Zheng, Huan Yuan, Jia Chen, Jianhang Guo, Jianlin Su, Jianzhou Wang, Jie Zhao, Jin Zhang, Jingyuan Liu, Junjie Yan, Junyan Wu, Lidong Shi, Ling Ye, Longhui Yu, Mengnan Dong, Neo Zhang, Ningchen Ma, Qiwei Pan, Qucheng Gong, Shaowei Liu, Shengling Ma, Shupeng Wei, Sihan Cao, Siying Huang, Tao Jiang, Weihao Gao, Weimin Xiong, Weiran He, Weixiao Huang, Wenhao Wu, Wenyang He, Xianghui Wei, Xianqing Jia, Xingzhe Wu, Xinran Xu, Xinxing Zu, Xinyu Zhou, Xuehai Pan, Y. Charles, Yang Li, Yangyang Hu, Yangyang Liu, Yanru Chen, Yejie Wang, Yibo Liu, Yidao Qin, Yifeng Liu, Ying Yang, Yiping Bao, Yulun Du, Yuxin Wu, Yuzhi Wang, Zaida Zhou, Zhaoji Wang, Zhaowei Li, Zhen Zhu, Zheng Zhang, Zhexu Wang, Zhilin Yang, Zhiqi Huang, Zihao Huang, Ziyao Xu, and Zonghan Yang. Kimi k1.5: Scaling reinforcement learning with llms. CoRR, abs/2501.12599, 2025. doi: 10. 48550/ARXIV.2501.12599. URL https://doi.org/10.48550/arXiv.2501.12599. 





Luong Quoc Trung, Xinbo Zhang, Zhanming Jie, Peng Sun, Xiaoran Jin, and Hang Li. Reft: Reasoning with reinforced fine-tuning. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024, pp. 7601–7614. Association for Computational Linguistics, 2024. doi: 10.18653/V1/2024.ACL-LONG.410. URL https://doi.org/10.18653/v1/2024.acl-long.410. 





Shenzhi Wang, Le Yu, Chang Gao, Chujie Zheng, Shixuan Liu, Rui Lu, Kai Dang, Xionghui Chen, Jianxin Yang, Zhenru Zhang, Yuqiong Liu, An Yang, Andrew Zhao, Yang Yue, Shiji Song, Bowen Yu, Gao Huang, and Junyang Lin. Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for LLM reasoning. CoRR, abs/2506.01939, 2025a. doi: 10. 48550/ARXIV.2506.01939. URL https://doi.org/10.48550/arXiv.2506.01939. 





Zengzhi Wang, Fan Zhou, Xuefeng Li, and Pengfei Liu. Octothinker: Mid-training incentivizes reinforcement learning scaling. CoRR, abs/2506.20512, 2025b. doi: 10.48550/ARXIV.2506. 20512. URL https://doi.org/10.48550/arXiv.2506.20512. 





Liang Wen, Yunke Cai, Fenrui Xiao, Xin He, Qi An, Zhenyu Duan, Yimin Du, Junchen Liu, Lifu Tang, Xiaowei Lv, Haosheng Zou, Yongchao Deng, Shousheng Jia, and Xiangzheng Zhang. Light-r1: Curriculum sft, DPO and RL for long COT from scratch and beyond. CoRR, abs/2503.10460, 2025. doi: 10.48550/ARXIV.2503.10460. URL https://doi.org/10. 48550/arXiv.2503.10460. 





Ronald J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Mach. Learn., 8:229–256, 1992. doi: 10.1007/BF00992696. URL https://doi. org/10.1007/BF00992696. 





Zhiheng Xi, Wenxiang Chen, Boyang Hong, Senjie Jin, Rui Zheng, Wei He, Yiwen Ding, Shichun Liu, Xin Guo, Junzhe Wang, Honglin Guo, Wei Shen, Xiaoran Fan, Yuhao Zhou, Shihan Dou, Xiao Wang, Xinbo Zhang, Peng Sun, Tao Gui, Qi Zhang, and Xuanjing Huang. Training large language models for reasoning through reverse curriculum reinforcement learning. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net, 2024a. URL https://openreview.net/forum?id=t82Y3fmRtk. 





Zhiheng Xi, Wenxiang Chen, Boyang Hong, Senjie Jin, Rui Zheng, Wei He, Yiwen Ding, Shichun Liu, Xin Guo, Junzhe Wang, et al. Training large language models for reasoning through reverse curriculum reinforcement learning. arXiv preprint arXiv:2402.05808, 2024b. 





Zhiheng Xi, Dingwen Yang, Jixuan Huang, Jiafu Tang, Guanyu Li, Yiwen Ding, Wei He, Boyang Hong, Shihan Do, Wenyu Zhan, et al. Enhancing llm reasoning via critique models with test-time and training-time supervision. arXiv preprint arXiv:2411.16579, 2024c. 





Zhiheng Xi, Jixuan Huang, Chenyang Liao, Baodai Huang, Honglin Guo, Jiaqi Liu, Rui Zheng, Junjie Ye, Jiazheng Zhang, Wenxiang Chen, et al. Agentgym-rl: Training llm agents for long-horizon decision making through multi-turn reinforcement learning. arXiv preprint arXiv:2509.08755, 2025. 





An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, Keming Lu, Mingfeng Xue, Runji Lin, Tianyu Liu, Xingzhang Ren, and Zhenru Zhang. Qwen2.5-math technical report: Toward mathematical expert model via self-improvement. CoRR, abs/2409.12122, 2024. doi: 10.48550/ARXIV.2409.12122. URL https://doi.org/10.48550/arXiv.2409.12122. 





An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jian Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report. CoRR, abs/2505.09388, 2025a. doi: 10.48550/ARXIV.2505.09388. URL https://doi.org/10.48550/arXiv.2505.09388. 





Shihui Yang, Chengfeng Dou, Peidong Guo, Kai Lu, Qiang Ju, Fei Deng, and Rihui Xin. Dcpo: Dynamic clipping policy optimization. arXiv preprint arXiv:2509.02333, 2025b. 





Yuming Yang, Mingyoung Lai, Wanxu Zhao, Xiaoran Fan, Zhiheng Xi, Mingqi Wu, Chiyue Huang, Jun Zhao, Haijun Lv, Jian Tong, et al. Which reasoning trajectories teach students to reason better? a simple metric of informative alignment. arXiv preprint arXiv:2601.14249, 2026. 





Zhihe Yang, Xufang Luo, Zilong Wang, Dongqi Han, Zhiyuan He, Dongsheng Li, and Yunjian Xu. Do not let low-probability tokens over-dominate in RL for llms. CoRR, abs/2505.12929, 2025c. doi: 10.48550/ARXIV.2505.12929. URL https://doi.org/10.48550/arXiv.2505. 12929. 





Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Weinan Dai, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Mu Qiao, Yonghui Wu, and Mingxuan Wang. DAPO: an opensource LLM reinforcement learning system at scale. CoRR, abs/2503.14476, 2025. doi: 10. 48550/ARXIV.2503.14476. URL https://doi.org/10.48550/arXiv.2503.14476. 





Ruifeng Yuan, Chenghao Xiao, Sicong Leng, Jianyu Wang, Long Li, Weiwen Xu, Hou Pong Chan, Deli Zhao, Tingyang Xu, Zhongyu Wei, Hao Zhang, and Yu Rong. Vl-cogito: Progressive curriculum reinforcement learning for advanced multimodal reasoning. CoRR, abs/2507.22607, 2025. doi: 10.48550/ARXIV.2507.22607. URL https://doi.org/10.48550/arXiv.2507. 22607. 





Kaiyan Zhang, Yuxin Zuo, Bingxiang He, Youbang Sun, Runze Liu, Che Jiang, Yuchen Fan, Kai Tian, Guoli Jia, Pengfei Li, et al. A survey of reinforcement learning for large reasoning models. arXiv preprint arXiv:2509.08827, 2025. 





Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, Jingren Zhou, and Junyang Lin. Group sequence policy optimization. CoRR, abs/2507.18071, 2025. doi: 10.48550/ARXIV.2507.18071. URL https://doi.org/10.48550/arXiv.2507.18071. 





Xuekai Zhu, Daixuan Cheng, Dinghuai Zhang, Hengli Li, Kaiyan Zhang, Che Jiang, Youbang Sun, Ermo Hua, Yuxin Zuo, Xingtai Lv, et al. Flowrl: Matching reward distributions for llm reasoning. arXiv preprint arXiv:2509.15207, 2025. 



## A THE USE OF LARGE LANGUAGE MODELS

LLMs are utilized in this manuscript for partial grammatical checks and language polishing. The authors are fully responsible for the final content. 

## B PERFORMANCE ON OCTOTHINKER-LLAMA

We illustrate the training dynamics on OctoThinker-Llama in Figure 13. Since Llama family models behave badly in RL training, we choose the model after mid-training (Wang et al., 2025b) to show the robustness of BAPO. We can find that BAPO provides consistent and significant improvement in training. For training details, we set the low bound as 0.8-0.9, high bound as 1.2-2.0, and target positive loss contribution as 0.45. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/b0c03305-03db-4992-947b-779a4fb581f3/a62c1d9dac96e2138fca9b1351b1e44080ac0d41b4ba727fb8dcdf45667f3cb4.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/b0c03305-03db-4992-947b-779a4fb581f3/81cf5dad6536d495aa241b6ac8a74718bab080679652a89f2a904f0d63be074f.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/b0c03305-03db-4992-947b-779a4fb581f3/481dfa0fc6fac20f5d4522d9272150160fa06c331b56c42843a76da1e48e738c.jpg)



Figure 13: Training dynamics of OctoThinker-Llama-3B-Long-Zero.


## C PERFORMANCE ON GENERAL REASONER DATASET

We illustrate the training dynamics of R1-Distill-Qwen2.5-7B trained on the General Reasoner Dataset Ma et al. (2025), which spans a wide range of domains such as physics, chemistry, finance, and electronics. Figure 14 shows that BAPO effectively stabilizes the training process and yields substantial gains across these diverse domains. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/b0c03305-03db-4992-947b-779a4fb581f3/19c995841b77880cf718053c97f513722379ca8086b83f586d4cf8837d0ef6aa.jpg)



Figure 14: Training dynamics on General-Reasoner.


## D COMPARISON WITH OTHER ALGORITHMS

Here, we present the results comparing BAPO with other algorithms. 

Table 4: Performance comparison of different RL variants on the BP-Math-7B backbone. BAPO achieves consistent improvements over SFT and other state-of-the-art baselines. 

<table><tr><td>Method</td><td>AIME 24</td><td>AIME 25</td><td>AVG</td></tr><tr><td>SFT</td><td>66.9</td><td>59.0</td><td>62.9</td></tr><tr><td>GRPO</td><td>69.2</td><td>59.2</td><td>64.2</td></tr><tr><td>DAPO</td><td>68.3</td><td>60.8</td><td>64.6</td></tr><tr><td>TOPR</td><td>68.8</td><td>60.4</td><td>64.6</td></tr><tr><td>DCPO</td><td>68.8</td><td>57.9</td><td>63.4</td></tr><tr><td>BAPO (Ours)</td><td>70.8</td><td>62.5</td><td>66.7</td></tr></table>

## E ROBUSTNESS TO HYPERPARAMETERS

1. Positive Token Ratio $\rho _ { 0 } \colon$ We varied $\rho _ { 0 }$ around the default value of 0.4. Results in Table 5 show minimal fluctuation in performance, suggesting the method is robust as long as a reasonable proportion of positive signal is maintained. 


Table 5: Ablation studies on positive token contribution threshold $\rho _ { 0 }$


<table><tr><td><eq>\rho_0</eq></td><td>AIME 2024</td><td>AIME 2025</td><td>Average</td></tr><tr><td>0.4 (default)</td><td>70.8</td><td>62.5</td><td>66.7</td></tr><tr><td>0.45</td><td>69.6</td><td>63.8</td><td>66.7</td></tr><tr><td>0.50</td><td>71.3</td><td>60.8</td><td>66.1</td></tr></table>

2. Clipping Ranges: We tested various intervals for the lower and upper constraints. As shown in Table 6, BAPO consistently achieves high scores, proving its ability to find effective clipping points automatically. 


Table 6: Ablation studies on movable range of clipping bounds.


<table><tr><td><eq>[a_{-}, b_{-}]</eq></td><td><eq>[a_{+}, b_{+}]</eq></td><td>AIME 2024</td><td>AIME 2025</td><td>Average</td></tr><tr><td>[0.6, 0.9] (default)</td><td>[1.2, 3.0] (default)</td><td>70.8</td><td>62.5</td><td>66.7</td></tr><tr><td>[0.5, 0.9]</td><td>[1.2, 3.0]</td><td>70.8</td><td>60.8</td><td>65.8</td></tr><tr><td>[0.6, 0.8]</td><td>[1.2, 3.0]</td><td>71.3</td><td>59.6</td><td>65.5</td></tr><tr><td>[0.6, 0.9]</td><td>[1.0, 3.0]</td><td>71.3</td><td>62.5</td><td>66.9</td></tr><tr><td>[0.6, 0.9]</td><td>[1.2, +∞]</td><td>72.1</td><td>60.4</td><td>66.3</td></tr></table>

3. Step Sizes δ: We analyzed the impact of varying the step sizes used for updating bounds. Table 7 confirms that the algorithm is insensitive to minor variations in $\delta _ { 1 }$ and $\delta _ { 2 }$ 


Table 7: Ablation studies on step sizes $\delta _ { 1 }$ and $\delta _ { 2 }$ .


<table><tr><td><eq>\delta_1</eq></td><td><eq>\delta_2</eq></td><td>AIME 2024</td><td>AIME 2025</td><td>Average</td></tr><tr><td>0.05 (default)</td><td>0.02 (default)</td><td>70.8</td><td>62.5</td><td>66.7</td></tr><tr><td>0.02</td><td>0.02</td><td>71.3</td><td>62.5</td><td>66.9</td></tr><tr><td>0.05</td><td>0.05</td><td>70.8</td><td>62.5</td><td>66.7</td></tr></table>

## F DERIVATION FROM EQUATION 4 TO EQUATION 5

## F.1 RESTATING THE PPO OBJECTIVE (EQUATION 4)

In large-scale reinforcement learning for language models, the PPO surrogate objective is widely used: 

$$
J ^ {\mathrm{PPO}} (\theta) = \mathbb {E} _ {x \sim \mathcal {D}, y \sim \pi_ {\mathrm{rollout}} (\cdot | x)} \left[ \sum_ {t = 1} ^ {T} \min \bigl (r _ {t} \cdot A _ {t}, \operatorname{clip} (r _ {t}, 1 - \varepsilon , 1 + \varepsilon) \cdot A _ {t} \bigr) \right],\tag{9}
$$

where $\begin{array} { r } { r _ { t } = \frac { \pi _ { \theta } ( y _ { t } | x , y _ { < t } ) } { \pi _ { \mathrm { r o l l o u t } } ( y _ { t } | x , y _ { < t } ) } , A _ { t } } \end{array}$ denotes the advantage function, and $\varepsilon \in ( 0 , 1 )$ controls the clipping range. 

## F.2 DECOMPOSING THE PPO LOSS STRUCTURE

Fixing a particular time step t, we can rewrite: 

$$
J ^ {\mathrm{PPO}} (\theta) = \sum_ {x, y _ {t}} \pi_ {\text {rollout}} (y _ {t} \mid x, y _ {<   t}) \cdot \min \bigl (r _ {t} \cdot A _ {t}, \operatorname{clip} (r _ {t}, 1 - \varepsilon , 1 + \varepsilon) \cdot A _ {t} \bigr).\tag{10}
$$

Note that when $r _ { t }$ lies within the clipping interval, $\mathrm { c l i p } ( \boldsymbol { r } _ { t } , \cdot , \cdot ) = \boldsymbol { r } _ { t }$ , and thus the two terms are equal. When $r _ { t }$ exceeds the interval, the second term becomes a constant, resulting in zero gradient with respect to θ. Meanwhile, observe that $r _ { t } \cdot \pi _ { \mathrm { r o l l o u t } } = \pi _ { \theta }$ . Therefore, we can express the objective in a piecewise form based on the sign of $A _ { t } \mathrm { : }$ 

$$
\begin{array}{c} J ^ {\mathrm{PPO}} (\theta) = \sum_ {x, y _ {t}} \Big [ \mathbb {I} \{A _ {t} > 0 \} \cdot \mathbb {I} \{r _ {t} \leq 1 + \varepsilon \} \cdot \pi_ {\theta} (y _ {t} \mid x, y _ {<   t}) \cdot A _ {t} \\ + \mathbb {I} \{A _ {t} <   0 \} \cdot \mathbb {I} \{r _ {t} \geq 1 - \varepsilon \} \cdot \pi_ {\theta} (y _ {t} \mid x, y _ {<   t}) \cdot A _ {t} \Big ]. \end{array}\tag{11}
$$

In this way, the expectation form is converted into a summation over sampled pairs $( x , y _ { t } )$ , and the behavioral policy term is eliminated, paving the way for the gradient derivation in the next section. 

## F.3 TAKING THE GRADIENT WITH RESPECT TO PARAMETERS (EQUATION 5)

For the PPO objective in Equation 11, when differentiating with respect to $\theta ,$ only $\pi _ { \boldsymbol { \theta } } ( y _ { t } \mid x , y _ { < t } )$ depends on θ. Thus we have: 

$$
\begin{array}{c} \nabla_ {\theta} J ^ {\mathrm{PPO}} = \sum_ {x, y _ {t}} \Big [ \mathbb {I} \{A _ {t} > 0 \} \cdot \mathbb {I} \{r _ {t} \leq 1 + \varepsilon \} \cdot A _ {t} \cdot \nabla_ {\theta} \pi_ {\theta} (y _ {t} \mid x, y _ {<   t}) \\ + \mathbb {I} \{A _ {t} <   0 \} \cdot \mathbb {I} \{r _ {t} \geq 1 - \varepsilon \} \cdot A _ {t} \cdot \nabla_ {\theta} \pi_ {\theta} (y _ {t} \mid x, y _ {<   t}) \Big ]. \end{array}\tag{12}
$$

Using the log-likelihood identity $\nabla _ { \theta } \pi _ { \theta } = \pi _ { \theta } \nabla _ { \theta }$ log $\pi _ { \theta } .$ the above becomes: 

$$
\begin{array}{c} \nabla_ {\theta} J ^ {\mathrm{PPO}} = \sum_ {x, y _ {t}} \Big [ \mathbb {I} \{A _ {t} > 0 \} \cdot \mathbb {I} \{r _ {t} \leq 1 + \varepsilon \} \cdot \pi_ {\theta} (y _ {t} \mid x, y _ {<   t}) \cdot A _ {t} \cdot \nabla_ {\theta} \log \pi_ {\theta} (y _ {t} \mid x, y _ {<   t}) \\ + \mathbb {I} \{A _ {t} <   0 \} \cdot \mathbb {I} \{r _ {t} \geq 1 - \varepsilon \} \cdot \pi_ {\theta} (y _ {t} \mid x, y _ {<   t}) \cdot A _ {t} \cdot \nabla_ {\theta} \log \pi_ {\theta} (y _ {t} \mid x, y _ {<   t}) \Big ]. \end{array}\tag{13}
$$

This corresponds precisely to Equation 5 in the PPO derivation: When $A _ { t } ~ > ~ 0$ and $r _ { t } < 1 + \varepsilon .$ or when $A _ { t } < 0$ and $r _ { t } > 1 - \varepsilon .$ , the sample contributes an effective gradient; outside the clipping range, the gradient is suppressed to zero, enforcing smooth policy updates. 

## G PROOFS OF EQUATION 6

## G.1 EXPLANATIONS FOR ALL VARIABLES AND EXPRESSIONS

All notation used in the following justification, including variables and expressions, is provided with detailed explanations in Table 8. 

## G.2 PREPARATION: REWRITE THE PPO DERIVATIVES

To facilitate the justification of the propositions below, we rewrite the PPO loss function in the following form: 

$$
\begin{array}{l} \nabla J ^ {\mathrm{PPO}} = \underbrace {\sum_ {A (y _ {t}) > 0} \pi_ {\theta} (y _ {t}) \cdot \mathbb {I} \{r (y _ {t}) <   1 + \varepsilon \} \cdot A (y _ {t}) \cdot \nabla \log \pi_ {\theta} (y _ {t})} _ {\text {positive tokens}} \\ + \underbrace {\sum_ {A (y _ {t}) <   0} \pi_ {\theta} (y _ {t}) \cdot \mathbb {I} \{r (y _ {t}) > 1 - \varepsilon \} \cdot A (y _ {t}) \cdot \nabla \log \pi_ {\theta} (y _ {t})} _ {\text {negative tokens}} \end{array}
$$

where 

$$
\pi_ {\theta} (y _ {t}) = \pi_ {\theta} (y _ {t} | \boldsymbol {x}, \boldsymbol {y} _ {<   t}), r (y _ {t}) = \frac {\pi_ {\theta} (y _ {t} \mid \boldsymbol {x} , \boldsymbol {y} _ {<   t})}{\pi_ {\theta_ {\text { rollout}}} (y _ {t} \mid \boldsymbol {x} , \boldsymbol {y} _ {<   t})}, A (y _ {t}) = A (y _ {t} | \boldsymbol {x}, \boldsymbol {y} _ {<   t}).
$$


Table 8: Notation used in justification below.


<table><tr><td>Category</td><td>Symbol</td><td>Meaning</td></tr><tr><td rowspan="6">Variables</td><td><eq>\pi_{\theta}</eq></td><td>The policy parameterized by <eq>\theta</eq></td></tr><tr><td><eq>\pi_{\theta_{rollout}}</eq></td><td>The standard sampling policy</td></tr><tr><td><eq>x</eq></td><td>Given prompt</td></tr><tr><td><eq>y</eq></td><td>A T-token response generated by <eq>\pi_{\theta}</eq> when given <eq>x</eq></td></tr><tr><td><eq>y_t</eq></td><td>The t-th token of y</td></tr><tr><td><eq>\eta</eq></td><td>Learning rate</td></tr><tr><td rowspan="10">Expressions</td><td><eq>\pi_{\theta}(\cdot|x, y_{&lt;t})</eq></td><td>Probability of generating token · under policy <eq>\pi_{\theta}</eq> given input <eq>x</eq> and previous tokens <eq>y_{&lt;t}</eq></td></tr><tr><td><eq>\pi_{\theta_{rollout}}(y_t|x, y_{&lt;t})</eq></td><td>Probability of generating token · under standard sampling policy <eq>\pi_{\theta_{rollout}}</eq> given input <eq>x</eq> and previous tokens <eq>y_{&lt;t}</eq></td></tr><tr><td><eq>A(\cdot|x, y_{&lt;t})</eq></td><td>The measurement of how much better(or worse) selecting token · is compared to the expected value under the current policy, given <eq>x</eq> and <eq>y_{&lt;t}</eq></td></tr><tr><td><eq>\mathcal{H}(\cdot|x, y_{&lt;t})</eq></td><td>The information entropy of policy · given <eq>x</eq> and <eq>y_{&lt;t}</eq></td></tr><tr><td><eq>Cov_{y_t \sim \pi_{\theta}(\cdot|x, y_{&lt;t})}(a(y_t), b(y_t))</eq></td><td>The expected covariance of <eq>a(y_t)</eq> and <eq>b(y_t)</eq> over <eq>y_t</eq> sampled from the policy <eq>\pi_{\theta}</eq>, given <eq>x</eq> and <eq>y_{&lt;t}</eq></td></tr><tr><td><eq>\mathbb{I}(a = b)</eq></td><td>Indicator function that equals 1 if <eq>a = b</eq> and 0 otherwise</td></tr><tr><td><eq>Q^{(\pi_{\theta})}(\cdot, x)</eq></td><td>The expected cumulative reward obtained by taking token · given input <eq>x</eq> and previous tokens under policy <eq>\pi_{\theta}</eq></td></tr><tr><td><eq>V^{(\pi_{\theta})}(x)</eq></td><td>The expected return of the new taking token given input <eq>x</eq> and previous tokens under policy <eq>\pi_{\theta}</eq></td></tr><tr><td><eq>z_{y,x}</eq></td><td>A quantity representing the cumulative weight of sequence <eq>y</eq> given input <eq>x</eq> under policy <eq>\pi_{\theta}</eq>, reflecting its contribution to the policy taken at the current optimization step</td></tr><tr><td><eq>\nabla_{\theta_{y_t,x}} J(\theta)</eq></td><td>The gradient of the policy taken with respect to the logit parameter <eq>\theta_{y_t,x}</eq>, representing how the policy <eq>\pi_{\theta}</eq> should be adjusted for token <eq>y_t</eq> given input <eq>x</eq></td></tr></table>

## G.3 PROOFS OF THE MAIN PROPOSITIONS

The following derivation is inspired by the proof framework in Cui et al. (2025). While the original work focuses mainly on the basic gradient formulation of naive REINFORCE to provide a heuristic explanation, our study advances this approach by deriving the gradient expression specific to the PPO objective. This refinement offers a specific, intuitive yet theoretical account of how policy entropy is intrinsically shaped by the interaction between token-level advantages and their sampling probabilities. 

## G.3.1 PRECLAIMS

Proofs of these three lemmas below are available in Cui et al. (2025). 

Lemma 1. Let the actor policy π<sub>θ</sub> be a tabular softmax policy, the difference ofinformation entropy given prompt x between two consecutive steps k and $k + 1$ satisfies 

$$
\mathcal {H} (\pi_ {\theta} ^ {k + 1} | \boldsymbol {x}, \boldsymbol {y} _ {<   t}) - \mathcal {H} (\pi_ {\theta} ^ {k} | \boldsymbol {x}, \boldsymbol {y} _ {<   t}) \approx - \operatorname{Cov} _ {y _ {t} \sim \pi_ {\theta} ^ {k} (\cdot | \boldsymbol {x}, \boldsymbol {y} _ {<   t})} \left(\log \pi_ {\theta} ^ {k} (y _ {t}), z _ {\boldsymbol {y}, \boldsymbol {x}} ^ {k + 1} - z _ {\boldsymbol {y}, \boldsymbol {x}} ^ {k}\right).
$$

Lemma 2 (Derivative of softmax function). 

$$
\frac {\partial \log \pi_ {\theta} (y _ {t})}{\partial \theta_ {y _ {t} ^ {\prime} , \pmb {x}}} = \mathbb {I} \{y _ {t} = y _ {t} ^ {\prime} \} - \pi_ {\theta} (y _ {t} ^ {\prime})
$$

Lemma 3 (Expectation of Advantage function given prompt x). 

$$
\begin{array}{r l} & {\mathbb {E} _ {y _ {t} \sim \pi_ {\theta} (\cdot | x, \pmb {y} _ {<   t})} \big [ A ^ {\pi_ {\theta}} (y _ {t}) \big ] = \mathbb {E} _ {y _ {t} \sim \pi_ {\theta} (\cdot | x, \pmb {y} _ {<   t})} \big [ Q ^ {\pi_ {\theta}} (y _ {t}, \pmb {x}) - V ^ {\pi_ {\theta}} (\pmb {x}) \big ]} \\ & {\qquad = \mathbb {E} _ {y _ {t} \sim \pi_ {\theta} (\cdot | x, \pmb {y} _ {<   t})} \big [ Q (y _ {t}, \pmb {x}) \big ] - \mathbb {E} _ {y _ {t} \sim \pi_ {\theta} (\cdot | x, \pmb {y} _ {<   t})} \big [ V (\pmb {x}) \big ]} \\ & {\qquad = V (\pmb {x}) - V (\pmb {x})} \\ & {\qquad = 0} \end{array}
$$

## G.3.2 PRINCIPLE PROPOSITIONS

Proposition 1: Assume the actor policy $\pi _ { \theta }$ follows a tabular softmax policy and is optimized via the PPO objective, the difference of $z _ { y , x }$ between two consecutive steps k and k+1 satisfies 

$$
z _ {\boldsymbol {y}, \boldsymbol {x}} ^ {k + 1} - z _ {\boldsymbol {y}, \boldsymbol {x}} ^ {k} = \eta \cdot \pi_ {\theta} (y _ {t}) \cdot [ A (y _ {t}) \cdot \mathcal {X} (y _ {t}) + C ],
$$

where 

$$
\mathcal {X} (y _ {t}) = \left\{ \begin{array}{l l} 1, & \text { if } A (y _ {t}) > 0 \& r (y _ {t}) <   1 + \epsilon \\ & \text { or } A (y _ {t}) <   0 \& r (y _ {t}) > 1 - \epsilon \\ 0, & \text { otherwise } \end{array} \right.
$$

and C includes all clauses irrelevant to $y _ { t }$ 

It is worth noting that $\pmb { \mathcal { X } } ( y _ { t } ) = \mathbf { 0 }$ if and only if ${ \mathbf { } } _ { { \mathbf { } } _ { } } { \mathbf { } } _ { { \mathbf { } } _ { } } { \mathbf { } } _ { { \mathbf { } } _ { } } { \mathbf { } } _ { { \mathbf { } } _ { } } { \mathbf { } } _ { { \mathbf { } } _ { } } { \mathbf { } } _ { { \mathbf { } } _ { } } { \mathbf { } } _ { { \mathbf { } } _ { } } { \mathbf { } } _ { { \mathbf { } } } { \mathbf { } } _ { { \mathbf { } } _ { } } { \mathbf { } } _ { { \mathbf { } } } { \mathbf { } } _ { { \mathbf { } } } { \mathbf { } } _ { { \mathbf { } } } { \mathbf { } } _ { { } } { \mathbf { } } _ { { \mathbf { } } } { \mathbf { } } _ { { } } { \mathbf { } } _ { { } } { \mathbf { } } _ { { } } { \mathbf { } } _ { { } } { \mathbf { } } _ { { } } { \mathbf { } } _ { { } } { \mathbf { } } _ { { } } { \mathbf { } } _ { { } } { \mathbf { } } _ { { } } { \mathbf { } } _ { { } } { \mathbf { } } _ { { } } { \mathbf { } } _ { { } } { \mathbf { } } _ { { } } { \mathbf { } } _ { { } } { \mathbf { } } _ { { } \mathbf { } } _ { { } } { \mathbf { } } _ { { } \mathbf { } } _ { { } \mathbf { } } _ { { } \mathbf { } } _ { { } \mathbf { } } _ { } { \mathbf } _ { { } } _ { { } \mathbf { } } _ { } { \mathbf } _ { { } \mathbf { } } _ { } { \mathbf } _ { } { \mathbf } _ { } { } _ { \mathbf { } } _ { } { \mathbf } _ { } { \mathbf } _ { } _ { } { \mathbf } _ { } { \mathbf } _ { } _ { } \mathbf { } _ { } \mathbf { } _ { } \mathbf _ { } { } _ \mathbf { } _ { } \mathbf { } _ { } _ { \mathbf } _ { } \mathbf { } _ { } \mathbf _ { } \mathbf { } _ { } $ is clipped. 

Proof. In tabular softmax policy, each trajectory-prompt pair $( { \pmb y } , { \pmb x } )$ is associated with an individual logit parameter $z _ { y , x } = \theta _ { y _ { t } , x }$ . Through gradient backtracking, $z _ { y , x }$ is updated via $z _ { y , x } ^ { k + 1 } = z _ { y , x } ^ { k } + \eta \cdot$ $\nabla _ { \theta _ { y _ { t } , x } } J ( \theta )$ . According to the loss function of PPO, we have 

$$
\begin{array}{l}z_{\boldsymbol {y},\boldsymbol{x}}^{k + 1} - z_{\boldsymbol {y},\boldsymbol{x}}^{k} = \eta \cdot \nabla_{\theta_{y_{t},x}}J_{PPO}(\theta)\\ = \eta \cdot \mathbb{E}_{\substack{y_{t}^{\prime}\sim \pi_{\theta}(\cdot |\boldsymbol {x},\boldsymbol {y} <   t)\\ A(y_{t}^{\prime}) > 0}}\left[ \mathbb{I}\{r(y_{t}^{\prime}) <   1 + \varepsilon \} \cdot \nabla_{\theta_{y_{t},x}}\log \pi_{\theta}(y_{t}^{\prime})\cdot A(y_{t}^{\prime}) \right]\\ +\eta \cdot \mathbb{E}_{\substack{y_{t}^{\prime}\sim \pi_{\theta}(\cdot |\boldsymbol {x},\boldsymbol {y} <   t)\\ A(y_{t}^{\prime}) <   0}}\left[ \mathbb{I}\{r(y_{t}^{\prime}) > 1 - \varepsilon \} \cdot \nabla_{\theta_{y_{t},x}}\log \pi_{\theta}(y_{t}^{\prime})\cdot A(y_{t}^{\prime}) \right]\\ = \underbrace{\eta\cdot\mathbb{E}_{y_{t}^{\prime}\sim\pi_{\theta}(\cdot|\boldsymbol {x},\boldsymbol{y}<   t)}\left[ \nabla_{\theta_{y_{t},x}}\log\pi_{\theta}(y_{t}^{\prime})\cdot A(y_{t}^{\prime}) \right]}_ {\text{①}}\\ -\underbrace{\eta\cdot\mathbb{E}_{y_{t}^{\prime}\sim\pi_{\theta}(\cdot|\boldsymbol {x},\boldsymbol{y}<   t)}\left[ \mathbb{I}\{r(y_{t}^{\prime}) > 1 + \varepsilon \} \cdot \nabla_{\theta_{y_{t},x}}\log\pi_{\theta}(y_{t}^{\prime})\cdot A(y_{t}^{\prime}) \right]}_ {\text{②}}\\ -\underbrace{\eta\cdot\mathbb{E}_{y_{t}^{\prime}\sim\pi_{\theta}(\cdot|\boldsymbol {x},\boldsymbol{y}<   t)}\left[ \mathbb{I}\{r(y_{t}^{\prime}) <   1 - \varepsilon \} \cdot \nabla_{\theta_{y_{t},x}}\log\pi_{\theta}(y_{t}^{\prime})\cdot A(y_{t}^{\prime}) \right]}_ {\text{③}}\\ = \\ (\text{②} +\\ (3)) \end{array}\tag{8}
$$

We first perform the derivation on the term marked as $\textcircled{1} \colon$ 

$$
\begin{array}{l} \text {①} = \eta \cdot \mathbb {E} _ {y _ {t} ^ {\prime} \sim \pi_ {\theta} (\cdot | \boldsymbol {x}, \boldsymbol {y} <   t)} \left[ \frac {\partial \log \pi_ {\theta} (y _ {t} ^ {\prime})}{\partial \theta_ {y _ {t} , \boldsymbol {x}}} \cdot A (y _ {t} ^ {\prime}) \right] \\ \stackrel {{\text {Lemma 2}}} {{=}} \eta \cdot \sum_ {y _ {t} ^ {\prime}} [ \pi_ {\theta} (y _ {t} ^ {\prime}) \cdot (\mathbb {I} \{y _ {t} ^ {\prime} = y _ {t} \} - \pi_ {\theta} (y _ {t})) \cdot A (y _ {t} ^ {\prime}) ] \\ = \eta \cdot \pi_ {\theta} (y _ {t}) \cdot \left[ (1 - \pi_ {\theta} (y _ {t})) \cdot A (y _ {t}) - \sum_ {y _ {t} ^ {\prime} \neq y _ {t}} \pi_ {\theta} (y _ {t} ^ {\prime}) \cdot A (y _ {t} ^ {\prime}) \right] \\ = \eta \cdot \pi_ {\theta} (y _ {t}) \cdot \left[ A (y _ {t}) - \sum_ {y _ {t} ^ {\prime}} \pi_ {\theta} (y _ {t} ^ {\prime}) \cdot A (y _ {t} ^ {\prime}) \right] \\ \stackrel {{\text {Lemma 3}}} {{=}} \eta \cdot \pi_ {\theta} (y _ {t}) \cdot [ A (y _ {t}) - 0 ] \\ = \eta \cdot \pi_ {\theta} (y _ {t}) \cdot A (y _ {t}) \end{array}
$$

To keep the presentation concise, we provide only the resulting derivations of Term $\textcircled{2}$ and $\textcircled{3}$ as the detailed steps follow similarly to those for Term 1 . 

$$
\begin{array}{l} \text {②} + \text {③} = \eta \cdot \pi_ {\theta} (y _ {t}) \cdot A (y _ {t}) \cdot (1 - \mathcal {X} (y _ {t})) \\ \qquad - \eta \cdot \pi_ {\theta} (y _ {t}) \cdot \sum_ {A (y _ {t} ^ {\prime}) > 0} [ \mathbb {I} \{r (y _ {t} ^ {\prime}) > 1 + \varepsilon \} \cdot \pi_ {\theta} (y _ {t} ^ {\prime}) \cdot A (y _ {t} ^ {\prime}) ] \\ \qquad - \eta \cdot \pi_ {\theta} (y _ {t}) \cdot \sum_ {A (y _ {t} ^ {\prime}) <   0} [ \mathbb {I} \{r (y _ {t} ^ {\prime}) <   1 - \varepsilon \} \cdot \pi_ {\theta} (y _ {t} ^ {\prime}) \cdot A (y _ {t} ^ {\prime}) ] \end{array}
$$

By substituting the results of the above derivation into Clause (8), we observe that: 

$$
\begin{array}{l} (8) = \text {①} - (\text {②} + \text {③}) \\ \qquad = \eta \cdot \pi_ {\theta} (y _ {t}) \cdot \Big \{A (y _ {t}) \cdot \mathcal {X} (y _ {t}) \\ \qquad + \sum_ {A (y _ {t} ^ {\prime}) > 0} [ \mathbb {I} \{r (y _ {t} ^ {\prime}) > 1 + \varepsilon \} \cdot \pi_ {\theta} (y _ {t} ^ {\prime}) \cdot A (y _ {t} ^ {\prime}) ] \\ \qquad + \sum_ {A (y _ {t} ^ {\prime}) <   0} [ \mathbb {I} \{r (y _ {t} ^ {\prime}) <   1 - \varepsilon \} \cdot \pi_ {\theta} (y _ {t} ^ {\prime}) \cdot A (y _ {t} ^ {\prime}) ] \Big \} \end{array}
$$

By grouping all elements unrelated to $y _ { t }$ into $C ,$ , we are able to successfully establish our proposition. 

Building on Proposition 1, we establish the relationship between policy entropy and the covariance of specific tokens, which is stated as Proposition 2 below. 

Proposition 2 (Equation ${ \bf 6 ) : }$ Let the actor policy $\pi _ { \theta }$ be tabular softmax policy, and $\pi _ { \theta }$ is updated via PPO objective, the difference of information entropy given prompt $x$ and trajectory part $y _ { < t }$ between two consecutive steps k and k+1 satisfies 

$$
\mathcal {H} \left(\pi_ {\theta} ^ {k + 1} | \boldsymbol {x}, \boldsymbol {y} _ {<   t}\right) - \mathcal {H} \left(\pi_ {\theta} ^ {k} | \boldsymbol {x}, \boldsymbol {y} _ {<   t}\right) \approx - \eta \cdot \operatorname{Cov} _ {y _ {t} \sim \pi_ {\theta} ^ {k} (\cdot | \boldsymbol {x}, \boldsymbol {y} _ {<   t})} \left(\log \pi_ {\theta} ^ {k} (y _ {t}), A (y _ {t}) \cdot \mathcal {X} (y _ {t}) + C\right).
$$

Proof. Leveraging the conclusions of Lemma 1 and Proposition 1, we find that, under policy optimization and iteration via the PPO algorithm, the following relationship is satisfied: 

$$
z _ {\boldsymbol {y}, \boldsymbol {x}} ^ {k + 1} - z _ {\boldsymbol {y}, \boldsymbol {x}} ^ {k} = \eta \cdot (A (y _ {t}) \cdot \mathcal {X} (y _ {t}) + C).
$$

Applying this into Lemma 1, we have 

$$
\mathcal {H} \left(\pi_ {\theta} ^ {k + 1} | \boldsymbol {x}, \boldsymbol {y} _ {<   t}\right) - \mathcal {H} \left(\pi_ {\theta} ^ {k} | \boldsymbol {x}, \boldsymbol {y} _ {<   t}\right) \approx - \eta \cdot \operatorname{Cov} _ {y _ {t} \sim \pi_ {\theta} ^ {k} (\cdot | \boldsymbol {x}, \boldsymbol {y} _ {<   t})} \left(\log \pi_ {\theta} ^ {k} (y _ {t}), A (y _ {t}) \cdot \mathcal {X} (y _ {t}) + C\right).
$$

## G.4 ANALYSIS

## G.4.1 DIRECT ANALYSIS: WHY VARYING ε ALTERS ENTROPY?

We begin by examining the covariance of the clipped token, denoted as α. 

Based on the observation stated above, the contribution of α to the entropy can be expressed as: 

$$
- \eta \cdot \pi_ {\theta} ^ {k} (\alpha) \cdot \operatorname{Cov} \left(\log \pi_ {\theta} ^ {k} (\alpha), C\right) = 0,
$$

which indicates that only the retained tokens contribute to the overall entropy. 

In other words, we manipulate the number of tokens that can contribute to the entropy by altering the parameter ε. 

## G.4.2 ADVANCED ANALYSIS : WHICH TYPE OF TOKENS MATTER MOST FOR ENTROPY?

To understand how individual tokens contribute to the overall entropy, we first revisit the Proposition G.3.2 established above. In this section, we provide a more precise definition of tokens with low/high probabilities and advantages. It should be noted that in the analysis experiment (Figure 5), we adopt the naive REINFORCE algorithm without clipping. Consequently, tokens with high or low advantages are defined according to the sign of their advantage values, $\mathrm { i . e . , > 0 }$ for high advantage and $< 0$ for low advantage. 

$$
\begin{array}{r l} & {\mathcal {H} (\pi_ {\theta} ^ {k + 1} | \pmb {x}, \pmb {y} _ {<   t}) - \mathcal {H} (\pi_ {\theta} ^ {k} | \pmb {x}, \pmb {y} _ {<   t}) \approx - \eta \cdot \mathrm{Cov} _ {y _ {t} \sim \pi_ {\theta} ^ {k} (\cdot | \pmb {x}, \pmb {y} _ {<   t})} \left(\log \pi_ {\theta} ^ {k} (y _ {t}), A (y _ {t}) \cdot \mathcal {X} (y _ {t}) + C\right)} \\ & {\qquad = - \eta \cdot \sum_ {p = 1} ^ {T} \pi_ {\theta} ^ {k} (y _ {p} | \pmb {x}, \pmb {y} _ {<   t}) \cdot \left(\log \pi_ {\theta} ^ {k} (y _ {p}) - \mathbb {E} _ {y _ {i} \sim \pi_ {\theta} ^ {k} (\cdot | \pmb {x}, \pmb {y} _ {<   t})} [ \log \pi_ {\theta} ^ {k} (y _ {i}) ]\right)} \\ & {\qquad \cdot \left(A (y _ {p}) \cdot \mathcal {X} (y _ {p}) - \mathbb {E} _ {y _ {i} \sim \pi_ {\theta} ^ {k} (\cdot | \pmb {x}, \pmb {y} _ {<   t})} [ A (y _ {i}) \cdot \mathcal {X} (y _ {i}) ]\right).} \end{array}
$$

where T is the size of the dictionary. 

For convenience, we denote $\mathbb { E } _ { y _ { i } }$ as $\mathbb { E } _ { y _ { i } \sim \pi _ { \theta } ^ { k } ( \cdot | \boldsymbol { x } , y _ { < t } ) }$ . As only retained tokens contribute to the entropy, we focus only on tokens that are not clipped. We begin by making the following simplification: 

$$
\mathbb {E} _ {y _ {i}} \left(A \left(y _ {i}\right) \cdot \mathcal {X} \left(y _ {i}\right)\right) = \mathbb {E} _ {y _ {\text { clipped }}} \left(A \left(y _ {i}\right) \cdot 0\right) + \mathbb {E} _ {y _ {\text { retained }}} \left(A \left(y _ {i}\right) \cdot 1\right) = \mathbb {E} _ {y _ {\text { retained }}} \left(A \left(y _ {i}\right)\right).
$$

So for a selected token $y _ { s } ,$ , its contribution to the overall entropy can be expressed as: 

$$
- \eta \cdot \pi_ {\theta} (y _ {s}) \cdot (\log \pi_ {\theta} (y _ {s}) - \mathbb {E} _ {y _ {i}} (\log \pi_ {\theta} (y _ {i}))) \cdot (A (y _ {s}) - \mathbb {E} _ {y _ {\text {retained}}} A (y _ {\text {retained}})).
$$

Next, we analyze how different types of tokens contribute to the overall entropy. To avoid ambiguity, we first give strict definitions that distinguish between tokens with high/low probabilities and tokens with high/low advantages. 

Definition 1. For a token $y _ { s } ,$ , we classify it as follows: 

• High advantage: if 

$$
A (y _ {s}) > \mathbb {E} _ {y _ {\text { retained }}} A (y _ {\text { retained }})
$$

Otherwise, it is called low advantage. 

• High probability: if 

$$
\pi_ {\theta} (y _ {s}) > \exp (\mathbb {E} _ {y _ {i}} (\log \pi_ {\theta} (y _ {i})))))
$$

Otherwise, it is called low probability. 

Secondly, we present two propositions that directly follow from the above definitions. 

Proposition 3. For a token $y _ { s } ,$ we have 

$$
A (y _ {s}) - \mathbb {E} _ {y _ {\text {retained}}} A (y _ {\text {retained}}) \left\{ \begin{array}{l l} > 0, & \text {if y_{s} is a high - advantage token}, \\ <   0, & \text {if y_{s} is a low - advantage token}. \end{array} \right.
$$

Proposition 4. For a token $y _ { s } ,$ , we have 

$$
\pi_ {\theta} (y _ {s}) \cdot (\log \pi_ {\theta} (y _ {s}) - \mathbb {E} _ {y _ {i}} (\log \pi_ {\theta} (y _ {i}))) \left\{ \begin{array}{l l} > 0, & \text {if y_{s} is a high - probability token}, \\ <   0, & \text {if y_{s} is a low - probability token}. \end{array} \right.
$$

Proof. Let us denote 

$$
C = \mathbb {E} _ {y _ {i}} (\log \pi_ {\theta} (y _ {i}))),
$$

which is independent of $y _ { s }$ , and let $x = \pi _ { \theta } ( y _ { s } )$ . As $\pi _ { \theta } ( y ) < 1$ for every $y , C < 0$ Consider the function 

$$
f (x) = x \cdot (\log (x) - C).
$$

Figure 15 illustrates the behavior of this function. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/b0c03305-03db-4992-947b-779a4fb581f3/24cad9ceef12ca95b7e511ebd396992da915353fe633fe658aa994ecf873b3d9.jpg)



Figure 15: Graph of the function $f ( x ) = x ( \log x - C )$


The proposition follows directly from the properties of $f ( x )$ as observed in the figure. □ 

Due to the propositions given above, we have the table below: 

$$
\Delta \mathcal {H} (y _ {s}) \approx - \eta \cdot \underbrace {\pi_ {\theta} (y _ {s}) \cdot (\log \pi_ {\theta} (y _ {s}) - \mathbb {E} _ {y _ {i}} (\log \pi_ {\theta} (y _ {i})))} _ {④} \cdot \underbrace {(A (y _ {s}) - \mathbb {E} _ {y _ {\text {retained}}} A (y _ {\text {retained}})} _ {⑤}
$$


Table 9: Influence of token characteristics on $\Delta \mathcal { H } ( y _ { s } )$ . The “prob” denotes the probability $\pi _ { \theta } ( y _ { s } )$ and the $\mathrm { \ddot { a d v } \vec { \Omega } }$ represents the advantage $A ( y _ { s } )$


<table><tr><td>Token properties</td><td>4</td><td>5</td><td><eq>\Delta \mathcal{H}(y_s) (-\eta \cdot 4 \cdot 5)</eq></td></tr><tr><td>high prob, high adv</td><td>&gt;0</td><td>&gt;0</td><td>&lt;0</td></tr><tr><td>high prob, low adv</td><td>&gt;0</td><td>&lt;0</td><td>&gt;0</td></tr><tr><td>low prob, high adv</td><td>&lt;0</td><td>&gt;0</td><td>&gt;0</td></tr><tr><td>low prob, low adv</td><td>&lt;0</td><td>&lt;0</td><td>&lt;0</td></tr></table>

Therefore, we observe that tokens which are positive with high probabilities and high advantages, or negative with low probabilities and low advantages, contribute to a reduction in the overall entropy. Conversely, positive tokens with high probabilities but low advantages, and negative tokens with high probabilities but low advantages, contribute to an increase in the overall entropy. This observation justifies the statement made in the main part of the thesis. 

It should be noted that a token $y _ { s }$ decreases the entropy i $\left( \Delta \mathcal { H } ( y _ { s } ) < 0 \right)$ , and increases it otherwise. 