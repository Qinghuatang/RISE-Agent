# See-Saw Modality Balance: See Gradient, and Sew Impaired Vision-Language Balance to Mitigate Dominant Modality Bias

JuneHyoung Kwon<sup>1*</sup>, MiHyeon Kim<sup>1*†</sup>, Eunju Lee<sup>2</sup>, Juhwan Choi<sup>1</sup>, YoungBin Kim<sup>1,2</sup> 

<sup>1</sup> Department of Artificial Intelligence, Chung-Ang University 

<sup>2</sup> Graduate School of Advanced Imaging Sciences, Multimedia and Film, Chung-Ang University {dirchdmltnv, mh10967, dmswn5829, gold5230, ybkim85}@cau.ac.kr 

## Abstract

Vision-language (VL) models have demonstrated strong performance across various tasks. However, these models often rely on a specific modality for predictions, leading to “dominant modality bias.” This bias significantly hurts performance, especially when one modality is impaired. In this study, we analyze model behavior under dominant modality bias and theoretically show that unaligned gradients or differences in gradient magnitudes prevent balanced convergence of the loss. Based on these findings, we propose a novel framework, BALGRAD to mitigate dominant modality bias. Our approach includes inter-modality gradient reweighting, adjusting the gradient of KL divergence based on each modality’s contribution, and inter-task gradient projection to align task directions in a non-conflicting manner. Experiments on UPMC Food-101, Hateful Memes, and MM-IMDb datasets confirm that BALGRAD effectively alleviates over-reliance on specific modalities when making predictions. 

## 1 Introduction

Vision-language (VL) models combine image and text modalities, resulting in powerful multi-modal representations. Owing to this integration of two modalities, these models can achieve higher performance in vision-language tasks. Recently, leveraging extensive datasets, VL models have demonstrated remarkable performance across various tasks such as image captioning (Hu et al., 2022), visual question answering (Khademi et al., 2023), and cross-modal retrieval (Liu et al., 2023), showcasing their capability to harness the complementary strengths of visual and textual data. 

However, these models often rely on a single modality rather than treating and utilizing them 

Illustration of Dominant Modality Bias 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/ceb22bb8-93ba-472a-b9f2-21044acb3d25/01268bdbc6ead3cbe8ad22a89590029446d17a278e81cfb31fc70c5a5bb13c6d.jpg)



(a) Hate Recongition



(b) Food Classification



Figure 1: Conceptual visualization of dominant modality bias. The key modality differs by task: (a) For the hate recognition task, text descriptions of memes lead, while (b) for the food classification task, food images play a crucial role in prediction.


equally, leading to the dominance of a certain modality on the overall performance. A conceptual overview of this effect can be seen in Figure 1. This phenomenon, where a specific modality disproportionately influences the model’s outcomes, is referred to as “dominant modality bias” (Woo et al., 2023). For instance, VL models tend to be biased towards the text modality when recognizing hate expressions (Kiela et al., 2020; Aggarwal et al., 2024), thereby limiting the VL model’s ability to effectively integrate and interpret images. 

This bias behaves particularly detrimentally when one modality is impaired, such as when data is noisy and it is difficult to gather paired data (Garg et al., 2022; Woo et al., 2023; Yang et al., 2024). This issue is common in real-world scenarios due to privacy-related data sharing restrictions or stringent data storage policies (Voigt and Von dem Bussche, 2017) and can severely degrade the model’s performance. Additionally, the failure to sufficiently explore the weak modality limits the overall performance of the VL model (Wang et al., 2020; Huang et al., 2022; Peng et al., 2022), highlighting the need for robust solutions to mitigate dominant modality bias. 

To address this issue and balance the information between modalities, numerous studies have been conducted. Several studies have focused on modulating the gradients of each encoder based on the confidence of individual modalities (Peng et al., 2022; Li et al., 2023). Other approaches have involved training multimodal models using the best-performing learning rates from unimodal models (Yao and Mihalcea, 2022). However, these methods often induce negative transfer (Wang et al., 2019; Yu et al., 2020), which occurs when the model’s performance decreases with the addition of modality data compared to solely using unimodal data. 

We first analyze the behavior of models after the dominant modality bias has taken root. Our analysis reveals that certain modalities are more crucial to target performance and observes that the dominant and weak modalities converge at different rates during training. Additionally, we theoretically demonstrate that the balanced convergence of the loss is influenced by both the magnitude and direction of the gradient. Based on these findings, we propose BALGRAD (Balancing Gradients) to mitigate dominant modality bias. Firstly, we adopt a mutual KL divergence between the predictions of each modality to ensure balanced updates. However, a naive approach that equally aligns the distributions of two modalities can hinder the representation learning of each modality. To address this, we introduce inter-modality gradient reweighting, which adjusts the magnitude of the gradient of the KL divergence term based on the learning status of each modality. Additionally, we propose inter-task gradient projection, which updates the gradient of the target task to establish a balance between both modalities. We project the target task’s gradient in a direction orthogonal to the KL divergence gradient if a conflict between the gradients occurs, encouraging stabilized training between the two modalities. 

We evaluate the effectiveness of BALGRAD on models using three vision-language datasets: UPMC Food-101 (Wang et al., 2015), Hateful Memes (Kiela et al., 2020), and MM-IMDb (Arevalo et al., 2017). To simulate the influence of individual modalities, we conduct experiments under conditions where specific modalities are missing or impaired by noise. The experimental results demonstrate that the proposed method reduces the gap between the modalities while avoiding negative transfer. The contributions of our proposed method are as follows: 

• We analyze the dominant modality bias and theoretically demonstrate that the balanced convergence of loss is influenced by both the magnitude and direction of the gradient. 

• We propose BALGRAD, which reweights the gradients between modalities to ensure stable convergence and projects the target task’s gradient to avoid conflicts that hinder balanced learning. 

• Experimental results across UPMC Food-101, Hateful Memes, and MM-IMDb under different impaired conditions confirm the effectiveness of our proposed method in mitigating dominant modality bias. 

## 2 Related Work

In multimodal models, such as VL models, a bias towards a preferred or easier-to-learn modality often leads to the under-exploration of others (Wang et al., 2020; Huang et al., 2022; Peng et al., 2022). Studies have analyzed this, noting that multimodal models are prone to overfitting and show discrepancies in generalization across modalities (Wang et al., 2020). Differences in convergence speeds also contribute to this bias (Yao and Mihalcea, 2022; Wu et al., 2022). An early study in this field finds that certain modalities, correlating with their network’s random initialization, dominate the learning process (Huang et al., 2022), while other researchers attribute the preference to unimodal representation margins and insufficient integration of modalities (Yang et al., 2024). Another line of study highlights that spurious correlations with instance labels cause imbalances in modality utilization (Guo et al., 2023). In this paper, we identify that the dominant modality bias in VL models arises from the influence of gradient magnitude and direction on the model’s loss function, hindering balanced learning across modalities. 

In response to the challenge of balancing modalities in multimodal learning, various strategies have been proposed. MSLR suggests using different optimal learning rates for each modality during multimodal learning to enhance performance (Yao and Mihalcea, 2022). Another approach involves using a conditional utilization rate to re-scale modality features, ensuring balanced contributions from each modality (Wu et al., 2022). Gradient blending optimizes the mixing of modalities based on the model’s overfitting behavior (Wang et al., 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/ceb22bb8-93ba-472a-b9f2-21044acb3d25/eeef51781793a43fd0f99e3e36ac9832c1f6d70fb6d7f27fd47c53f1c0600848.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/ceb22bb8-93ba-472a-b9f2-21044acb3d25/b36f34be43119de544f52f238d075575ff8550109486d058fa82079936963190.jpg)



(a)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/ceb22bb8-93ba-472a-b9f2-21044acb3d25/a4b6edb71a813110dd49a251a13dbb272f6f1df01bea42bc61b301822ed237a7.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/ceb22bb8-93ba-472a-b9f2-21044acb3d25/76d85da931050a46d6395ce601ab1585e5e63a1c37a75cd8e5e4dd3af37038a0.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/ceb22bb8-93ba-472a-b9f2-21044acb3d25/3e0950d8322854dcc5ab70c4d3372a3371175e38894d97de8c1879984e16e9ac.jpg)



(b)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/ceb22bb8-93ba-472a-b9f2-21044acb3d25/d0905adaaaf53ca917c6e686488b13afed50be02d4415d590b97020b3c77c8dd.jpg)



Figure 2: Experimental results on the UPMC Food-101, Hateful Memes, and MM-IMDb datasets in the presence of dominant modality bias. (a) Performance visualization under different missing conditions (full, image only (missing text), text only (missing image)) for each dataset. (b) Illustration of learning curves for each modality across datasets.


2020). OGM-GE adaptively controls the optimization process using modality-specific confidence scores (Peng et al., 2022). AGM employs Shapley values to modulate gradients through mono-modal responses, aiming to balance the learning process across modalities (Li et al., 2023). However, these methods often lack consideration of negative transfer and may introduce adverse effects. In this paper, we propose BALGRAD, which reweights gradients considering the learning status of each modality and projects the gradients to mitigate dominant modality bias without disrupting the balance between modalities. 

## 3 Method

In this section, we analyze the dominant modality bias and propose BALGRAD to mitigate such bias. In Section 3.1, we observe the behavior of VL models and theoretically demonstrate the factors influencing balanced loss convergence. In Section 3.2, based on these findings, we introduce BALGRAD, which reweights and projects gradients to ensure balanced learning across modalities. 

## 3.1 Analysis of Dominant Modality Bias

We introduce a controlled experiment to analyze the behavior of VL models biased by dominant modality. We denote the training dataset as $\mathcal { D } =$ $\{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ , where $x _ { i } ~ = ~ ( x _ { i } ^ { v } , x _ { i } ^ { l } )$ is a pair of data from the image and text modalities, respectively, and $y _ { i }$ represents the label. We extract features from the image and text encoders, passing them through their respective embedding layers, $h _ { v } ( \cdot )$ and $h _ { l } ( \cdot )$ . These embeddings are then fused via concatenation and passed through a classifier, $f _ { T } ( \cdot )$ , to yield the predicted probability $p \tau$ . Details on the architecture and training scheme are provided in the Appendix B. 

Analysis on Performance Gap. To analyze the impact of individual modalities on the performance of VL models, we mute one modality by inputting empty values at the data level, rendering it noninformative. This method is applied while testing on the UPMC Food-101, Hateful Memes, and MM-IMDb datasets. The experimental results in Figure 2 (a) show a significant performance drop when a specific modality is missing. In the case of UPMC Food-101, the image modality significantly influences the overall performance, while in Hateful Memes, the text modality plays a more crucial role. Conversely, the performance drop is relatively minor when the weak modality (text for UPMC Food-101 and image for Hateful Memes) is missing. In contrast, for MM-IMDb, the performance drop is similar when either modality is missing, indicating that the model is not biased towards a specific modality. 

Analysis on Training Dynamics. To observe the loss dynamics of each modality during the training phase, we add linear classifiers $f _ { v } ( \cdot )$ and $f _ { l } ( \cdot )$ on top of the image and text embedding layers, respectively. These classifiers output probabilities $p _ { i } ^ { v }$ and $p _ { i } ^ { l }$ , which are then used to predict the label $y _ { i }$ , and each target objective is represented as $\mathcal { L } _ { \mathcal { T } } ^ { v }$ and $\mathcal { L } _ { T } ^ { l }$ , respectively. We find that the loss of the dominant modality decreases rapidly, while the loss of the weak modality decreases relatively slowly, as shown in Figure 2 (b). For MM-IMDb specifically, the loss gap decreases as training iterations increase, demonstrating that the model is not biased toward any single modality. This indicates that, during training, one modality is overly exploited while the other modality is relatively underexplored, which is consistent with previous research (Wang et al., 2020; Huang et al., 2022; Peng et al., 2022). We conjecture that this phenomenon appears inherently task-dependent, with the VL model inclined to update based on the easy-to-learn modality that can quickly reduce the loss (Arpit et al., 2017; Nam et al., 2020). 

Theoretical Analysis of Gradient Influence. To theoretically analyze why VL models struggle to balance the utilization of both modalities, we examine the loss reduction in terms of gradient updates. The loss function for a target is defined as $\mathcal { L } ( \theta _ { v } , \theta _ { l } , \theta _ { \mathcal { T } } )$ , where $\theta _ { v }$ and $\theta _ { l }$ are the parameters of image and text embedding layers, respectively, and the $\theta \tau$ represents the parameters of the classifier $f _ { T } ( \cdot )$ . The objective is to find the optimal parameters $\Theta = \{ \theta _ { v } , \theta _ { l } , \theta _ { T } \}$ that minimize $\mathcal { L } ( \theta _ { v } , \theta _ { l } , \theta _ { \mathcal { T } } )$ . To analyze how each modality contributes to the overall loss reduction, we decompose the target task loss gradient with respect to the model parameters Θ into modality-specific components, denoted by $\mathcal { G } ^ { \tau } = \{ g _ { l } , g _ { v } , g _ { T } \}$ . These partial gradients capture the influence of linguistic, visual, and task-related parameters, respectively, under standard gradient-descent updates. Additionally, $\begin{array} { r } { g _ { \mathcal { T } } = \sum _ { i \in \{ v , l \} } \nabla _ { \theta _ { i } } \hat { y } \nabla _ { \hat { y } } p _ { \mathcal { T } } \nabla _ { p _ { \mathcal { T } } } \mathcal { L } = \sum _ { i \in \{ v , l \} } g _ { \mathcal { T } } ^ { i } } \end{array}$ denotes the gradient for parameters $\theta _ { T }$ of the linear classifier $f _ { T } ( \cdot )$ , where $g _ { T } ^ { i }$ denotes the gradient of each modality in $f _ { T } ( \cdot )$ . We theoretically analyze how the target objective is influenced by the varying magnitudes and directions of gradients for each modality. 

Proposition 1. (Gradient Effect on Change of Loss) Let the parameters $\theta _ { v } , \theta _ { l }$ , and $\theta _ { T }$ of a multimodal model be updated with gradients $g _ { v } , g _ { l }$ , and $g \tau$ using a sufficiently small step size $\lambda > 0 ,$ , resulting in updated parameters $\hat { \theta } _ { v } , \hat { \theta } _ { l }$ , and $\hat { \theta } _ { T }$ . Then the change in the loss function satisfies 

$$
\begin{array}{l} \triangle \mathcal {L} = - 2   \lambda \left(g _ {\mathcal {T}} ^ {v} \cdot g _ {\mathcal {T}} ^ {l}\right) \\ \quad -   \lambda \sum_ {i \in \{v, l, \mathcal {T} \}} \Big (g _ {i} \cdot g _ {i} + g _ {\mathcal {T}} ^ {i} \cdot g _ {\mathcal {T}} ^ {i} \Big) + O (\lambda^ {2}), \end{array}\tag{1}
$$

where the cross term $- 2 \lambda ( g _ { T } ^ { v } ) \cdot ( g _ { T } ^ { l } )$ captures the interaction between the visual and language gradients and the magnitudes and directions of each gradient $g _ { T } ^ { v }$ and $g _ { T } ^ { l }$ governs how much the overall loss is reduced. 

Proof. See Appendix A.1 

□ 

If the gradients for the two modalities $g _ { T } ^ { v }$ and $g _ { T } ^ { l }$ do not align well, meaning they have conflicting directions or have significantly different magnitudes, the loss reduction will not be balanced. Gradients with larger magnitudes substantially impact loss reduction, while gradients with directions that align more closely between modalities facilitate more effective joint learning. Consequently, the loss is likely to decrease more under the influence of the dominant modality, leading to an uneven contribution from each modality. 

## 3.2 BALGRAD

Based on the findings above, we propose BAL-GRAD to mitigate the dominant modality bias, which consists of two components: inter-modality gradient reweighting and inter-task gradient projection. Inter-modality gradient reweighting addresses the imbalance caused by different gradient magnitudes, ensuring more equal contributions from each modality. Inter-task gradient projection aligns the gradient directions of the modalities, facilitating more effective joint learning and preventing the dominant modality from disproportionately influencing loss reduction. The overall process of BALGRAD can be seen in Figure 3. 

## 3.2.1 Inter-modality Gradient Reweighting

Standard VL models lack the consideration to ensure that both modalities are updated equally, leading to the stronger modality dominating the training phase, as we observed in the previous section. Therefore, inspired by knowledge distillation (Hinton et al., 2014; Zhang et al., 2018; Phuong and Lampert, 2019), we aim to balance the gradients received from each modality by aligning the distributions of their predictions. To achieve this, we compute the mutual Kullback-Leibler (KL) divergence between the predictions $p _ { i } ^ { v }$ and $p _ { i } ^ { l }$ of the two modalities. This involves aligning the predictions of the image modality with those of the text modality and vice versa. The KL divergence from $p _ { i } ^ { l }$ to $p _ { i } ^ { v }$ is as follows: 

$$
\mathcal {L} _ {k l} ^ {l} = - \sum_ {i} p _ {i} ^ {l} \log \frac {p _ {i} ^ {v}}{p _ {i} ^ {l}}\tag{2}
$$

We also compute $\mathcal { L } _ { k l } ^ { v }$ in the same manner. We represent the gradients of $\mathcal { L } _ { k l } ^ { l }$ and $\mathcal { L } _ { k l } ^ { v }$ as $g _ { k l } ^ { l } =$ $\nabla \mathcal { L } _ { k l } ^ { l }$ and $\begin{array} { r } { g _ { k l } ^ { v } = \nabla \mathcal { L } _ { k l } ^ { v } . } \end{array}$ , respectively. In this way, each modality’s embedding layer learns to correctly predict the label and match the probability estimate of other modalities, thereby alleviating the severe imbalance. However, symmetrically aligning the distributions between the two modalities overlooks the differences in their convergence status, as observed in Section 3.1. This can cause the layers of the faster-converging modality to be hindered in their representation learning, leading to performance degradation. Therefore, we propose an inter-modality gradient reweighting method that adjusts the magnitude to which each modality receives the KL divergence gradient based on its contribution to the learning objective. We reweight the gradient of the KL divergence term for $p _ { i } ^ { l }$ to $p _ { i } ^ { v }$ and $p _ { i } ^ { v }$ to $p _ { i } ^ { l }$ using the following terms, respectively: 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/ceb22bb8-93ba-472a-b9f2-21044acb3d25/00b07fc541e0c79203c5ef50674cd35d9e688bb1407ff91e3ae9338000c362db.jpg)



Figure 3: (a) The overall training framework of our proposed BALGRAD. The final classifier $f _ { T } ( \cdot )$ is updated with the gradient $g _ { T } ^ { \bot }$ for cross entropy (CE) loss. The image and text embedding layers $h _ { v } ( \cdot ) , h _ { l } ( \cdot )$ are also updated with $g _ { T } ^ { \bot }$ along with the gradients of the CE loss for each modality $g _ { T } ^ { v } , g _ { T } ^ { l }$ , and the gradients of the KL divergence between the two modalities’ predictions $g _ { k l } ^ { v } , g _ { k l } ^ { l }$ . (b) Inter-modality gradient reweighting adjusts the magnitudes of $g _ { k l } ^ { v }$ and $g _ { k l } ^ { l }$ to obtain $g _ { k l }$ . If a conflict occurs, we project $g _ { T } ^ { \perp }$ on the orthogonal direction of $g _ { k l }$ by inter-task gradient projection.


$$
\mathcal {W} ^ {l} = \frac {\mathcal {L} _ {\mathcal {T}} ^ {l}}{\mathcal {L} _ {\mathcal {T}} ^ {v} + \mathcal {L} _ {\mathcal {T}} ^ {l}}, \mathcal {W} ^ {v} = \frac {\mathcal {L} _ {\mathcal {T}} ^ {v}}{\mathcal {L} _ {\mathcal {T}} ^ {v} + \mathcal {L} _ {\mathcal {T}} ^ {l}}\tag{3}
$$

In this configuration, if the target task loss for a modality is low (i.e., it has converged more), the gradient receives a lower weight. This ensures that the gradient of the weak modality is updated more toward matching the dominant modality’s prediction, thereby reducing the training gap. In contrast, the dominant modality receives less influence from the underperforming predictions, allowing it to effectively learn its representation. Additionally, to ensure that each modality is trained for the target task independently, we introduce an additional term that increases the reweighting factor as iteration t progresses. This ensures that the impact of mutual learning grows over time, allowing individual encoders to learn effectively in the initial stages and progressively encouraging balanced learning between modalities. The final reweighted gradient for the KL divergence is as follows: 

$$
g _ {k l} = (\gamma + \frac {\gamma}{1 + e ^ {- t}}) (\mathcal {W} ^ {l} g _ {k l} ^ {l} + \mathcal {W} ^ {v} g _ {k l} ^ {v})\tag{4}
$$

$\gamma$ is the initial weighting factor, and we set $\gamma =$ $1 / 2$ 

## 3.2.2 Inter-task Gradient Projection

Proposition 1 highlights that properly aligning gradients with different directions and magnitudes is crucial for effective joint learning. However, when gradients are not aligned and exhibit negative cosine similarity, known as conflicting gradients, the optimization process becomes suboptimal (Yu et al., 2020; Shi et al., 2023). Such conflicts can arise between the gradients of different tasks, potentially causing the dominant gradient to overwhelm the optimization process at the expense of the other task’s performance. 

For our case, as confirmed in Section 3.1, the target task, which is $\mathcal { L } _ { T }$ alone, fails to balance the modalities and fully explore the weak modality. Therefore, we introduce the balance between the predictions of each modality as an additional task. However, as mentioned earlier, naive joint training can cause conflict between the gradient of the target task and KL divergence (i.e., g and g ). 

Proposition 2. (Gradient Conflicts on Loss Reduction with KL Loss) Let $\mathcal { G } ^ { \tau } = \{ g _ { v } , g _ { l } , g _ { \tau } \}$ and $\mathcal { G } ^ { k l } = \{ g _ { v } ^ { k l } , g _ { l } ^ { k l } , 0 \}$ be the gradients from a target loss $\scriptstyle { \mathcal { L } } _ { \tau }$ and a KL loss $\mathcal { L } _ { k l }$ , respectively, with parameters $\theta = [ \theta _ { v } , \theta _ { l } , \theta _ { \tau } ] ^ { \top }$ . Assume the parameters are updated by gradient descent with a small step size $\lambda > 0 \colon \theta _ { v } ^ { \prime } = \theta _ { v } - \lambda \bigl ( g _ { v } + g _ { v } ^ { k l } \bigr ) , \theta _ { l } ^ { \prime } =$ $\theta _ { l } - \lambda \bigl ( g _ { l } + g _ { l } ^ { k l } \bigr ) , \theta _ { \tau } ^ { \prime } = \theta _ { \tau } - \lambda g _ { \tau }$ . Then, for the combined loss $\mathcal { L } = \mathcal { L } _ { \tau } + \mathcal { L } _ { k l }$ , the change in the loss is 

$$
\begin{array}{l} \Delta \mathcal {L} = \mathcal {L} \big (\theta^ {\prime} \big) - \mathcal {L} \big (\theta \big) \\ \qquad = - \lambda \Big (\| \mathcal {G} ^ {\tau} \| ^ {2} + \| \mathcal {G} ^ {k l} \| ^ {2} + 2 \left(\mathcal {G} ^ {\tau}\right) ^ {\top} \mathcal {G} ^ {k l} \Big) \\ \qquad + O (\lambda^ {2}). \end{array}\tag{5}
$$

In particular, if $( \mathcal { G } ^ { \tau } ) ^ { \top } \mathcal { G } ^ { k l } < 0$ , the gradients from the target and KL losses conflict, reducing the effective loss reduction. 

## Proof. See Appendix A.2

□ 

Building upon Proposition 2, we aim to ensure that the gradient of the target task does not disrupt the balance between modalities. Specifically, we propose inter-task gradient projection, which projects $g \tau$ onto $g _ { k l }$ in a non-conflicting manner. First, we consider the relationship between the two gradients to determine if they conflict and compute the cosine similarity between the two gradients. If $g _ { \mathcal { T } } \cdot g _ { k l } \ge 0 .$ , we assume that $g \tau$ is being updated in a direction that aligns with modality balance, and we use the original $g \tau$ for updating the model. Conversely, if $g _ { T } \cdot g _ { k l } < 0$ , indicating a potential disruption to the balance between modalities, we project $g \tau$ in a direction orthogonal to $g _ { k l }$ . This process can be represented as follows: 

$$
g _ {\mathcal {T}} ^ {\perp} = \left\{ \begin{array}{l l} g _ {\mathcal {T}} - \left(\frac {g _ {\mathcal {T}} \cdot g _ {k l}}{\| g _ {k l} \| ^ {2}}\right) g _ {k l}, & \text { if } g _ {\mathcal {T}} \cdot g _ {k l} <   0 \\ g _ {\mathcal {T}}, & \text { otherwise } \end{array} \right.\tag{6}
$$

This projection ensures that $g _ { T } ^ { \perp }$ is adjusted to maintain the balance between the modalities while preventing conflicts with $g _ { k l }$ . In a nutshell, the proposed BALGRAD allows for extensively learning different modalities and tasks, effectively optimizing the target task while maintaining the balance between the modalities. 

## 4 Experiments

## 4.1 Experimental Setup

We conduct experiments on three vision-language datasets: UPMC Food-101 (Wang et al., 2015), 

Hateful Memes (Kiela et al., 2020), and MM-IMDb (Arevalo et al., 2017). For image and text encoding, we utilize ViT (Dosovitskiy et al., 2021) and BERT (Devlin et al., 2019), respectively, employing a late concatenation architecture for final predictions. To minimize extensive fine-tuning, we adopt linear probing, freezing all encoder parameters and training only the embedding and classifier layers. Further implementation details are provided in Appendix B. 

To assess the robustness of the VL model against dominant modality bias, we introduced two impaired conditions: missing and noisy. For the missing modality, empty strings were used for text and zero pixels for images (Lee et al., 2023). In the noisy condition, 30% salt and pepper noise is added to images (Lim et al., 2023), and 15% of text tokens were randomly deleted (Manolache et al., 2021; Yuan et al., 2023). All experiments were conducted with the model trained on unimpaired full modality data, with impairments applied to the entire data of a specific modality during testing. Further implementation details are provided in Appendix C. 

## 4.2 Experimental Results

We train with full modality data and evaluate the performance of the VL model under conditions where one modality is entirely impaired across three datasets, as shown in Table 1. “Full” refers to the scenario where no modalities are impaired during testing. For the impaired cases (missing and noisy), each modality is impaired according to the specified method. “Avg.” denotes the average performance when each modality is impaired individually, while $^ { * * } \Delta _ { G a p } \} ^ { * }$ represents the performance difference between the image-impaired and textimpaired conditions. A smaller $\Delta _ { G a p }$ indicates a more balanced model that does not overly rely on a single modality, thereby exhibiting less dominant modality bias. 

For the UPMC Food-101 dataset, BALGRAD demonstrates the highest performance across all conditions—full, missing image, and missing text. Notably, it improves the performance on the weak modality, text, by 12.5%p compared to the baseline. Additionally, it achieves the highest average performance and exhibits the smallest gap, effectively mitigating bias despite the dominant influence of the image modality. In the noisy condition, our method shows robustness comparable to AGM (Li et al., 2023) and achieves the highest Avg. 

BALGRAD exhibits the highest performance in conditions where the dominant text modality is missing, as well as in the full modality, $A \nu g .$ and $\Delta _ { G a p }$ for the Hateful Memes dataset. OGM GE (Peng et al., 2022) and AGM perform better in the image missing condition than in the full modality condition, indicating a heavy reliance on the text modality, with performance increases of 0.13%p and 1.56%p, respectively. In other words, adding the image modality results in a decrease in performance compared to using text alone, exhibiting negative transfer (Wang et al., 2019). In the noisy condition, BALGRAD demonstrates the highest Avg. performance and the smallest $\Delta _ { G a p } ,$ showcasing that BALGRAD sufficiently explores the image modality. 

<table><tr><td rowspan="2" colspan="2">Modality</td><td colspan="5">UPMC Food-101</td><td colspan="5">Hateful Memes</td><td colspan="5">MM-IMDb</td></tr><tr><td>Baseline</td><td>MSLR</td><td>OGM-GE</td><td>AGM</td><td>BALGRAD</td><td>Baseline</td><td>MSLR</td><td>OGM-GE</td><td>AGM</td><td>BALGRAD</td><td>Baseline</td><td>MSLR</td><td>OGM-GE</td><td>AGM</td><td>BALGRAD</td></tr><tr><td colspan="2">Full</td><td>76.01</td><td>78.43</td><td>77.42</td><td><eq>\underline{78.93}</eq></td><td>80.32</td><td>65.10</td><td>65.58</td><td><eq>\underline{66.70}</eq></td><td>64.69</td><td>67.35</td><td>44.09</td><td>44.09</td><td>42.22</td><td>43.93</td><td>43.19</td></tr><tr><td rowspan="4">Missing</td><td>Image</td><td>12.99</td><td>20.52</td><td>13.86</td><td><eq>\underline{22.60}</eq></td><td>25.49</td><td>64.34</td><td>66.04</td><td><eq>66.83^*</eq></td><td><eq>\underline{66.25^*}</eq></td><td>65.86</td><td>18.85</td><td><eq>\underline{19.26}</eq></td><td>24.48</td><td>17.57</td><td>18.81</td></tr><tr><td>Text</td><td><eq>\underline{63.52}</eq></td><td>63.00</td><td>61.45</td><td>63.13</td><td>65.03</td><td>55.60</td><td>55.66</td><td><eq>\underline{57.20}</eq></td><td>56.20</td><td>57.58</td><td>18.40</td><td>14.67</td><td>12.31</td><td>15.46</td><td>17.47</td></tr><tr><td>Avg.<eq>\uparrow</eq></td><td>38.26</td><td>41.76</td><td>37.66</td><td><eq>\underline{42.87}</eq></td><td>45.26</td><td>59.97</td><td>60.85</td><td><eq>\underline{62.02}</eq></td><td>61.23</td><td><eq>\underline{61.72}</eq></td><td>18.63</td><td>16.97</td><td><eq>\underline{18.40}</eq></td><td>16.52</td><td>18.14</td></tr><tr><td><eq>\Delta_{Gap}\downarrow</eq></td><td>50.53</td><td>42.48</td><td>47.59</td><td><eq>\underline{40.53}</eq></td><td>39.54</td><td><eq>\underline{8.74}</eq></td><td>10.38</td><td>9.63</td><td>10.05</td><td>8.28</td><td>0.45</td><td>4.59</td><td>12.17</td><td>2.11</td><td><eq>\underline{1.34}</eq></td></tr><tr><td rowspan="4">Noisy</td><td>Image</td><td>41.92</td><td>52.92</td><td>46.50</td><td><eq>\underline{56.57}</eq></td><td><eq>\underline{55.58}</eq></td><td>63.64</td><td><eq>\underline{64.21}</eq></td><td>63.72</td><td>61.85</td><td>65.78</td><td>30.89</td><td>33.86</td><td>35.31</td><td><eq>\underline{35.73}</eq></td><td>37.76</td></tr><tr><td>Text</td><td>67.28</td><td><eq>\underline{77.71}</eq></td><td>75.94</td><td>77.43</td><td>78.54</td><td>65.09</td><td>63.66</td><td><eq>67.16^*</eq></td><td>63.68</td><td><eq>\underline{65.60}</eq></td><td>38.09</td><td>43.00</td><td>40.33</td><td><eq>\underline{42.66}</eq></td><td>41.80</td></tr><tr><td>Avg.<eq>\uparrow</eq></td><td>54.60</td><td>65.32</td><td>61.22</td><td><eq>\underline{67.00}</eq></td><td>67.06</td><td>64.37</td><td>63.94</td><td><eq>\underline{65.44}</eq></td><td>62.77</td><td>65.69</td><td>34.49</td><td>38.43</td><td>37.82</td><td><eq>\underline{39.20}</eq></td><td>39.78</td></tr><tr><td><eq>\Delta_{Gap}\downarrow</eq></td><td>25.36</td><td>24.79</td><td>29.44</td><td><eq>\underline{20.86}</eq></td><td><eq>\underline{22.96}</eq></td><td>1.45</td><td><eq>\underline{0.55}</eq></td><td>3.44</td><td>1.83</td><td>0.18</td><td>7.20</td><td>9.14</td><td><eq>\underline{5.02}</eq></td><td>6.93</td><td>4.04</td></tr></table>


Table 1: The experimental result to validate the effectiveness of BALGRAD on the UPMC Food-101, Hateful Memes, and MM-IMDb datasets. The best result in each test dataset is boldfaced, and the second best is presented with underlining. $^ { \bullet \bullet } A \nu g . ^ { \prime \prime }$ represents the average performance under conditions where one of the modalities is impaired (missing or noisy), while $^ { * * } \Delta _ { G a p } \mathrm { , }$ ” indicates the performance difference. The value that is displayed in gray* represents a negative transfer. The unit for $^ { \ast } \Delta _ { G a p } \} ^ { \ast }$ is $\% \mathbf { p } .$ , and the unit for all other values is %.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/ceb22bb8-93ba-472a-b9f2-21044acb3d25/4bca291aa5283c12fd4150cd3e5fb12b4280f4f81ca1d0d027de52bd25b15acc.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/ceb22bb8-93ba-472a-b9f2-21044acb3d25/831ec7fc886df774d28e610d979b8b73ae131172a48e7cb11c132e9b81e46ac4.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/ceb22bb8-93ba-472a-b9f2-21044acb3d25/105ba7333c81fada14aa3a2385ac95db9141d114caafdc314b213b30d81ced78.jpg)



Figure 4: Evaluation on robustness to different missing ratio r of $\mathbf { B A L G R A D }$ and existing methods on UPMC Food-101, Hateful Memes, and MM-IMDb datasets.


Furthermore, BALGRAD maintains the balance between the modalities even despite the absence of any dominant modality. For the MM-IMDb dataset, our proposed method shows slightly lower performance compared to the baseline but exhibits the second-smallest $\Delta _ { G a p } ,$ , indicating balanced results without a dominant modality. Although OGM-GE demonstrates high performance, it exhibits a significant imbalance between modalities, as evidenced by the considerably higher gap, which is 10.83%p more than our method. BALGRAD achieves the highest average performance and the lowest gap in the noisy condition, showcasing that our proposed method effectively explores both modalities without being biased towards one. 

Additionally, to investigate the robustness under varying degrees of impairment, we mute a specific modality according to the missing ratio $^ { r , }$ and the results are shown in Figure 4. For each dataset, we randomly drop a certain percentage $r \%$ of the data from each modality and measure the resulting performance $\Delta _ { G a p }$ . We set missing ratios $r \in \{ 0 . 2 , 0 . 4 , 0 . 6 , 0 . 8 \}$ Experimental results indicate that BALGRAD consistently exhibits a lower gap compared to existing methods across varying missing ratios, demonstrating robustness to impaired modalities. While BALGRAD exhibits a slightly larger gap compared to the baseline, it is noteworthy that BALGRAD significantly reduces the gap for datasets with dominant modality bias. Additionally, it introduces a small gap for datasets where dominant modality bias is not present. 

Additional experimental results on various fu sion mechanisms, backbone models, and datasets are provided in Appendix C. The results demonstrate that BALGRAD consistently delivers robust performance across different biases, modality types, datasets, and perturbed conditions, underscoring its effectiveness in synergistically integrating modalities to prevent negative transfer and ensure reliable, real-world multimodal learning. 

<table><tr><td rowspan="2" colspan="2">Modality</td><td colspan="4">UPMC Food-101</td><td colspan="4">Hateful Memes</td><td colspan="4">MM-IMDb</td></tr><tr><td>Baseline</td><td colspan="2">w/ Gradient w/ Gradient reweighting projection</td><td>BALGRAD</td><td>Baseline</td><td colspan="2">w/ Gradient w/ Gradient reweighting projection</td><td>BALGRAD</td><td>Baseline</td><td colspan="2">w/ Gradient w/ Gradient reweighting projection</td><td>BALGRAD</td></tr><tr><td colspan="2">Full</td><td>76.01</td><td>78.17</td><td>76.20</td><td>80.32</td><td>65.10</td><td>65.80</td><td>66.30</td><td>67.35</td><td>44.09</td><td>44.30</td><td>42.30</td><td>43.19</td></tr><tr><td rowspan="4">Missing</td><td>Image</td><td>12.99</td><td>22.30</td><td>19.82</td><td>25.49</td><td>64.34</td><td>66.37*</td><td>65.40</td><td>65.86</td><td>18.85</td><td>21.48</td><td>18.47</td><td>18.81</td></tr><tr><td>Text</td><td>63.52</td><td>64.10</td><td>63.76</td><td>65.03</td><td>55.60</td><td>57.03</td><td>56.20</td><td>57.48</td><td>18.40</td><td>17.20</td><td>18.80</td><td>17.47</td></tr><tr><td>Avg.<eq>\uparrow</eq></td><td>38.26</td><td>43.20</td><td>41.79</td><td>45.26</td><td>59.97</td><td>61.70</td><td>60.80</td><td>61.67</td><td>18.63</td><td>19.34</td><td>18.64</td><td>18.14</td></tr><tr><td>Gap</td><td>50.53</td><td>41.80</td><td>43.94</td><td>39.54</td><td>8.74</td><td>9.34</td><td>9.20</td><td>8.38</td><td>0.45</td><td>4.28</td><td>0.33</td><td>1.34</td></tr></table>


Table 2: Ablation study results compares performance with and without inter-modality gradient reweighting and inter-task gradient projection to evaluate their impact on modality balance and transfer effects on UPMC Food-101, Hateful Memes, and MM-IMDb datasets. The best results are highlighted in bold, the second-best in italics, and values shown in $\mathrm { g r a y ^ { * } }$ indicate negative transfer. $^ { * * } \Delta _ { G a p } \} ^ { * }$ is reported in %p, while all other values are in %.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/ceb22bb8-93ba-472a-b9f2-21044acb3d25/a3e7da952f591dfa289dd72e1bcf7639168adea9e8d2fb8aee10cf17ab6cd558.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/ceb22bb8-93ba-472a-b9f2-21044acb3d25/e677f258a4115ea46d3b2a75704d37c0565e9872dcccec99d78077e864028619.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/ceb22bb8-93ba-472a-b9f2-21044acb3d25/488fe62772dc2c06338365b69b29160fc084ec4746e4c708da2cc389087aa13d.jpg)



Figure 5: Bar plots comparing the performance of existing methods and BALGRAD using BLIP. Each bar represents $\Delta _ { G a p } ( \% )$ , defined as the performance difference between missing image and missing text conditions.


## 4.3 Ablation and Analysis

Analysis of Each Component. We conduct ablation experiments to assess the impact of gradient reweighting and projection as shown in Table 2. While gradient reweighting shares a common approach with existing methods (Peng et al., 2022; Li et al., 2023), helps mitigate modality imbalance, it induces negative transfer in the Hateful Memes dataset and leaves the MM-IMDb dataset overly reliant on text. In contrast, incorporating gradient projection eliminates negative transfer and balances modality use. By aligning the gradient of the target loss with the KL loss term, we reduce reliance on any single modality, effectively preventing negative transfer. These points clarify how our approach differs from existing work and address the gaps in empirical validation and mitigation of negative effects. 

Evaluation on Text Decoder-based Vision-Language Model. To examine BALGRAD’s effectiveness in text decoder-based architectures, we conduct additional experiments using BLIP (Li et al., 2022), which generates textual outputs from visual inputs via a text decoder. This setup differs from encoder-only VL models and aligns with autoregressive language modeling approaches. As shown in Figure 5, BALGRAD achieves the lowest $\Delta _ { G a p }$ across all datasets, indicating its ability to balance modality contributions in decoder-based VL models. These results highlight BALGRAD’s potential for extension to decoder-only LLMs, as it effectively mitigates dominant modality bias across different VL architectures. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/ceb22bb8-93ba-472a-b9f2-21044acb3d25/3d16cf6a570461cc26645b65b40f4da909e19f89347ebf051c27193750746870.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/ceb22bb8-93ba-472a-b9f2-21044acb3d25/12870afeb7167009be07e599835d36c741829d502a4d9e3fc583f4099f717ffd.jpg)



Figure 6: Training iteration loss curves for image and text modalities on the UPMC Food-101 and Hateful Memes datasets, comparing the effects of the existence of inter-modality gradient reweighting.


Ablation on Inter-modality Gradient Reweighting. To validate the efficacy of inter-modality gradient reweighting, we track the training loss dynamics for each modality on datasets with dominant modality bias (UPMC Food-101 and Hateful Memes), as shown in Figure 6. Without reweighting, weights are fixed at $\mathcal { W } ^ { v } = 1 / 2$ and $\mathcal { W } ^ { l } = 1 / 2 .$ equally distilling information between modalities. Experimental results show that reweighting leads to faster and more stable convergence of loss for each modality. This supports Proposition 1 in Section 3.1, indicating that gradient reweighting optimizes the exploration of individual modalities while maintaining balance in the VL model. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/ceb22bb8-93ba-472a-b9f2-21044acb3d25/72d928793263fbeaeb7f8619017165ff9df174eebea09c1c6126779d00892213.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/ceb22bb8-93ba-472a-b9f2-21044acb3d25/1eb8d6af14ff527e36359b02bc5434be33905db93464d29b87e5490760e94c56.jpg)



Figure 7: Histogram visualization of the frequency of gradient conflicts between image and text gradients during training iterations on the UPMC Food-101 and Hateful Memes datasets. $\mu _ { w / o }$ and $\mu _ { w / }$ represent the average cosine similarity values w/o and w/ projection, respectively.


Analysis on Inter-task Gradient Projection.To assess the impact of inter-task gradient projection, we visualize the cosine similarity between the gradients of KL divergence (g<sub>kl</sub>) and the target task (g ) throughout the entire training process using histograms, as shown in Figure 7. Without gradient projection, negative similarity between gradients is prevalent throughout training, resulting in imbalanced updates to the target task. Conversely, BALGRAD, incorporating inter-task gradient projection, shows a positive mean cosine similarity between gradients, indicating fewer conflicts during training. This suggests that the gradients for the target task are more balanced between the two modalities, leading to more balanced convergence. This reduction in conflicts narrows the performance gap between image and text modalities, mitigating over-reliance on any specific modality, aligning with our analysis in Section 3.1. 

To further quantify this effect, we conduct an ablation study measuring the frequency of conflicting gradients with and without projection across three datasets, as shown in Table 3. The fraction indicates the percentage of gradient conflicts that occur between the gradients of KL divergence $( g _ { k l } )$ and the target task $( g \tau )$ throughout the entire training process. The results demonstrate that there is a high incidence of conflicting gradients across all datasets without projection. In contrast, the use of projection significantly reduces gradient conflicts, especially in datasets with dominant modality bias, such as UPMC Food-101 and Hateful Memes. 

<table><tr><td rowspan="2"></td><td colspan="2">UPMCFood-101</td><td colspan="2">Hateful Memes</td><td colspan="2">MM-IMDb</td></tr><tr><td>Fraction↓</td><td><eq>\Delta_{Gap}</eq>↓</td><td>Fraction↓</td><td><eq>\Delta_{Gap}</eq>↓</td><td>Fraction↓</td><td><eq>\Delta_{Gap}</eq>↓</td></tr><tr><td>w/o Projection</td><td>0.66</td><td>43.27</td><td>0.78</td><td>10.21</td><td>0.28</td><td>4.21</td></tr><tr><td>w/ Projection</td><td>0.36</td><td>39.54</td><td>0.32</td><td>8.28</td><td>0.26</td><td>4.04</td></tr></table>


Table 3: Ablative results show the fraction of conflicting gradients and $\Delta _ { G a p }$ on the UPMC Food-101 and Hateful Memes datasets, comparing scenarios without inter-task gradient projection (“w/o Projection”) and with standard BALGRAD (“w/ Projection”).


## 5 Conclusion

In this paper, we addressed the challenge of dominant modality bias, where a VL model disproportionately relies on one modality, undermining the contributions of others. Our analysis shows that unaligned gradients and differences in gradient magnitudes hinder balanced loss convergence. Based on these findings, BALGRAD mitigates this bias by incorporating inter-modality gradient reweighting, which adjusts the KL divergence gradient based on each modality’s contribution, and inter-task gradient projection to align task directions nonconflictingly. Experiments on UPMC Food-101, Hateful Memes, and MM-IMDb datasets demonstrate that BALGRAD effectively reduces dominant modality bias, enhances model robustness, and improves accuracy. These results highlight the potential for more stable and balanced training in VL models, paving the way for future advancements. 

## Limitation

While BALGRAD has shown efficacy in mitigating dominant modality bias in VL models, extending this approach to multimodal models with more than two modalities presents additional challenges. When dealing with three or more modalities, the training cost rapidly increases due to the need to consider the relationships between the gradients of each pair of modalities. This increased complexity in gradient management makes the balancing process more computationally intensive and difficult to maintain effectively. Thus, while BALGRAD is effective in bi-modal settings, its application in multimodal scenarios requires further refinement to manage the higher computational demands and ensure balanced performance across all modalities. 

## Acknowledgment

This research was supported by the Institute for Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. 2021-0-01341, Artificial Intelligence Graduate School Program, Chung-Ang University). This research was also supported by the MSIT (Ministry of Science and ICT), Korea , under the Graduate School of Metaverse Convergence support program (IITP-2025-RS-2024- 00418847) supervised by the IITP (Institute for Information & Communications Technology Planning & Evaluation). 

## References



Piush Aggarwal, Jawar Mehrabanian, Weigang Huang, Özge Alacam, and Torsten Zesch. 2024. Text or image? what is more important in cross-domain generalization capabilities of hate meme detection models? In Findings of the Association for Computational Linguistics: EACL 2024, pages 104–117. 





John Arevalo, Thamar Solorio, Manuel Montes-y Gómez, and Fabio A González. 2017. Gated multimodal units for information fusion. In Proceedings ofthe International Conference on Learning Representations: Workshop Track. 





Devansh Arpit, Stanisław Jastrz˛ebski, Nicolas Ballas, David Krueger, Emmanuel Bengio, Maxinder S Kanwal, Tegan Maharaj, Asja Fischer, Aaron Courville, Yoshua Bengio, et al. 2017. A closer look at memorization in deep networks. In Proceedings ofthe International Conference on Machine Learning, pages 233–242. 





Veronika Cheplygina, Marleen De Bruijne, and Josien PW Pluim. 2019. Not-so-supervised: a survey of semi-supervised, multi-instance, and transfer learning in medical image analysis. Medical image analysis, 54:280–296. 





Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, pages 4171–4186. 





Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. 2021. An image is worth 16x16 words: Transformers for image recognition at scale. In Proceedings of the International Conference on Learning Representations. 





Muskan Garg, Seema Wazarkar, Muskaan Singh, and Ondˇrej Bojar. 2022. Multimodality for nlp-centered applications: Resources, advances and frontiers. In Proceedings ofthe Thirteenth Language Resources and Evaluation Conference, pages 6837–6847. 





Yangyang Guo, Liqiang Nie, Harry Cheng, Zhiyong Cheng, Mohan Kankanhalli, and Alberto Del Bimbo. 2023. On modality bias recognition and reduction. ACM Transactions on Multimedia Computing, Communications and Applications, 19(3):1–22. 





Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. Deep residual learning for image recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 770–778. 





Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. 2014. Distilling the knowledge in a neural network. In Proceedings of the NeurIPS 2014 Deep Learning and Representation Learning Workshop. 





Xiaowei Hu, Zhe Gan, Jianfeng Wang, Zhengyuan Yang, Zicheng Liu, Yumao Lu, and Lijuan Wang. 2022. Scaling up vision-language pre-training for image captioning. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17980–17989. 





Yu Huang, Junyang Lin, Chang Zhou, Hongxia Yang, and Longbo Huang. 2022. Modality competition: What makes joint training of multi-modal network fail in deep learning? (provably). In Proceedings of the International Conference on Machine Learning, pages 9226–9259. 





Mahmoud Khademi, Ziyi Yang, Felipe Frujeri, and Chenguang Zhu. 2023. Mm-reasoner: A multimodal knowledge-aware framework for knowledgebased visual question answering. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2023, pages 6571–6581. 





Douwe Kiela, Suvrat Bhooshan, Hamed Firooz, and Davide Testuggine. 2019. Supervised multimodal bitransformers for classifying images and text. arXiv preprint arXiv:1909.02950. 





Douwe Kiela, Hamed Firooz, Aravind Mohan, Vedanuj Goswami, Amanpreet Singh, Pratik Ringshia, and Davide Testuggine. 2020. The hateful memes challenge: Detecting hate speech in multimodal memes. In Advances in Neural Information Processing Systems, pages 2611–2624. 





Julia Kruk, Jonah Lubin, Karan Sikka, Xiao Lin, Dan Jurafsky, and Ajay Divakaran. 2019. Integrating text and image: Determining multimodal document intent in instagram posts. arXiv preprint arXiv:1904.09073. 





Gokul Karthik Kumar and Karthik Nandakumar. 2022. Hate-clipper: Multimodal hateful meme classification based on cross-modal interaction of clip features. arXiv preprint arXiv:2210.05916. 





Yi-Lun Lee, Yi-Hsuan Tsai, Wei-Chen Chiu, and Chen-Yu Lee. 2023. Multimodal prompting with missing modalities for visual recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14943–14952. 





Hong Li, Xingyu Li, Pengbo Hu, Yinuo Lei, Chunxiao Li, and Yi Zhou. 2023. Boosting multi-modal model performance with adaptive gradient modulation. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 22214–22224. 





Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. 2022. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In Proceedings of the International Conference on Machine Learning, pages 12888–12900. PMLR. 





Jongin Lim, Youngdong Kim, Byungjai Kim, Chanho Ahn, Jinwoo Shin, Eunho Yang, and Seungju Han. 2023. Biasadv: Bias-adversarial augmentation for model debiasing. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3832–3841. 





Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2024. Visual instruction tuning. Advances in Neural Information Processing Systems, 36. 





Zhenghao Liu, Chenyan Xiong, Yuanhuiyi Lv, Zhiyuan Liu, and Ge Yu. 2023. Universal vision-language dense retrieval: Learning a unified representation space for multi-modal retrieval. In Proceedings of the International Conference on Learning Representations. 





Andrei Manolache, Florin Brad, and Elena Burceanu. 2021. Date: Detecting anomalies in text via selfsupervision of transformers. In Proceedings of the 2021 Conference ofthe North American Chapter of the Associationfor Computational Linguistics: Hu man Language Technologies, pages 267–277. 





Stefano Menini, Alessio Palmero Aprosio, and Sara Tonelli. 2020. A multimodal dataset of images and text to study abusive language. In Proceedings of the Seventh Italian Conference on Computational Linguistics, CLiC-it 2020, volume 2769. CEUR-WS. org. 





Shreyash Mishra, S Suryavardan, Parth Patwa, Megha Chakraborty, Anku Rani, Aishwarya Reganti, Aman Chadha, Amitava Das, Amit Sheth, Manoj Chinnakotla, et al. 2023. Memotion 3: Dataset on sentiment and emotion analysis of codemixed hindienglish memes. arXiv preprint arXiv:2303.09892. 





Junhyun Nam, Hyuntak Cha, Sungsoo Ahn, Jaeho Lee, and Jinwoo Shin. 2020. Learning from failure: Debiasing classifier from biased classifier. In Advances in Neural Information Processing Systems, pages 20673–20684. 





Xiaokang Peng, Yake Wei, Andong Deng, Dong Wang, and Di Hu. 2022. Balanced multimodal learning via 





on-the-fly gradient modulation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8238–8247. 





Mary Phuong and Christoph Lampert. 2019. Towards understanding knowledge distillation. In Proceedings of the International Conference on Machine Learning, pages 5142–5151. 





Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In Proceedings of the International Conference on Machine Learning, pages 8748–8763. PMLR. 





Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. 2019. Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108. 





Guangyuan Shi, Qimai Li, Wenlong Zhang, Jiaxin Chen, and Xiao-Ming Wu. 2023. Recon: Reducing conflicting gradients from the root for multi-task learning. In Proceedings of the International Conference on Learning Representations. 





Paul Voigt and Axel Von dem Bussche. 2017. The eu general data protection regulation (gdpr). A Practical Guide, 1st Ed., Cham: Springer International Publishing, 10(3152676):10–5555. 





Weiyao Wang, Du Tran, and Matt Feiszli. 2020. What makes training multi-modal classification networks hard? In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12695–12705. 





Xin Wang, Devinder Kumar, Nicolas Thome, Matthieu Cord, and Frederic Precioso. 2015. Recipe recognition with large multimodal food dataset. In Proceedings ofthe 2015 IEEE International Conference on Multimedia & Expo Workshops, pages 1–6. 





Zirui Wang, Zihang Dai, Barnabás Póczos, and Jaime Carbonell. 2019. Characterizing and avoiding negative transfer. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11293–11302. 





Peter Welinder, Steve Branson, Takeshi Mita, Catherine Wah, Florian Schroff, Serge Belongie, and Pietro Perona. 2010. Caltech-ucsd birds 200. Technical Report CNS-TR-2010-001, California Institute of Technology. 





Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, et al. 2020. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38–45. 





Sangmin Woo, Sumin Lee, Yeonju Park, Muhammad Adi Nugroho, and Changick Kim. 2023. Towards good practices for missing modality robust action recognition. In Proceedings ofthe AAAI Conference on Artificial Intelligence, pages 2776–2784. 





Nan Wu, Stanislaw Jastrzebski, Kyunghyun Cho, and Krzysztof J Geras. 2022. Characterizing and overcoming the greedy nature of learning in multi-modal deep neural networks. In Proceedings of the International Conference on Machine Learning, pages 24043–24055. 





Zequn Yang, Yake Wei, Ce Liang, and Di Hu. 2024. Quantifying and enhancing multi-modal robustness with modality preference. In Proceedings ofthe In ternational Conference on Learning Representations. 





Yiqun Yao and Rada Mihalcea. 2022. Modality-specific learning rates for effective multimodal additive latefusion. In Findings ofthe Associationfor Computa tional Linguistics: ACL 2022, pages 1824–1834. 





Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. 2020. Gradient surgery for multi-task learning. In Advances in Neural Information Processing Systems, pages 5824–5836. 





Hongyi Yuan, Zheng Yuan, Chuanqi Tan, Fei Huang, and Songfang Huang. 2023. Hype: Better pre-trained language model fine-tuning with hidden representation perturbation. In Proceedings ofthe 61st Annual Meeting of the Association for Computational Linguistics, pages 3246–3264. 



Ying Zhang, Tao Xiang, Timothy M Hospedales, and Huchuan Lu. 2018. Deep mutual learning. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4320–4328. 

## A Appendix of Propositions

## A.1 Proof of Proposition 1

Proposition 1. (Gradient Effect on Change of Loss) Let the parameters $\theta _ { v } , \theta _ { l }$ , and $\theta _ { T }$ of a multimodal model be updated with gradients $g _ { v } , g _ { l } .$ , and $g \tau$ using a sufficiently small step size $\lambda > 0$ , resulting in updated parameters $\hat { \theta } _ { v } , \hat { \theta } _ { l }$ , and $\hat { \theta } _ { T }$ . Then the change in the loss function satisfies 

$$
\begin{array}{l} \triangle \mathcal {L} = - 2   \lambda \left(g _ {\mathcal {T}} ^ {v} \cdot g _ {\mathcal {T}} ^ {l}\right) \\ \quad -   \lambda \sum_ {i \in \{v, l, \mathcal {T} \}} \Big (g _ {i} \cdot g _ {i} + g _ {\mathcal {T}} ^ {i} \cdot g _ {\mathcal {T}} ^ {i} \Big) + O (\lambda^ {2}), \end{array}\tag{7}
$$

where the cross term $- 2 \lambda ( g _ { T } ^ { v } ) \cdot ( g _ { T } ^ { l } )$ captures the interaction between the visual and language gradients and the magnitudes and directions of each gradient $g _ { T } ^ { v }$ and $g _ { T } ^ { l }$ governs how much the overall loss is reduced. 

## Proof of Proposition 1.

Proof. Let the $\theta _ { v } , \theta _ { l } , \theta _ { T }$ be updated in the direction of negative gradients $g _ { v } , g _ { l } , g _ { T }$ with step size $\lambda >$ 0. Then the updated $\hat { \theta } _ { v } , \hat { \theta } _ { l } , \hat { \theta } _ { T }$ are $\theta _ { v } - \lambda g _ { v } , \theta _ { l }$ $\lambda g _ { l } , \theta _ { T } - \lambda g _ { T }$ . In that case, the change in the loss function with updated parameters is 

$$
\triangle \mathcal {L} = \mathcal {L} (\hat {\theta_ {v}}, \hat {\theta_ {l}}, \hat {\theta_ {\mathcal {T}}}) - \mathcal {L} (\theta_ {v}, \theta_ {l}, \theta_ {\mathcal {T}})
$$

By the first-order taylor expansion with a point $( \theta _ { v } , \theta _ { l } , \theta _ { T } )$ 

$$
\begin{array}{r l} & {\mathcal {L} (\hat {\theta_ {v}}, \hat {\theta_ {l}}, \hat {\theta_ {\mathcal {T}}}) - \mathcal {L} (\theta_ {v}, \theta_ {l}, \theta_ {\mathcal {T}})} \\ & {= \mathcal {L} (\theta_ {v} - \lambda g _ {v}, \theta_ {l} - \lambda g _ {l}, \theta_ {\mathcal {T}} - \lambda g _ {\mathcal {T}})} \\ & {- \mathcal {L} (\theta_ {v}, \theta_ {l}, \theta_ {\mathcal {T}})} \\ & {= \mathcal {L} (\theta_ {v}, \theta_ {l}, \theta_ {\mathcal {T}})} \\ & {+ (\theta_ {v} - \lambda g _ {v} - \theta_ {v}) ^ {T} g _ {v} (\theta_ {l} - \lambda g _ {l} - \theta_ {l}) ^ {T} g _ {l}} \\ & {+ (\theta_ {\mathcal {T}} - \lambda g _ {\mathcal {T}} - \theta_ {\mathcal {T}}) ^ {T} g _ {\mathcal {T}}} \\ & {- \mathcal {L} (\theta_ {v}, \theta_ {l}, \theta_ {\mathcal {T}}) + + O (\lambda^ {2})} \\ & {= - \lambda (g _ {v} ^ {T} \cdot g _ {v} + g _ {l} ^ {T} \cdot g _ {l})} \\ & {- \lambda (g _ {\mathcal {T}} ^ {l} + g _ {\mathcal {T}} ^ {v}) ^ {T} (g _ {\mathcal {T}} ^ {l} + g _ {\mathcal {T}} ^ {v})} \\ & {+ O (\lambda^ {2})} \\ & {= - 2 \lambda g _ {\mathcal {T}} ^ {v} \cdot g _ {\mathcal {T}} ^ {l} - \lambda \sum_ {i \in \{v, l, \mathcal {T} \}} (g _ {i} \cdot g _ {i} + g _ {\mathcal {T}} ^ {i} \cdot g _ {\mathcal {T}} ^ {i})} \\ & {+ O (\lambda^ {2})} \end{array}
$$

Influence of Fusion Methods. The term $\left( g _ { T } ^ { v } \right) ^ { \top } ( g _ { T } ^ { l } )$ captures how the visual and language gradients interact within the classifier parameters $\theta _ { T }$ Different fusion methods yield different dependencies of $g _ { T } ^ { v }$ and $g _ { T } ^ { l }$ on v and l: 

• Addition: The fused input is $x \ = \ v + l .$ Because v and l are merged by simple addition, their representations feed directly into the same part of the classifier. Consequently, $\left( g _ { T } ^ { v } \right) ^ { \top } ( g _ { T } ^ { l } )$ often remains significant due to the shared pathway. 

• Concatenation: The fused input is $x = [ v ; l ]$ Each modality is placed in distinct segments of the classifier’s input vector, reducing direct interactions. As a result, $g _ { T } ^ { v }$ and $g _ { T } ^ { l }$ may be more independent, potentially lowering the cross term $\left( g _ { T } ^ { v } \right) ^ { \top } ( g _ { T } ^ { l } )$ 

• Attention: The fused input is $x \quad =$ Attention $( v , l )$ This method can create strong interdependence between v and l within $\theta _ { T }$ . Hence, it $\left( g _ { T } ^ { v } \right) ^ { \top } ( g _ { T } ^ { l } )$ can become highly influential since changes v affect l and vice versa through the attention mechanism. 

Hence, the sign and magnitude of the cross term reflect how strongly the parameters for the two modalities are tied together under each fusion strategy. 

## A.2 Proof of proposition 2

Proposition 2. (Gradient Conflicts on Loss Reduction with KL Loss) Let $\mathcal { G } ^ { \tau } = \{ g _ { v } , g _ { l } , g _ { \tau } \}$ and $\mathcal { G } ^ { k l } = \{ g _ { v } ^ { k l } , g _ { l } ^ { k l } , 0 \}$ be the gradients from a target loss $\scriptstyle { \mathcal { L } } _ { \tau }$ and a KL loss $\mathcal { L } _ { k l }$ , respectively, with parameters $\theta = [ \theta _ { v } , \theta _ { l } , \theta _ { \tau } ] ^ { \top }$ . Assume the parameters are updated by gradient descent with a small step size $\lambda > 0 \colon \theta _ { v } ^ { \prime } = \theta _ { v } - \lambda \bigl ( g _ { v } + g _ { v } ^ { k l } \bigr ) , \quad \theta _ { l } ^ { \prime }$ = $\theta _ { l } - \lambda \left( g _ { l } + g _ { l } ^ { k l } \right) , \quad \theta _ { \tau } ^ { \prime } = \theta _ { \tau } - \lambda g _ { \tau }$ . Then, for the combined loss $\mathcal { L } = \mathcal { L } _ { \tau } + \mathcal { L } _ { k l }$ , the change in the loss is 

$$
\begin{array}{l} \Delta \mathcal {L} = \mathcal {L} \big (\theta^ {\prime} \big) - \mathcal {L} \big (\theta \big) \\ \qquad = - \lambda \Big (\| \mathcal {G} ^ {\tau} \| ^ {2} + \| \mathcal {G} ^ {k l} \| ^ {2} + 2 \left(\mathcal {G} ^ {\tau}\right) ^ {\top} \mathcal {G} ^ {k l} \Big) \\ \qquad + O (\lambda^ {2}). \end{array}\tag{8}
$$

In particular, if $( \mathcal { G } ^ { \tau } ) ^ { \top } \mathcal { G } ^ { k l } < 0$ , the gradients from the target and KL losses conflict, reducing the effective loss reduction. 

Proof of Proposition 2. 

Proof. Because $\mathcal { L } = \mathcal { L } _ { \tau } + \mathcal { L } _ { k l }$ , its gradient is 

$$
\nabla_ {\theta} \mathcal {L} = \mathcal {G} ^ {\tau} + \mathcal {G} ^ {k l}.
$$

Under a small step size λ, a first-order Taylor expansion about θ gives $\Delta \mathcal { L } \approx - \lambda \| \mathcal { G } ^ { \tau } + \mathcal { G } ^ { k l } \| ^ { 2 }$ Since $\mathcal { G } ^ { \tau } = \{ g _ { v } ^ { \tau } , g _ { l } ^ { \tau } , g _ { \tau } ^ { \tau } \}$ and $\mathcal { G } ^ { k l } = \overset { \cdot \cdot } { \{ \boldsymbol { g } _ { v } ^ { k l } , \boldsymbol { g } _ { l } ^ { k l } , 0 \} }$ the relevant parameters are updated as: 

$$
\begin{array}{r l} & {\theta_ {v} ^ {\prime} = \theta_ {v} - \lambda (g _ {v} ^ {\tau} + g _ {v} ^ {k l})} \\ & {\theta_ {l} ^ {\prime} = \theta_ {l} - \lambda (g _ {l} ^ {\tau} + g _ {l} ^ {k l})} \\ & {\theta_ {\tau} ^ {\prime} = \theta_ {\tau} - \lambda g _ {\tau} ^ {\tau}.} \end{array}
$$

By decomposing norm: 

$$
\begin{array}{r l} & {\Delta \mathcal {L} = - \lambda \Big (\| g _ {v} ^ {\tau} \| ^ {2} + \| g _ {v} ^ {k l} \| ^ {2} + 2 g _ {v} ^ {\tau}) ^ {\top} (g _ {v} ^ {k l})} \\ & {\qquad + \| g _ {l} ^ {\tau} \| ^ {2} + \| g _ {l} ^ {k l} \| ^ {2} + 2 g _ {l} ^ {\tau}) ^ {\top} (g _ {l} ^ {k l}) + \| g _ {\tau} ^ {\tau} \| ^ {2} \Big)} \\ & {\qquad + O (\lambda^ {2})} \end{array}
$$

Hence, if either $( g _ { v } ^ { \tau } ) ^ { \top } ( g _ { v } ^ { k l } ) < 0$ or $( g _ { l } ^ { \tau } ) ^ { \top } ( g _ { l } ^ { k l } ) < 0$ , the negative cross-term reduces the effective loss decrease for that modality. 

## B Further Implementation Details

## B.1 Dataset and Evaluation Metrics

UPMC Food-101 (Wang et al., 2015) is a food classification dataset with 101 categories and 90,840 image-text pairs, involving the classification of food items using both images and textual recipe descriptions; to create a validation split, we extracted 5,000 samples from the training set (Kiela et al., 2019), as the dataset only provides training and testing sets. 

Hateful Memes (Kiela et al., 2020) is designed to detect hate speech by combining image and text modalities, comprising 8,500 training samples, 1,000 validation samples, and 500 test samples. 

MM-IMDb (Arevalo et al., 2017) is a multi-label movie genre classification dataset that incorporates poster images and plot descriptions, containing 23 genre tags with 15,552 training samples, 2,608 validation samples, and 7,799 test samples. 

We utilize classification accuracy, AUROC, and F1-Macro as evaluation metrics for the UPMC Food-101, Hateful Memes, and MM-IMDb datasets, respectively. 

## B.2 Architecture and Training Scheme

In all comparative experiments, we employ ViT (Dosovitskiy et al., 2021) and BERT (Devlin et al., 2019) as image and text encoders, respectively. We adopt a late concatenation architecture where the embeddings from each modality are concatenated to make the final prediction. We employ linear probing as our fine-tuning strategy, which freezes all the encoder parameters and trains only the embedding and classifier layers. 

We adopt this modular architecture and finetuning scheme for several key reasons: First, the modular design of BALGRAD allows it to extend to various encoders, easily accommodating different architectures. This flexibility is crucial in real-world scenarios where resources are often constrained. Our structure supports a range of scalable encoder configurations, ensuring adaptability to different resource availability and application requirements. Additionally, in some cases, data access is restricted due to privacy concerns, necessitates the use of pre-extracted features (Cheplygina et al., 2019; Kruk et al., 2019; Menini et al., 2020), making the application of early fusion-based large VLMs (Liu et al., 2024) impractical. Also, to focus improvements on BALGRAD’s gradient reweighting and projection, we adopt linear probing as a fine-tuning strategy, ensuring that the gains were not merely due to the encoders’ inherent capabilities but to our method’s effectiveness. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/ceb22bb8-93ba-472a-b9f2-21044acb3d25/53bebd5b80e60d5369c43360a58515e71e1d51802ccce79833609640236768f4.jpg)



Figure 8: Bar plots illustrating the performance of existing methods and BALGRAD with different fusion mech anisms: (a) addition and (b) attention, evaluated on the UPMC Food-101, Hateful Memes, and MM-IMDb datasets. Each bar indicates $\Delta _ { G a p } ( \% )$ , which quantifies the performance variation between missing image and missing text conditions.


As a baseline, we adopt a standard linear probing approach and compare our proposed method against existing methods designed to balance modalities in VL models, specifically MSLR (Yao and Mihalcea, 2022), OGM-GE (Peng et al., 2022), and AGM (Li et al., 2023). 

## B.3 Implementation Details

We use vit-base and bert-base-uncased checkpoints as the image and text encoders, respectively, loading them from Transformers library (Wolf et al., 2020). The embeddings extracted from each encoder have a dimensionality of 768, and we concatenate these embeddings to form a 1568- dimensional vector, which is then passed to a final classifier. We resize all images to $2 2 4 \times 2 2 4$ and apply a random horizontal flip for augmentation. For text, the maximum sequence lengths are set to 1024 for MM-IMDb, 512 for UPMC Food-101, and 128 for Hateful Memes. We use the Adam optimizer with a momentum of 0.9 for all experiments, training for 20 epochs with a batch size of 128. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/ceb22bb8-93ba-472a-b9f2-21044acb3d25/ad67126eee67beb74e4be9e38b588748bd4279e9b1f84ff34c0f1cf98349622e.jpg)



Figure 9: Bar plots presenting the performance comparison between existing methods and BALGRAD across different backbone models: (a) ResNet and DistilBERT, and (b) CLIP, on the UPMC Food-101, Hateful Memes, and MM-IMDb datasets. Each bar represents $\Delta _ { G a p } ( \% )$ measuring the performance discrepancy under missing image and missing text conditions.


## C Additional Experimental Results

## C.1 Experimental Results on Different Fusion Mechanisms

The way embeddings from different modalities are fused can significantly impact a model’s ability to capture and leverage cross-modal interactions. We conducted experiments on different fusion strategies in the baseline and BALGRAD, specifically exploring element-wise addition and attention-based fusion mechanisms following previous work (Kumar and Nandakumar, 2022). We tested these mechanisms on the UPMC Food-101, Hateful Memes, and MM-IMDb datasets, evaluating the $\Delta _ { G a p }$ in performance under conditions where either the image or text modality was missing. Results for addition and attention are presented in Figure 8. Across all datasets, BALGRAD demonstrated the smallest $\Delta _ { G a p }$ with both fusion mechanisms, effectively mitigating dominant modality bias. This confirms that BALGRAD effectively captures and leverages cross-modal interactions across different fusion mechanisms. 

## C.2 Experimental Results on Different Backbone Models

We conduct extensive experiments across diverse backbone models, underscoring its consistent performance and adaptability to varying architectures and computational resources. Specifically, we employ lower-capacity models—ResNet-50 (He et al., 2016) for the image encoder and DistilBERT (Sanh et al., 2019) for the text encoder—to assess robustness concerning model size. Additionally, we leverage the widely-used multimodal pretrained VLM, CLIP (Radford et al., 2021), for further evaluation due to its strong ability to seamlessly integrate visual and textual information, providing a rigorous test for BALGRAD. Experiments were carried out on the UPMC Food-101, Hateful Memes, and MM-IMDb datasets, assessing the performance gap under conditions where either the image or text modality was missing. Results for ResNet-DistilBERT and CLIP are presented in Figure 9. Across all datasets, BALGRAD consistently exhibited the smallest $\Delta _ { G a p }$ , effectively balancing the contributions between modalities. 

Intriguingly, while earlier experiments using ViT and BERT encoders on the MM-IMDb dataset showed no over-reliance on a specific modality, our additional studies reveal that conventional methods tend to heavily rely on the text modality, when employing ResNet and DistilBERT. These findings indicate that such bias is influenced not only by the task but also by the choice of backbone model. Our comprehensive experiments affirm that BAL-GRAD effectively mitigates bias irrespective of the backbone model employed, showcasing its superior scalability. 

## C.3 Experimental Results with Additional Datasets

To validate the generalizability of BALGRAD, we conduct experiments on two additional datasets: Memotion (Mishra et al., 2023) and CUB-200- 2011 (Welinder et al., 2010). The Memotion dataset, used for classifying the humor level of meme images based on their descriptions, includes annotations such as “not funny”, “funny”, “very funny”, and “hilarious”. The CUB-200-2011 dataset is a fine-grained bird classification dataset, requiring the categorization of 200 bird species based on images and descriptions. We evaluate each dataset using weighted F1 score and classification accuracy. 

The results for the Memotion dataset, presented in Table 4, show that when the text modality is missing, performance drops significantly more than when the image modality is missing, indicating a bias toward the text modality. BALGRAD not only achieves the highest performance with the image modality alone but also excels in the Avg. and $\Delta _ { G a p }$ metrics, demonstrating effective modality balance. 

<table><tr><td rowspan="2" colspan="2">Modality</td><td colspan="5">Memotion</td></tr><tr><td>Baseline</td><td>MSLR</td><td>OGM-GE</td><td>AGM</td><td>BALGRAD</td></tr><tr><td colspan="2">Full</td><td>70.55</td><td>70.36</td><td>70.28</td><td>71.12</td><td>70.77</td></tr><tr><td rowspan="4">Missing</td><td>Image</td><td>58.34</td><td>59.24</td><td>59.66</td><td>59.54</td><td>59.48</td></tr><tr><td>Text</td><td>49.29</td><td>51.32</td><td>50.38</td><td>51.44</td><td>52.78</td></tr><tr><td>Avg.↑</td><td>53.82</td><td>55.28</td><td>55.02</td><td>55.49</td><td>56.13</td></tr><tr><td><eq>\Delta_{Gap} \downarrow</eq></td><td>4.53</td><td>3.96</td><td>4.64</td><td>4.05</td><td>3.35</td></tr></table>


Table 4: The experimental result of BALGRAD on the Memotion dataset. The best result in each test dataset is boldfaced, and the second best is presented with underlining. “Avg.” represents the average performance under conditions where one of the modality is missing, while $\begin{array} { r } { \ddot { \Delta } _ { G a p } ( \% ) \displaylimits ^ { , } } \end{array}$ indicates the performance difference.


<table><tr><td rowspan="2" colspan="2">Modality</td><td colspan="5">CUB-200-2011</td></tr><tr><td>Baseline</td><td>MSLR</td><td>OGM-GE</td><td>AGM</td><td>BALGRAD</td></tr><tr><td colspan="2">Full</td><td>74.71</td><td>72.12</td><td>75.15</td><td>76.28</td><td>75.84</td></tr><tr><td rowspan="4">Missing</td><td>Image</td><td>37.38</td><td>40.21</td><td>39.49</td><td>41.24</td><td>45.47</td></tr><tr><td>Text</td><td>61.24</td><td>60.20</td><td>59.14</td><td>61.42</td><td>62.72</td></tr><tr><td>Avg.↑</td><td>49.31</td><td>50.21</td><td>49.32</td><td>51.33</td><td>54.10</td></tr><tr><td><eq>\Delta_{Gap} \downarrow</eq></td><td>11.93</td><td>9.99</td><td>9.83</td><td>10.09</td><td>8.63</td></tr></table>


Table 5: The results of BALGRAD on the CUB-200- 2011 dataset are presented. The highest performance in each test dataset is shown in bold, with the secondhighest underlined. “Avg.” reflects the average performance when one modality is absent, and $^ { * } \Delta _ { G a p } ( \% ) ^ { 3 }$ denotes the performance difference.


As shown in Table 5, the CUB-200-2011 dataset exhibits a strong reliance on the image modality. BALGRAD outperforms AGM by more than 4%p in accuracy when the image modality is missing and achieves the smallest $\Delta _ { G a p }$ at 8.63%, demonstrating its superiority in handling fine-grained classification tasks even under challenging conditions. 