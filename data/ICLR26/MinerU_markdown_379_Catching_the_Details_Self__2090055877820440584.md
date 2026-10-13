# CATCHING THE DETAILS: SELF-DISTILLED ROI PRE-DICTORS FOR FINE-GRAINED MLLM PERCEPTION

Yuheng Shi<sup>1</sup> Xiaohuan Pei<sup>1</sup> Minjing Dong<sup>2</sup> Chang Xu<sup>1</sup> 

<sup>1</sup>University of Sydney <sup>2</sup>City University of Hong Kong 

yshi0087@uni.sydney.edu.au, xiaohuan.pei@sydney.edu.au, minjdong@cityu.edu.hk, c.xu@sydney.edu.au 

## ABSTRACT

Multimodal Large Language Models (MLLMs) require high-resolution visual information to perform fine-grained perception, yet processing entire highresolution images is computationally prohibitive. While recent methods leverage a Region-of-Interest (RoI) mechanism to focus on salient areas, they typically present a difficult trade-off: training-based approaches depend on large-scale annotated datasets, while training-free methods that utilize the model’s internal attention are computationally inefficient and less accurate, requiring either multipass prefill stages or reliance on the slow auto-regressive decoding process. In this paper, we propose an efficient, annotation-free Self-Distilled Region Proposal Network (SD-RPN) that resolves this trade-off. The SD-RPN is built around a pipeline that transforms the noisy attention maps from the MLLM’s middle layers into high-quality pseudo-RoI labels by explicitly denoising the signal and resolving ambiguity. We use these labels to train a Region Proposal Network (RPN) that learns a more precise localization. This RPN is also highly efficient, predicting the RoI in a single forward pass using features from the MLLM’s middle layers, decoupling RoI identification from the auto-regressive generation and avoiding costly multi-pass operations. To validate our approach, we integrate the framework into the LLaVA-1.5 architecture. Despite being trained on only a few (e.g. 10K) question-answer pairs, our method demonstrates exceptional data efficiency and generalization, achieving over a 10% absolute accuracy improvement on unseen benchmarks, including TextVQA, DocVQA, and V-Star. Our work presents a practical and scalable solution for enhancing the fine-grained perception of MLLMs without requiring costly supervision or full model fine-tuning. Code is available at https://github.com/YuHengsss/SD-RPN. 

## 1 INTRODUCTION

Recent years have witnessed significant advancements in Multimodal Large Language Models (MLLMs), which have evolved from foundational architectures like LLaVA (Liu et al., 2023b) to more sophisticated systems (Wang et al., 2024a; Bai et al., 2023; Chen et al., 2024a;b) such as Qwen2.5-VL (Bai et al., 2025) and InternVL-3.0 (Zhu et al., 2025). The performance of MLLMs is tied to the quality of their visual perception. In a typical MLLM architecture, a vision encoder, such as CLIP (Radford et al., 2021), processes visual signals and projects them into the embedding space of a Large Language Model (LLM) (Touvron et al., 2023; Zheng et al., 2023) for subsequent reasoning. Consequently, the richness of these visual features is crucial for the model to achieve a comprehensive and fine-grained understanding of the input. 

To enhance this fine-grained perception, scaling the resolution of visual inputs has emerged as a direct and effective strategy. The approaches to achieve this have evolved considerably. Initial methods (Liu et al., 2023b; Dai et al., 2023; Bai et al., 2023) relied on a fixed-resolution input. Subsequent research introduced more flexible techniques, such as S<sup>2</sup> (Shi et al., 2024), Any-Resolution (Liu et al., 2023a; Chen et al., 2024b), and Naive Dynamic Resolution (Wang et al., 2024a; Bai et al., 2025) to handle higher-resolution imagery. However, processing entire high-resolution images uniformly is computationally intensive. More recently, an alternative paradigm (Yu et al., 2025; Shi et al., 2025a) has gained traction: identifying a Region-of-Interest (RoI) within a low-resolution input and then selectively scaling up only that specific region. This RoI-centric approach has proven to be a more efficient and effective means of improving visual detail perception. 


(a)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/e11b9957-d09a-473b-8ecb-edef7d68b919/a1d255507706f7c034ab56b36fc3b8afe06ee7bcd73061f5a2887bdde0e9febe.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/e11b9957-d09a-473b-8ecb-edef7d68b919/bcb1478d46742e14c109a35e39c3def50252e6cc7590b47a353042452ce47469.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/e11b9957-d09a-473b-8ecb-edef7d68b919/a5c1601c4a94398af162c05528949b348f4c0f20e299e124b39f5dd5117dad45.jpg)



(b)



Figure 1: (a) The Pipeline of SD-RPN. The RPN is trained with pseudo-labels to effectively predict RoIs. These RoIs are then used to crop fine-grained sub-images for the final inference stage. (b) Performance Comparison. Performance evaluation with $S ^ { 2 }$ (Shi et al., 2024) and ViCrop (Zhang et al., 2025a) on the LLaVA-1.5-7B baseline. Accuracy is averaged over five Document and OCR benchmarks.


While the RoI paradigm marks a significant step toward efficient high-resolution processing, current methodologies for identifying these regions still present notable limitations. Approaches such as VILA-HD (Shi et al., 2025a), for instance, rely on large-scale pre-training with detailed annotations, a process that is both data-intensive and computationally demanding. Furthermore, they often require a complete prefilling stage for the initial low-resolution image, which can impede both training and inference efficiency. An alternative direction leverages the intrinsic localization capabilities of MLLMs (Kang et al., 2025b; Zhang et al., 2025a) without training, specifically by computing cross-attention scores between image tokens and the corresponding textual description. However, leveraging this internal attention for RoI identification is often computationally inefficient in inference, as current methods typically require either complex, multi-pass operations during the prefill stage (Zhang et al., 2025a) or rely on the inherently sequential and slow auto-regressive decoding stage (Wang et al., 2024b). Consequently, an effective and efficient method for RoI identification that avoids both reliance on extensive annotated data and the high latency of auto-regressive decoding or multiple forward passes remains a critical, yet underexplored, challenge. 

In this work, we introduce SD-RPN, a novel self-distillation framework (as shown in Fig. 1a) designed to overcome these limitations by efficiently harnessing the MLLM’s intrinsic localization capabilities. Our approach is motivated by the insight that while an MLLM’s internal attention provides a strong RoI signal, it is too noisy (e.g., attention sinks, incomplete activation) for direct supervision. Previous studies reveal that using such noisy signals for dense supervision yields suboptimal results (Wang et al., 2022; Tian et al., 2021). To address this, we propose a pseudo-labeling pipeline that transforms the noisy attention map into a sparse and reliable supervisory signal. This pipeline first denoises the map by removing sink tokens based on feature norms. Subsequently, it employs a selective classification strategy that assigns discrete labels based on confidence thresholds and a minimal bounding box around high-attention tokens, which resolves the ambiguity inherent in the original noisy map. We use these pseudo labels to train a RPN. By learning from this distilled knowledge, the RPN develops a more precise localization function than methods relying on raw, ambiguous attention. Beyond its accuracy, the RPN’s architectural design confers significant efficiency gains. Composed of a few transformer blocks built upon the frozen MLLM backbone, the RPN operates on features from the model’s middle layers. This strategic placement allows it to predict the RoI by executing only a partial forward pass up to the MLLM’s middle layers, completely decoupling localization from the slow, auto-regressive generation process. The entire framework is trained end-to-end, distilling the localization knowledge from the model’s own response-guided attention into the efficient RPN. This enables a dynamic two-stage inference process where the model first predicts salient regions and then analyzes high-resolution crops to generate its final response. 

Our SD-RPN achieves a significant enhancement (Fig. 1b) in fine-grained perception without the cost of full-model fine-tuning or large-scale annotated datasets. Moreover, our framework demonstrates exceptional training efficiency and generalization. Using only 10K random samples from GQA (Hudson & Manning, 2019) and OCR-VQA (Mishra et al., 2019) within the LLaVA-1.5 (Liu et al., 2023a) framework, our approach yields substantial gains on unseen data, boosting accuracy by over 10% on TextVQA (Singh et al., 2019), DocVQA (Mathew et al., 2020), and V-Star (Wu & Xie, 2023). We further confirm the robustness of our method by applying it to the more recent DeepSeek-VL (Lu et al., 2024) and Qwen2.5-VL (Bai et al., 2025) architectures. Our contributions are twofold. First, we introduce a robust pipeline to denoise the internal attention maps of an MLLM, generating high-quality pseudo-labels for supervision. Second, we propose a novel, annotation-free self-distillation framework that trains a RPN to predict RoIs by leveraging the MLLM’s intrinsic localization knowledge. 

## 2 RELATED WORKS

## 2.1 PERCEPTION IN MLLMS.

Recent studies have established that MLLMs often struggle with fine-grained perception, a limitation rooted in the challenge of efficiently processing high-resolution visual inputs (Tong et al., 2024b; Kang et al., 2025b; Shi et al., 2024; Liu et al., 2024a; Chen et al., 2024a). One line of research has focused on enhancing the model’s global visual understanding. This includes developing more sophisticated vision encoders (Wang et al., 2024a; Bai et al., 2025; Zhang et al., 2024b; Ge et al., 2024; Luo et al., 2024), supplementing the LLM with full high-resolution images (Liu et al., 2024a; Zhu et al., 2025; Huang et al., 2024; Tong et al., 2024a; Cha et al., 2024; Li et al., 2025; 2024b), and incorporating external tools (Zhao et al., 2024). More recent works Shi et al. (2025a); Yu et al. (2025); Zheng et al. (2025); Shao et al. (2024) have demonstrated that identifying the RoI first in relatively low-resolution visual input and then scaling the resolution specifically for the RoI is more efficient and effective. However, they come at the expense of extensive training, requiring massive supervision and costly annotations. While recent training-free methods (Zhang et al., 2025a; Wang et al., 2024b; Shi et al., 2025b) attempt to identify RoIs by utilizing the internal perceptual capabilities of VLMs, they are often hampered by noisy activations or require slow auto-regresssive decoding stage and multiple forward passes, which hinders both performance and efficiency. 

## 2.2 SELF-DISTILLATION IN MLLMS.

Knowledge distillation is an effective paradigm for transferring knowledge from a teacher to a student network (Hinton et al., 2015). Among its variants, self-distillation (Zhang et al., 2019) emerges as a unique form in which the teacher and student share the same architecture, and it has been widely adopted in vision–language tasks. In multimodal pre-training, several works (Oquab et al., 2023; Cai et al., 2024; Dong et al., 2023; Kim et al., 2025) leverage self-distillation to improve cross-modal alignment and representation learning, where it is commonly employed to enhance feature extraction. In multimodal downstream tasks, recent efforts (Kong et al., 2024; Peng et al., 2025; Hou et al., 2024) take advantage of self-distillation to refine vision–text region alignment and thereby improve grounding and reasoning performance. As self-distillation eliminates the need for a larger pre-trained teacher, it naturally offers strong potential for extracting high-resolution and fine-grained perceptual cues directly from MLLMs themselves. 

## 3 METHOD

## 3.1 PRELIMINARIES

The standard architecture of a Multimodal Large Language Model (MLLM) comprises three core components: a vision encoder, ${ \mathcal { E } } _ { v } ,$ a vision-language projector, ${ \mathcal { P } } _ { : }$ , and a Large Language Model (LLM), L. Given an image-text pair, $( x _ { v } , x _ { t } )$ , the MLLM processes the inputs to generate a textual response. The vision encoder ${ \mathcal { E } } _ { v }$ extracts feature vectors from the input image $x _ { v }$ . These visual features are then transformed by the projector $\mathcal { P }$ into a sequence of visual embeddings, $\mathbf { H } _ { v } ^ { 0 } \ = \ \mathcal { P } ( \mathcal { E } _ { v } ( x _ { v } ) )$ , which are aligned with the LLM’s input space. The input text $x _ { t }$ is processed by a tokenizer and an embedding layer to produce a sequence of text embeddings, which typically includes system prompt and user query, yielding $\mathbf { H } _ { s y s } ^ { 0 }$ and $\mathbf { H } _ { u s e r } ^ { 0 } ,$ , respectively. After encoding ${ \bf { H } } _ { { s y s } } ^ { 0 } , { \bf { H } } _ { v } ^ { 0 } , { \bf { H } } _ { u s e r } ^ { 0 }$ in parallel (called the prefilling stage), the MLLM generates responses auto-regressively (called the decoding stage). Due to the lack of parallelization, the decoding stage is typically much slower than the prefilling stage when processing the same number of tokens. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/e11b9957-d09a-473b-8ecb-edef7d68b919/fef1669182c97decdb51bf47ca2ac61422462e5b708c7ca33cbfa2daa709c18f.jpg)



Figure 2: An overview of our pseudo-label generation pipeline. FG and BG denote the foreground and background respectively. Layer index is omitted for simplicity.


The attention scores are key to identifying image regions relevant to the text (Kang et al., 2025b). From the cross-modal attention in layer l, we could derive a RoI map, $\mathbf { M } _ { \mathrm { R o I } } ^ { l } \ \in \ \overline { { \mathbb { R } } } ^ { H \times \dot { W } }$ , which represents the averaged importance of each visual token across textual tokens of user query or response. For a single attention head, this map is computed as $\begin{array} { r } { \mathbf { M } _ { \mathrm { R o I } } ^ { l } \ = \ \sum _ { i = 1 } ^ { N _ { t } } \mathbf { A } _ { i } ^ { l } / N _ { t } , \ \mathbf { A } \ = \ } \end{array}$ softmax $\left( \mathbf { Q } _ { t } ^ { l } ( \mathbf { K } _ { v } ^ { l } ) ^ { T } / \sqrt { d } \right)$ , where $\mathbf { Q } _ { t } ^ { l } \in \mathbb { R } ^ { N _ { t } \times d }$ and $\mathbf { K } _ { v } ^ { l } \in \mathbb { R } ^ { ( H \times W ) \times d }$ are the query and key matrices from the $N _ { t }$ response tokens and $H \times W$ visual tokens, respectively. 

## 3.2 PSEUDO-LABEL GENERATION FOR ROI

While the RoI map ${ \bf M } _ { \mathrm { R o I } }$ provides a valuable signal for localizing text-relevant image regions, it is often fraught with noise. As illustrated in Fig. 2, these raw maps can exhibit erroneously high attention in background areas and incomplete activation across the true foreground object. To overcome this, we propose a pseudo-label generation pipeline, which is depicted in Fig. 2, to transform the noisy RoI map into a sparse and reliable supervisory signal for training our RPN. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/e11b9957-d09a-473b-8ecb-edef7d68b919/51015b6bc448a6ece25f6cdf6e40fc90b48484b4a903c950ee54a3e754309b2e.jpg)


The first source of noise we address is the sink tokens. These are visual tokens that attract substantial attention despite being semantically irrelevant to the grounded object. As identified in recent studies (Darcet et al., 2023; Kang et al., 2025a), sink 

Figure 3: Attention magnitude VS. Localization accuracy. 

tokens can be identified by the high L2-norm of their corresponding feature vectors $\mathbf { H } _ { v }$ . To mitigate them, we first denoise the initial RoI map ${ \bf M } _ { \mathrm { R o I } }$ suppressing sink tokens identified via a predefined norm threshold $\tau _ { \mathrm { n o r m } } .$ yielding a cleaner version, $\mathbf { M } _ { \mathrm { R o I } } ^ { \prime } \colon$ 

$$
(\mathbf {M} _ {\mathrm{RoI}} ^ {\prime}) _ {j} = \left\{ \begin{array}{l l} 0 & \text { if } | | (\mathbf {H} _ {v}) _ {j} | | _ {2} > \tau_ {\mathrm{norm}} \\ (\mathbf {M} _ {\mathrm{RoI}}) _ {j} & \text { otherwise } \end{array} \right..\tag{1}
$$

Label Assignment. Following the removal of sink tokens, the denoised RoI map $\mathbf { M } _ { \mathrm { R o I } } ^ { \prime }$ more clearly highlights the foreground area. However, it is not yet an ideal supervisory signal, as it still suffers from an obscure foreground-background margin and incomplete object activation. To empirically investigate the margin issue, we analyze the denoised attention maps $\mathbf { M } _ { \mathrm { R o I } } ^ { \prime }$ on the TextVQA subset (Zhang et al., $2 0 2 \bar { 5 } \mathrm { a } )$ . Fig. 3 plots the proportion of tokens falling inside the ground-truth (GT) bounding box as a function of their sample-wise relative attention score, calculated as $a / a _ { \mathrm { m a x } }$ for each attention value $a \in \mathbf { M } _ { \mathrm { R o I } } ^ { \prime }$ . The histogram reveals that the proportion of tokens inside the GT box is approximately 40% for high attention scores $( \mathrm { e } . \mathrm { g } . , > 0 . 2 )$ and 10% for low scores $( \mathrm { e . g . }$ $< 0 . 1 )$ . However, the ambiguous middle-range of attention not only exhibit a distinct localization pattern from the more decisive high and low scores, but they also vastly outnumber the high-attention tokens. Using such a signal for dense supervision would inevitably force the model to learn from these numerous ambiguous regions. To address this, we avoid regressing on $\mathbf { M } _ { \mathrm { R o I } } ^ { \prime }$ directly and instead implement a selective binary classification that assigns foreground or background only to high-confidence tokens, leaving ambiguous tokens to be ignored. This approach is also designed to alleviate the incomplete activation problem. We define a minimal bounding box, $B _ { f g } ,$ , which encloses identified foreground tokens. Tokens inside this box that is not classified as foreground is explicitly ignored, which prevents the RPN from receiving erroneous background signals from inactivated parts of the true object. Guided by these criteria, the final pseudo-label map, $\bar { \bf M } _ { \mathrm { R o I } }$ is constructed by: 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/e11b9957-d09a-473b-8ecb-edef7d68b919/755fb13cb55d01857d19ce17c44ee8926e75a21aac3ff0928c1120d93588f017.jpg)



Figure 4: Overview of our SD–RPN framework. Our RPN (top) is initialized from and built upon a frozen MLLM backbone to efficiently predict a dense RoI map. It is trained via self-distillation (bottom), where pseudo-labels are generated by denoising the full MLLM’s internal response-toimage attention maps. Superscripts denote layer indices; subscripts denote token sources. We omit the system prompt tokens for brevity.


$$
(\bar {\mathbf {M}} _ {\mathrm{RoI}}) _ {j} = \left\{ \begin{array}{l l} 1 & \text { if   token } j \in \mathcal {S} _ {f g}, \\ 0 & \text { if   token } j \in \mathcal {S} _ {b g}, \\ - 1 & \text { otherwise   (ignored) }, \end{array} \right.\tag{2}
$$

where the foreground set is $S _ { f g } = \{ j \ | \ a _ { j } \ \geq \ \tau _ { f g } a _ { \mathrm { m a x } } \}$ and the background set is $S _ { b g } = \{ j \ | \ \}$ $j \notin B _ { f g }$ and $a _ { j } \leq \tau _ { b g } a _ { \mathrm { m a x } } \}$ . In Appendix F, we also provide a theoretical view of why learning to predict RoI labels is superior than using the raw attention maps directly. 

## 3.3 ROI PREDICTION VIA SELF-DISTILLATION

As validated by recent studies (Kang et al., 2025b; Shao et al., 2025), the middle layers of MLLMs exhibit significant potential for RoI prediction. Given these observations, can we leverage the localization capabilities inherent in the middle layers of MLLMs to build a more efficient RoI predictor in both inference and data? To answer this question, we propose a tunable Self-Distilled Region Proposal Network (SD-RPN) that consists of R transformer blocks built upon the first B frozen MLLM layers, which serve as the backbone. 

Predicting the RoI Map. The architecture of our SD-RPN is illustrated in Fig. 4. We initialize the RPN’s weights using those from layers B to $B + R$ of the pretrained MLLM, a strategy that enables the efficient transfer of learned representations (Shao et al., 2025). Instead of predicting sparse bounding boxes, our well-initialized RPN is trained to predict a dense RoI map, $\hat { { \bf M } } _ { \mathrm { R o I } }$ . The prediction process is triggered by specific query tokens derived from the conversational context. Concretely, for a given image and a conversation with n turns, we first collect the sequence of hidden states, H, from the second last layer of the RPN: 

$$
\mathbf {H} = [ \mathbf {H} _ {\mathrm{sys}}, \mathbf {H} _ {v}, \mathbf {H} _ {u (1)}, \mathbf {H} _ {r (1)}, \ldots , \mathbf {H} _ {u (n)}, \mathbf {H} _ {r (n)} ],\tag{3}
$$

where the subscript of u and r denote user and response tokens. From this sequence, we specifically isolate the hidden state corresponding to the final token of each user question, ${ \mathbf { H } } _ { u ( i ) } [ - 1 ]$ , as these tokens immediately precede the model’s answers and serve as the most direct prompt for generating a grounded response. These n query vectors are collected into a single tensor, $\mathbf { H } _ { \mathrm { R o I } }$ , which serves as the input for our RoI prediction head: 

$$
\mathbf {H} _ {\text { RoI }} = \operatorname{concat} (\mathbf {H} _ {u (1)} [ - 1 ], \dots , \mathbf {H} _ {u (n)} [ - 1 ]),\tag{4}
$$

where $\mathbf { H } _ { \mathrm { R o I } } \in \mathbb { R } ^ { n \times d }$ . These query vectors and token hidden states $\mathbf { H } _ { v } ,$ , are then projected into the query and key space using the linear layers $( L P _ { q }$ and $L P _ { k } )$ from the RPN’s final attention block: 

$$
\mathbf {Q} _ {\mathrm{RoI}} = L P _ {q} (\operatorname{Norm} \left(\mathbf {H} _ {\mathrm{RoI}}\right)), \quad \mathbf {K} _ {v} = L P _ {k} (\operatorname{Norm} \left(\mathbf {H} _ {v}\right)),\tag{5}
$$

where Norm denotes the layer normalization (Zhang & Sennrich, 2019; Ba et al., 2016). The predicted dense RoI map, $\hat { { \bf M } } _ { \mathrm { R o I } }$ is then computed via a simple matrix multiplication of these query and key matrices: $\hat { \mathbf { M } } _ { \mathrm { R o I } } = \mathbf { Q } _ { \mathrm { R o I } } \mathbf { K } _ { v } ^ { T }$ . For brevity, the head dimension is omitted here; in practice, attention scores are computed per head and then averaged. Notably, this design is more efficient than exsiting frameworks (Shi et al., 2025a; Zhang et al., 2025a) as it only need a single forward through partial LLM layers. 

Training via Self-Distillation. The RPN is trained entirely through self-distillation to predict the pseudo-label map $\bar { \bf M } _ { \mathrm { R o I } }$ which is generated via the pipeline in Section 3.2 by minimizing a binary cross-entropy (BCE) loss, defined as $\mathcal { L } _ { \mathrm { B C E } } ( \hat { \mathbf { M } } _ { \mathrm { R o I } } , \bar { \mathbf { M } } _ { \mathrm { R o I } } )$ . While the text responses used for pseudo-label generation could theoretically originate from a stronger teacher model $( e . g . , \mathrm { G P T } \ – 4 )$ or human annotations, our empirical results reveal a crucial insight: using the MLLM’s self-predicted responses yields superior performance (detailed in Tab. 4b). We attribute this counter-intuitive result to the principle of representational consistency. Although an external teacher’s response may be more accurate, the attention maps it induces can be out-of-distribution for the student RPN. In contrast, the attention maps from self-generated responses, even if imperfect, are inherently aligned with its own internal visual grounding mechanisms. This creates a more consistent and attainable distillation target, especially in our data-efficient setting. This self-sufficient framework thus removes any dependency on external models or annotated data. 

## 3.4 TWO-STAGE INFERENCE WITH ROI

The predicted RoI map, $\hat { { \bf M } } _ { \mathrm { R o I } }$ , enables a dynamic, two-stage inference process that significantly enhances the model’s fine-grained perception capabilities. The first stage involves predicting and post-processing $\hat { { \bf M } } _ { \mathrm { R o I } }$ to produce a clean binary foreground mask, B. To consolidate activated regions and ensure robustness against noise, the dense map $\hat { { \bf M } } _ { \mathrm { R o I } }$ is reshaped into a 2D map (γ), smoothed with a Gaussian filter (G), and then binarized using a fixed threshold (τ): 

$$
\mathcal {B} (x, y) = \left\{ \begin{array}{l l} 1, & \text { if } \mathcal {G} (\gamma (\hat {\mathbf {M}} _ {\mathrm{RoI}})) (x, y) > \tau , \\ 0, & \text { otherwise }, \end{array} \right.\tag{6}
$$

where $( x , y )$ represents the spatial coordinates. With this binary mask, we proceed to the second stage to extract fine-grained visual features. We explore two different upscaling strategies for the predicted RoI. The first, which we term Box Upscaling, processes each salient region independently. We first identify all distinct connected-component regions, $\{ \mathcal { R } ^ { i } \} _ { i = 1 } ^ { k }$ , within the mask B. A minimal, axis-aligned bounding box, $b _ { i } ,$ is then computed for each region. These bounding boxes are used to crop sub-images, $\{ x _ { v _ { i } } \} _ { i = 1 } ^ { k } .$ , from the original source image, which are then encoded to produce new, high-resolution visual embeddings $\mathbf { H } _ { v _ { \mathrm { b o x } } } ^ { 0 ^ { \scriptstyle { \mathsf { ^ { \scriptstyle { \psi } } } } } }$ 

$$
b _ {i} = \mathsf {b b o x} (\mathcal {R} _ {i}), \quad \mathbf {H} _ {v _ {\mathrm{box}}} ^ {0} = \{\mathcal {P} (\mathcal {E} _ {v} (x _ {v _ {i}})) \} _ {i = 1} ^ {k},\tag{7}
$$

where bbox(·) is the operator that returns the coordinates of the minimal bounding box enclosing the input region. Alternatively, our second strategy, which we term Masked Upscaling, takes a unified approach. This method first computes a single, all-encompassing bounding box, $b _ { \mathrm { a l l } }$ , that encloses the union of all connected foreground regions, $\textstyle \bigcup _ { i = 1 } ^ { k } { \mathcal { R } } _ { i }$ . This unified bounding box is used to crop a single sub-image, $x _ { v _ { \mathrm { a l l } } }$ , which is then passed through the vision encoder and projector to produce a set of high-resolution embeddings $\mathbf { H } _ { v _ { \mathrm { m a s k } } } ^ { 0 ^ { \star } } ;$ 

$$
b _ {\text { all }} = \operatorname{bbox} (\bigcup_ {i = 1} ^ {k} \mathcal {R} _ {i}), \quad \mathbf {H} _ {v _ {\text { mask }}} ^ {0} = \mathcal {P} (\mathcal {B} \circ \mathcal {E} _ {v} (x _ {v _ {\text { all}}})),\tag{8}
$$


Table 1: Performance on Document & OCR benchmarks. Dataset subscripts denote the evaluation split. Performance subscripts show the absolute improvement $( \uparrow )$ over the baseline. Throughput is relative to the baseline, measured on a single NVIDIA A6000 GPU.


<table><tr><td>Methods</td><td>Throughput</td><td>DocVQAval</td><td>ChartQAtest</td><td>OCRBenchtest</td><td>InfoVQAval</td><td>TextVQAval</td><td>Ave.</td></tr><tr><td>LLaVA-1.5-7B</td><td>1.0×</td><td>21.5</td><td>18.1</td><td>31.4</td><td>20.4</td><td>46.1</td><td>27.5</td></tr><tr><td>+S2</td><td>0.70×</td><td>27.1</td><td>18.9</td><td>32.6</td><td>22.5</td><td>52.6</td><td>30.7</td></tr><tr><td>+ViCrop</td><td>0.42×</td><td>27.0</td><td>20.0</td><td>33.2</td><td>21.4</td><td>57.2</td><td>31.8</td></tr><tr><td>+SD-RPN(Ours)</td><td>0.62×</td><td>34.2↑12.7</td><td>20.6↑2.5</td><td>37.3↑5.9</td><td>22.3↑1.9</td><td>58.8↑12.7</td><td>34.6↑7.1</td></tr><tr><td>LLaVA-1.5-13B</td><td>1.0×</td><td>23.5</td><td>18.1</td><td>33.7</td><td>23.4</td><td>48.7</td><td>29.5</td></tr><tr><td>+S2</td><td>0.71×</td><td>30.7</td><td>20.3</td><td>36.4</td><td>24.7</td><td>54.5</td><td>33.3</td></tr><tr><td>+ViCrop</td><td>0.39×</td><td>30.2</td><td>20.1</td><td>36.1</td><td>25.9</td><td>60.3</td><td>34.5</td></tr><tr><td>+SD-RPN(Ours)</td><td>0.51×</td><td>39.4↑15.9</td><td>21.2↑3.1</td><td>39.6↑5.9</td><td>24.8↑1.4</td><td>63.4↑14.5</td><td>37.7↑8.2</td></tr><tr><td>DeepSeek-VL-1.3B</td><td>1.0×</td><td>37.0</td><td>48.6</td><td>38.9</td><td>21.7</td><td>55.8</td><td>40.4</td></tr><tr><td>+SD-RPN(Ours)</td><td>0.47×</td><td>54.4↑17.4</td><td>51.7↑3.1</td><td>40.8↑1.9</td><td>24.1↑2.4</td><td>65.9↑10.1</td><td>47.4↑7.0</td></tr><tr><td>DeepSeek-VL-7B</td><td>1.0×</td><td>50.7</td><td>61.4</td><td>42.4</td><td>30.4</td><td>63.0</td><td>49.6</td></tr><tr><td>+SD-RPN(Ours)</td><td>0.40×</td><td>67.2↑16.5</td><td>64.2↑2.8</td><td>47.9↑5.5</td><td>36.8↑6.4</td><td>71.5↑8.5</td><td>57.5↑7.9</td></tr><tr><td>Qwen2.5-VL-3B</td><td>1.0×</td><td>87.1</td><td>82.6</td><td>73.8</td><td>62.7</td><td>76.9</td><td>76.6</td></tr><tr><td>+SD-RPN(Ours)</td><td>0.56×</td><td>89.5↑2.4</td><td>83.8↑1.2</td><td>76.3↑2.5</td><td>67.0↑4.3</td><td>79.7↑2.8</td><td>79.3↑2.7</td></tr><tr><td>Qwen2.5-VL-7B</td><td>1.0×</td><td>92.0</td><td>82.6</td><td>81.5</td><td>70.2</td><td>81.1</td><td>81.5</td></tr><tr><td>+SD-RPN(Ours)</td><td>0.57×</td><td>93.6↑1.6</td><td>85.5↑2.9</td><td>82.9↑1.4</td><td>76.9↑6.7</td><td>83.5↑2.4</td><td>84.5↑3.0</td></tr></table>


Table 2: Performance on Vision-Centric and High-Resolution benchmarks. † denotes results reported in the original publication.


<table><tr><td rowspan="2">Methods</td><td colspan="3">V* Bench</td><td colspan="2">POPE</td><td colspan="3">HR-Bench 4K</td><td colspan="3">HR-Bench 8K</td></tr><tr><td>Attr</td><td>Spatial</td><td>Overall</td><td>F1</td><td>Acc.</td><td>FSP</td><td>FCP</td><td>Overall</td><td>FSP</td><td>FCP</td><td>Overall</td></tr><tr><td>LLaVA-1.5-7B</td><td>48.7</td><td>52.6</td><td>50.3</td><td>85.9</td><td>87.4</td><td>39.5</td><td>35.5</td><td>37.5</td><td>32.5</td><td>33.8</td><td>33.8</td></tr><tr><td>+S2</td><td>53.0</td><td>59.1</td><td>55.5</td><td>87.1</td><td>87.9</td><td>49.8</td><td>38.3</td><td>44.0</td><td>40.5</td><td>36.5</td><td>38.5</td></tr><tr><td>+ViCrop</td><td>53.9</td><td>50.0</td><td>52.4</td><td>88.0</td><td>88.6</td><td>60.8</td><td>34.8</td><td>47.8</td><td>38.5</td><td>33.8</td><td>36.1</td></tr><tr><td>+SD-RPN(Ours)</td><td>70.4</td><td>71.1</td><td>70.7↑20.4</td><td>87.2↑1.4</td><td>87.9↑0.5</td><td>59.0</td><td>35.5</td><td>47.3↑9.8</td><td>48.8</td><td>34.5</td><td>41.6↑7.8</td></tr><tr><td>LLaVA-1.5-13B</td><td>47.0</td><td>56.6</td><td>50.8</td><td>85.9</td><td>87.1</td><td>41.5</td><td>44.5</td><td>43.0</td><td>36.6</td><td>38.5</td><td>36.6</td></tr><tr><td>+S2</td><td>43.5</td><td>59.2</td><td>49.7</td><td>87.3</td><td>88.1</td><td>51.8</td><td>46.5</td><td>49.1</td><td>41.3</td><td>44.5</td><td>42.9</td></tr><tr><td>+ViCrop</td><td>47.8</td><td>57.9</td><td>51.8</td><td>88.0</td><td>88.7</td><td>66.3</td><td>40.3</td><td>53.3</td><td>44.5</td><td>37.0</td><td>40.6</td></tr><tr><td>+SD-RPN(Ours)</td><td>60.9</td><td>65.8</td><td>62.8↑12.0</td><td>87.9↑2.0</td><td>88.6↑1.5</td><td>58.8</td><td>47.0</td><td>52.9↑9.9</td><td>51.8</td><td>39.8</td><td>45.8↑9.2</td></tr><tr><td>DeepSeek-VL-1.3B</td><td>34.8</td><td>55.3</td><td>42.9</td><td>86.1</td><td>87.1</td><td>40.5</td><td>33.0</td><td>36.8</td><td>30.1</td><td>29.3</td><td>30.1</td></tr><tr><td>+SD-RPN(Ours)</td><td>55.7</td><td>59.2</td><td>57.1↑14.2</td><td>87.5↑1.4</td><td>88.3↑1.2</td><td>56.8</td><td>34.0</td><td>45.4↑8.6</td><td>44.8</td><td>32.5</td><td>38.6↑8.5</td></tr><tr><td>DeepSeek-VL-7B</td><td>37.4</td><td>50.0</td><td>42.4</td><td>86.0</td><td>87.1</td><td>47.5</td><td>41.5</td><td>44.5</td><td>37.3</td><td>41.3</td><td>39.3</td></tr><tr><td>+SD-RPN(Ours)</td><td>52.2</td><td>55.3</td><td>53.4↑11.0</td><td>88.2↑2.2</td><td>89.0↑1.9</td><td>58.8</td><td>39.8</td><td>49.3↑4.8</td><td>52.3</td><td>40.0</td><td>46.1↑6.8</td></tr><tr><td>Qwen2.5-VL-3B</td><td>81.7</td><td>60.5</td><td>73.3</td><td>87.8</td><td>88.6</td><td>80.5</td><td>52.0</td><td>66.3</td><td>70.0</td><td>47.8</td><td>58.9</td></tr><tr><td>+SD-RPN(Ours)</td><td>91.3</td><td>65.8</td><td>81.2↑7.9</td><td>88.9↑1.1</td><td>89.3↑0.7</td><td>88.0</td><td>58.3</td><td>73.1↑6.8</td><td>79.5</td><td>51.0</td><td>65.3↑6.4</td></tr><tr><td>Qwen2.5-VL-7B</td><td>80.0</td><td>75.0</td><td>78.0</td><td>86.7</td><td>87.9</td><td>81.8</td><td>62.8</td><td>72.5</td><td>72.8</td><td>54.5</td><td>63.6</td></tr><tr><td>+Thyme†</td><td>83.5</td><td>80.3</td><td>82.2</td><td>-</td><td>-</td><td>91.0</td><td>63.0</td><td>77.0</td><td>86.5</td><td>57.5</td><td>72.0</td></tr><tr><td>+DeepEyes†</td><td>91.3</td><td>88.2</td><td>90.1</td><td>87.7</td><td>-</td><td>91.3</td><td>59.0</td><td>75.1</td><td>86.6</td><td>58.5</td><td>72.6</td></tr><tr><td>+SD-RPN(Ours)</td><td>96.5</td><td>79.0</td><td>89.5↑11.5</td><td>88.4↑1.7</td><td>88.9↑1.0</td><td>92.5</td><td>64.5</td><td>78.5↑6.0</td><td>86.5</td><td>60.5</td><td>73.5↑9.9</td></tr></table>


where ◦ represents a masking operation that uses B to select the foreground features from the encoder’s output. Box Upscaling can achieve a higher effective resolution for small, individual regions and Masked Upscaling better preserves the global spatial and positional relationships between all foreground elements, which is crucial for structured data. In the second stage, the high-resolution visual tokens are inserted into the sequence immediately following the original visual tokens. The LLM then performs auto-regressive decoding on this augmented context to generate the final answer. We present a empirical comparison of these two upscaling strategies in our ablation study. 


## 4 EXPERIMENTS

## 4.1 EXPERIMENT CONFIGURATIONS.

Benchmark Settings. To ensure a comprehensive evaluation, we test our SD-RPN framework across a range of benchmarks, following the protocols of established works (Shi et al., 2025a; Zhang et al., 2025a; Zheng et al., 2025). Our results are presented in two parts to clearly delineate performance across different domains. The first part focuses on the Document & OCR category, which assesses performance on text-rich images, and includes five benchmarks: DocVQA (Mathew et al., 2020), ChartQA (Masry et al., 2022), OCRBench (Liu et al., 2024b), InfoVQA (Mathew et al., 2022), and TextVQA (Singh et al., 2019). The second part focuses on Vision-Centric and High-Resolution tasks, including V-Star Bench (V*) (Wu & Xie, 2023), POPE (Li et al., 2023), and HR-Bench (Wang et al., 2024b). Notably, for TextVQA, we do not provide the model with external OCR annotations, a strategy adopted from ViCrop (Zhang et al., 2025a) to ensure a fair evaluation of the MLLM’s intrinsic perceptual capabilities. 

Baselines and Comparison Methods. To evaluate SD-RPN, we integrate it into three prominent MLLM families. Our primary experiments are conducted on the widely-used LLaVA-1.5 (Liu et al., 2023a) architecture, across both its 7B and 13B scales. To demonstrate the generalizability of our approach, we also apply it to the more recent DeepSeek-VL (Lu et al., 2024) and Qwen2.5-VL (Bai et al., 2025) models. We compare our method against several representative techniques for highresolution visual processing. For the LLaVA-1.5 models, we benchmark against $\dot { \mathrm { S } } ^ { 2 }$ (Shi et al., 2024), a full-tuning method, and ViCrop (Zhang et al., 2025a), a training-free cropping baseline. For the Qwen2.5-VL model, we additionally compare against DeepEyes (Zheng et al., 2025) and Thyme (Zhang et al., 2025b), which are recent reinforcement learning-based method. To ensure a rigorous and fair comparison for full reproducibility, we have re-implemented all competing methods within the unified lmms-eval evaluation library (Zhang et al., 2024a), unless noted otherwise. 

Implementation Details. To train our proposed SD-RPN, we generate pseudo-labels from a combined dataset of GQA (Hudson & Manning, 2019) and OCR-VQA (Mishra et al., 2019), which provides supervision for natural scenes and text-rich images, respectively. These pseudo-labels are derived from the models’ internal response-to-image attention maps. Inspired by recent analyses of MLLM attention mechanisms (Kang et al., 2025b), we select attention maps from the middle layers for this process. During the pseudo-label generation pipeline, the relative attention thresholds for defining the foreground $( S _ { f g } )$ and background $( S _ { b g } )$ sets are empirically set to 0.2 and 0.1, respectively. For experiments involving Qwen2.5-VL, which natively supports dynamic resolutions, we standardize the input for consistency. On most benchmarks, we set the number of visual tokens to 576 to align with the LLaVA and DeepSeek-VL baselines. For high-resolution benchmarks, we increase this to a maximum of 4096 visual tokens to effectively process the detailed inputs. Further details on the layer selection rationale and hyper-parameters can be found in Appendix A. 

## 4.2 MAIN RESULTS

We present our main results in Tab. 1 for text-rich benchmarks and Tab. 2 for vision-centric benchmarks. Our evaluation assesses both performance and efficiency. 

Performance. Integrating our SD-RPN framework yields substantial and consistent performance gains across all evaluated MLLM families. On the text-rich benchmarks (Tab. 1), our method boosts the average score by approximately 7% over the LLaVA-1.5 and DeepSeek-VL baselines and 3% over the more powerful Qwen2.5-VL baseline. The benefits are also evident on vision-centriand high-resolution tasks (Tab. 2), where our approach achieves average improvements of over 10% on V* Bench and over 6% on HR-Bench. Notably, on high-resolution benchmarks, SD-RPN achieves performance comparable or superior to DeepEyes which relies visual Chain-of-Thought (CoT) that incurs substantially greater computational cost. To ensure a fair comparison on V-Star Bench, we note that the image cropping strategy employed by ViCrop was disabled during our evaluation. Beyond these quantitative results, Fig. 5 offers a qualitative comparison with the LLaVA-1.5 and Qwen2.5-VL baselines. Additional visual examples are provided in Appendix E. 

Efficiency. The two-stage inference process used by our method and ViCrop inherently involves a trade-off, reducing throughput in exchange for higher accuracy. This additional latency stems from three primary sources: the initial RoI prediction stage, the feature extraction for the high-resolution RoI crops, and the increased number of visual tokens processed by the LLM in the final generation stage. However, our SD-RPN is architected for superior efficiency in the critical RoI prediction step. By leveraging only a subset of the MLLM’s layers in a single forward pass, our method is significantly faster than competing two-stage approaches. For example, when integrated into the LLaVA-1.5-7B model, our RoI prediction stage is 1.5× faster than that of ViCrop, demonstrating a more favorable balance between performance and computational cost. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/e11b9957-d09a-473b-8ecb-edef7d68b919/943e1087b7a4d23c8ee04723b0e4af5d7de6c98d272f0c9f708db54cdd689134.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/e11b9957-d09a-473b-8ecb-edef7d68b919/122616eeb20787fbd8e3ebad8abc6be889a4c4549a22f47f0cfd332b5334029a.jpg)



Figure 5: Qualitative comparisons on TextVQA and V-Star benchmarks. These examples show challenging cases where the baseline model fails while our SD-RPN succeeds. For each question, our model generates a dense RoI Heatmap (center) to localize the relevant area. The Mixed Image (right) shows the final high-confidence region identified by SD-RPN, which is used for fine-grained perception to find the correct answer.


## 4.3 ABLATION STUDY


Table 3: Ablation study on the key components of our method.


<table><tr><td>(#)</td><td>Setting</td><td>Throughput</td><td>OCRBench</td><td>TextVQA</td><td>POPE</td><td>V*</td><td>Ave.</td></tr><tr><td>(0)</td><td>LLaVA-1.5-7B (Baseline)</td><td><eq>1.0 \times</eq></td><td>31.4</td><td>46.1</td><td>85.9</td><td>50.3</td><td>53.4</td></tr><tr><td>(1)</td><td>Response-to-Image Attention</td><td><eq>0.42 \times</eq></td><td><eq>32.8 \uparrow 2.4</eq></td><td><eq>53.0 \uparrow 6.9</eq></td><td><eq>84.3 \downarrow 1.6</eq></td><td><eq>57.6 \uparrow 7.6</eq></td><td><eq>56.9 \uparrow 3.8</eq></td></tr><tr><td>(2)</td><td>Attention Prediction</td><td><eq>0.58 \times</eq></td><td><eq>32.2 \uparrow 0.8</eq></td><td><eq>56.8 \uparrow 10.7</eq></td><td><eq>85.7 \downarrow 0.2</eq></td><td><eq>61.3 \uparrow 10.0</eq></td><td><eq>59.0 \uparrow 5.3</eq></td></tr><tr><td>(3)</td><td>+Label Assignment</td><td><eq>0.55 \times</eq></td><td><eq>33.9 \uparrow 2.5</eq></td><td><eq>57.3 \uparrow 11.2</eq></td><td><eq>85.6 \downarrow 0.3</eq></td><td><eq>68.6 \uparrow 18.3</eq></td><td><eq>61.4 \uparrow 7.9</eq></td></tr><tr><td>(4)</td><td>+Remove Sink Tokens</td><td><eq>0.55 \times</eq></td><td><eq>36.2 \uparrow 4.8</eq></td><td><eq>57.9 \uparrow 11.8</eq></td><td><eq>86.8 \uparrow 0.9</eq></td><td><eq>68.6 \uparrow 18.3</eq></td><td><eq>62.4 \uparrow 9.0</eq></td></tr><tr><td>(5)</td><td>+Masked Upscaling</td><td><eq>\mathbf{0.62} \times</eq></td><td><eq>\mathbf{37.0} \uparrow 5.6</eq></td><td><eq>\mathbf{58.7} \uparrow 12.6</eq></td><td><eq>\mathbf{87.1} \uparrow 1.2</eq></td><td><eq>67.5 \uparrow 17.2</eq></td><td><eq>\mathbf{62.6} \uparrow 9.2</eq></td></tr></table>


In this section, we first present ablations for our core design choices, including the pseudo-label generation process, backbone layer depth, and data efficiency. These experiments were conducted using default thresholds $( \tau _ { f g } = 0 . 2 , \tau _ { b g } = 0 . 1 )$ . We then provide a detailed ablation study on these $\tau _ { f g }$ and $\tau _ { b g }$ thresholds to determine their optimal values.<sup>1</sup> 


On Pseudo-Label Gerneration & SD-RPN. Tab. 3 presents a detailed ablation study dissecting the core components of our proposed framework. We begin by evaluating two baseline strategies for leveraging attention. Box upscaling is adopted as the upscaling setting except additional note. In (#1), we use the raw Response-to-Image Attention map directly for RoI identification. In (#2), our RPN is trained to regress these attention scores via a Mean Squared Error (MSE) loss. While both methods improve upon the baseline, their average gains of 3.8% and 5.3% are limited, confirming our hypothesis that using noisy, unprocessed attention maps for direct supervision is suboptimal. The introduction of Label Assignment strategy (#3), which creates high-confidence labels and ignores ambiguous regions, yields a significant performance jump to a $7 . 9 \%$ average improvement. This is further enhanced by Remove Sink Tokens step (#4), which denoises the attention map to achieve a 9.0% average gain. These results validate that our proposed denoising pipeline is crucial for generating high-quality supervision and enabling precise RoI prediction. Aligning with previous analysis in Sec. 3.4, masked upscaling achieves superior performance on OCRBench and TextVQA. Given its better throughput (0.62x vs. 0.55x), we adopt it as the default setting. 

On Backbone Layers. In Tab. 4a, we conduct an ablation study on the number of frozen backbone layers (B) that the RPN is built upon, keeping the number of trainable RPN layers fixed. We observe a clear trend in performance: as the backbone depth increases, the average performance steadily improves, reaching its peak with the B15R3 configuration, which achieves a 9.2% gain over the baseline. Beyond this point, performance begins to decline. The relationship between backbone depth and efficiency is more nuanced. While the inference cost of the RoI prediction stage is positively correlated with the number of backbone layers, the overall throughput is not monotonic. This is because a more precise RoI prediction, often produced by a deeper backbone, can reduce the number of irrelevant or noisy visual tokens that are passed to the second, more costly generation stage. This interplay between the increasing cost of the first stage and the potentially decreasing cost of the second stage explains the fluctuating throughput values. 


Table 4: (a) Varying the number of frozen backbone layers (B) while keeping the RPN layers fixed. (b) Varying the number of training data samples. The † symbol indicates that pseudo-labels were generated using ground-truth responses from the LLaVA supervised fine-tuning dataset. All scores are accuracy (%) except for POPE, which reports the F1-score.


<table><tr><td>Setting</td><td>Through-put</td><td>OCR-Bench</td><td>Text-VQA</td><td>POPE</td><td>V*</td><td>Ave.</td></tr><tr><td>Baseline</td><td>1.0×</td><td>31.4</td><td>46.1</td><td>85.9</td><td>50.3</td><td>53.4</td></tr><tr><td>B3R3</td><td><eq>\mathbf{0.71} \times</eq></td><td><eq>33.9_{\uparrow 2.5}</eq></td><td><eq>51.0_{\uparrow 4.9}</eq></td><td><eq>87.0_{\uparrow 1.1}</eq></td><td><eq>53.9_{\uparrow 3.6}</eq></td><td><eq>56.5_{\uparrow 3.1}</eq></td></tr><tr><td>B6R3</td><td><eq>0.63 \times</eq></td><td><eq>34.4_{\uparrow 3.0}</eq></td><td><eq>53.5_{\uparrow 7.4}</eq></td><td><eq>\mathbf{87.7}_{\uparrow 1.8}</eq></td><td><eq>57.1_{\uparrow 6.8}</eq></td><td><eq>58.2_{\uparrow 4.8}</eq></td></tr><tr><td>B9R3</td><td><eq>0.66 \times</eq></td><td><eq>35.8_{\uparrow 4.4}</eq></td><td><eq>56.3_{\uparrow 10.2}</eq></td><td><eq>87.1_{\uparrow 1.2}</eq></td><td><eq>59.7_{\uparrow 9.4}</eq></td><td><eq>59.7_{\uparrow 6.3}</eq></td></tr><tr><td>B12R3</td><td><eq>0.60 \times</eq></td><td><eq>35.5_{\uparrow 4.1}</eq></td><td><eq>57.8_{\uparrow 11.7}</eq></td><td><eq>87.4_{\uparrow 1.5}</eq></td><td><eq>67.0_{\uparrow 16.7}</eq></td><td><eq>61.9_{\uparrow 8.5}</eq></td></tr><tr><td>B15R3</td><td><eq>0.62 \times</eq></td><td><eq>\mathbf{37.0}_{\uparrow 5.6}</eq></td><td><eq>\mathbf{58.7}_{\uparrow 12.6}</eq></td><td><eq>87.1_{\uparrow 1.2}</eq></td><td><eq>\mathbf{67.5}_{\uparrow 17.2}</eq></td><td><eq>\mathbf{62.6}_{\uparrow 9.2}</eq></td></tr><tr><td>B18R3</td><td><eq>0.52 \times</eq></td><td><eq>35.5_{\uparrow 4.1}</eq></td><td><eq>57.8_{\uparrow 11.7}</eq></td><td><eq>87.3_{\uparrow 1.4}</eq></td><td><eq>66.0_{\uparrow 15.7}</eq></td><td><eq>61.7_{\uparrow 8.3}</eq></td></tr></table>


(a)


<table><tr><td>Setting</td><td>OCR-Bench</td><td>Text-VQA</td><td>POPE</td><td>V*</td><td>Ave.</td></tr><tr><td>Baseline</td><td>31.4</td><td>46.1</td><td>85.9</td><td>50.3</td><td>53.4</td></tr><tr><td>10K</td><td><eq>35.2^{\uparrow 3.8}</eq></td><td><eq>58.3^{\uparrow 12.2}</eq></td><td><eq>\mathbf{87.6}^{\uparrow 1.7}</eq></td><td><eq>61.3^{\uparrow 11.0}</eq></td><td><eq>60.6^{\uparrow 7.2}</eq></td></tr><tr><td>25K</td><td><eq>35.8^{\uparrow 4.4}</eq></td><td><eq>58.5^{\uparrow 12.4}</eq></td><td><eq>87.1^{\uparrow 1.2}</eq></td><td><eq>59.7^{\uparrow 9.4}</eq></td><td><eq>60.3^{\uparrow 6.9}</eq></td></tr><tr><td>50K</td><td><eq>\mathbf{37.5}^{\uparrow 6.1}</eq></td><td><eq>58.3^{\uparrow 12.2}</eq></td><td><eq>86.9^{\uparrow 1.0}</eq></td><td><eq>67.0^{\uparrow 16.7}</eq></td><td><eq>62.4^{\uparrow 9.0}</eq></td></tr><tr><td>100K</td><td><eq>36.5^{\uparrow 5.1}</eq></td><td><eq>58.3^{\uparrow 12.2}</eq></td><td><eq>86.8^{\uparrow 0.9}</eq></td><td><eq>66.0^{\uparrow 15.7}</eq></td><td><eq>61.9^{\uparrow 8.5}</eq></td></tr><tr><td>152K</td><td><eq>37.0^{\uparrow 5.6}</eq></td><td><eq>\mathbf{58.7}^{\uparrow 12.6}</eq></td><td><eq>87.1^{\uparrow 1.2}</eq></td><td><eq>\mathbf{67.5}^{\uparrow 17.2}</eq></td><td><eq>\mathbf{62.6}^{\uparrow 9.2}</eq></td></tr><tr><td><eq>152\mathrm {K}^{\dagger}</eq></td><td><eq>35.2^{\uparrow 3.8}</eq></td><td><eq>56.9^{\uparrow 10.8}</eq></td><td><eq>86.6^{\uparrow 0.7}</eq></td><td><eq>63.9^{\uparrow 13.6}</eq></td><td><eq>60.7^{\uparrow 7.3}</eq></td></tr></table>


(b)


On Data Efficency. We conduct ablation studies on the size of the training set, with results presented in Tab. 4b. Performance generally improves with an increased number of training samples. Remarkably, even when trained on only 10K samples, the framework achieves substantial gains over the baseline (e.g., over 10% on improvement on both TextVQA and V-Star). Further increases in data size continue to yield performance gains. The peak performance is ultimately achieved when training on the full combined dataset of 152K samples. The final row of Tab. 4b presents a crucial comparison where we generate pseudo-labels using ground-truth responses from the LLaVA supervised fine-tuning (SFT) dataset. Counterintuitively, this approach yields a notably lower average performance of 60.7% compared to the 62.6% achieved by our standard self-distillation method. This suggests that the attention maps produced by the model’s own generated responses provide a more effective and internally consistent supervision signal for distilling localization knowledge. 


Table 5: Ablation study on pseudo-label thresholds $\tau _ { f g }$ and $\tau _ { b g }$


<table><tr><td><eq>\tau_{fg}</eq></td><td><eq>\tau_{bg}</eq></td><td>OCR-Bench</td><td>Text-VQA</td><td>POPE</td><td>V*</td><td>Ave.</td></tr><tr><td colspan="2">Baseline</td><td>31.4</td><td>46.1</td><td>85.9</td><td>50.3</td><td>53.4</td></tr><tr><td>0.10</td><td>0.10</td><td>35.1</td><td>57.4</td><td>87.4</td><td>65.4</td><td>61.3</td></tr><tr><td>0.15</td><td>0.10</td><td>36.3</td><td>57.9</td><td>87.2</td><td>67.5</td><td>62.2</td></tr><tr><td>0.20</td><td>0.10</td><td>37.0</td><td>58.7</td><td>87.1</td><td>67.5</td><td>62.6</td></tr><tr><td>0.25</td><td>0.10</td><td>37.0</td><td>58.6</td><td>86.7</td><td>68.1</td><td>62.6</td></tr><tr><td>0.30</td><td>0.10</td><td>36.6</td><td>58.1</td><td>86.9</td><td>64.9</td><td>61.6</td></tr><tr><td>0.20</td><td>0.075</td><td>36.7</td><td>58.4</td><td>86.9</td><td>70.7</td><td>63.2</td></tr><tr><td>0.20</td><td>0.05</td><td>37.3</td><td>58.8</td><td>87.2</td><td>70.7</td><td>63.5</td></tr><tr><td>0.20</td><td>0.03</td><td>37.8</td><td>58.7</td><td>87.6</td><td>68.6</td><td>63.2</td></tr></table>

On Foreground and Background Threshold. Tab. 5 ablates the pseudo-label thresholds. We find that setting $\tau _ { f g } = \tau _ { b g } = 0 . 1 0$ (a simple binary classification) performs poorly (61.3% avg.), as it suffers from ambiguous tokens. By optimizing the thresholds to $\tau _ { f g } = 0 . 2 0$ and $\tau _ { b g } = 0 . 0 5$ , we establish a new, stronger default. This setting achieves a 63.5% average performance, surpassing our original default (62.6%) and the baseline (53.4%) by 0.9% and 10.1%, respectively. 

## 5 CONCLUSION

We proposed SD-RPN, a self-distilled region proposal framework that efficiently exploits the intrinsic localization signals of MLLMs to identify Regions of Interest without external annotations or auto-regressive decoding. By attaching a RPN to frozen backbones and training it with denoised pseudo-labels, our method achieves consistent improvements in fine-grained perception while maintaining strong efficiency and generalization. Extensive experiments confirm its advantage over both full-image scaling and training-free heuristics, and ablations highlight its robustness and data efficiency. Our work establishes a principled direction for scalable high-resolution perception in MLLMs, and opens avenues toward adaptive token allocation and broader multimodal applications such as video and document understanding. 

## 6 ACKNOWLEDGEMENTS

This work was supported in part by the Start-up Grant (No. 9610680) of the City University of Hong Kong, Young Scientist Fund (No. 62406265) of NSFC, and the Australian Research Council under Projects DP240101848 and FT230100549. 

## REFERENCES



Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016. 





Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966, 2023. 





Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 





Yuxuan Cai, Jiangning Zhang, Haoyang He, Xinwei He, Ao Tong, Zhenye Gan, Chengjie Wang, Zhucun Xue, Yong Liu, and Xiang Bai. Llava-kd: A framework of distilling multimodal large language models. arXiv preprint arXiv:2410.16236, 2024. 





Junbum Cha, Wooyoung Kang, Jonghwan Mun, and Byungseok Roh. Honeybee: Locality-enhanced projector for multimodal llm. In CVPR, 2024. 





Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. Science China Information Sciences, 2024a. 





Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 24185–24198, 2024b. 





Wenliang Dai, Junnan Li, Dongxu Li, Anthony Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. NeurIPS, 2023. 





Timothee Darcet, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. Vision transformers need´ registers. arXiv preprint arXiv:2309.16588, 2023. 





Xiaoyi Dong, Jianmin Bao, Yinglin Zheng, Ting Zhang, Dongdong Chen, Hao Yang, Ming Zeng, Weiming Zhang, Lu Yuan, Dong Chen, et al. Maskclip: Masked self-distillation advances contrastive language-image pretraining. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10995–11005, 2023. 





Chunjiang Ge, Sijie Cheng, Ziming Wang, Jiale Yuan, Yuan Gao, Jun Song, Shiji Song, Gao Huang, and Bo Zheng. Convllava: Hierarchical backbones as visual encoder for large multimodal models. arXiv preprint arXiv:2405.15738, 2024. 





Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015. 





Xiaojun Hou, Jiazheng Xing, Yijie Qian, Yaowei Guo, Shuo Xin, Junhao Chen, Kai Tang, Mengmeng Wang, Zhengkai Jiang, Liang Liu, et al. Sdstrack: Self-distillation symmetric adapter learning for multi-modal visual object tracking. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 26551–26561, 2024. 





Mingxin Huang, Yuliang Liu, Dingkang Liang, Lianwen Jin, and Xiang Bai. Mini-monkey: Alleviating the semantic sawtooth effect for lightweight mllms via complementary image pyramid. arXiv preprint arXiv:2408.02034, 2024. 





Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In CVPR, 2019. 





Seil Kang, Jinyeong Kim, Junhyeok Kim, and Seong Jae Hwang. See what you are told: Visual attention sink in large multimodal models. In ICLR, 2025a. 





Seil Kang, Jinyeong Kim, Junhyeok Kim, and Seong Jae Hwang. Your large vision-language model only needs a few attention heads for visual grounding. In CVPR, 2025b. 





Sanghwan Kim, Rui Xiao, Mariana-Iuliana Georgescu, Stephan Alaniz, and Zeynep Akata. Cosmos: Cross-modality self-distillation for vision language pre-training. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pp. 14690–14700, 2025. 





Jun Kong, Jin Wang, Liang-Chih Yu, and Xuejie Zhang. Multimodality self-distillation for fast inference of vision and language pretrained models. IEEE Transactions on Multimedia, 26:8928– 8940, 2024. 





Wentong Li, Yuqian Yuan, Jian Liu, Dongqi Tang, Song Wang, Jie Qin, Jianke Zhu, and Lei Zhang. Tokenpacker: Efficient visual projector for multimodal llm. International Journal of Computer Vision, 2025. 





Yanwei Li, Yuechen Zhang, Chengyao Wang, Zhisheng Zhong, Yixin Chen, Ruihang Chu, Shaoteng Liu, and Jiaya Jia. Mini-gemini: Mining the potential of multi-modality vision language models. arXiv preprint arXiv:2403.18814, 2024a. 





Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. arXiv preprint arXiv:2305.10355, 2023. 





Zhang Li, Biao Yang, Qiang Liu, Zhiyin Ma, Shuo Zhang, Jingxu Yang, Yabo Sun, Yuliang Liu, and Xiang Bai. Monkey: Image resolution and text label are important things for large multi-modal models. In CVPR, 2024b. 





Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning, 2023a. 





Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning, 2023b. 





Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llavanext: Improved reasoning, ocr, and world knowledge, 2024a. 





Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, Cheng-Lin Liu, Lianwen Jin, and Xiang Bai. Ocrbench: on the hidden mystery of ocr in large multimodal models. Science China Information Sciences, 2024b. 





Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Hao Yang, et al. Deepseek-vl: towards real-world vision-language understanding. arXiv preprint arXiv:2403.05525, 2024. 





Gen Luo, Yiyi Zhou, Yuxin Zhang, Xiawu Zheng, Xiaoshuai Sun, and Rongrong Ji. Feast your eyes: Mixture-of-resolution adaptation for multimodal large language models. arXiv preprint arXiv:2403.03003, 2024. 





Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. arXiv preprint arXiv:2203.10244, 2022. 





Minesh Mathew, Dimosthenis Karatzas, R Manmatha, and CV Jawahar. Docvqa: A dataset for vqa on document images. corr abs/2007.00398 (2020). arXiv preprint arXiv:2007.00398, 2020. 





Minesh Mathew, Viraj Bagal, Ruben Tito, Dimosthenis Karatzas, Ernest Valveny, and CV Jawahar.` Infographicvqa. In WACV, 2022. 





Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and Anirban Chakraborty. Ocr-vqa: Visual question answering by reading text in images. In 2019 international conference on document analysis and recognition (ICDAR). IEEE, 2019. 





Maxime Oquab, Timothee Darcet, Th ´ eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov,´ Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 





Yi Peng, Peiyu Wang, Xiaokun Wang, Yichen Wei, Jiangbo Pei, Weijie Qiu, Ai Jian, Yunzhuo Hao, Jiachun Pan, Tianyidan Xie, et al. Skywork r1v: Pioneering multimodal reasoning with chain-ofthought. arXiv preprint arXiv:2504.05599, 2025. 





Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 





Hao Shao, Shengju Qian, Han Xiao, Guanglu Song, Zhuofan Zong, Letian Wang, Yu Liu, and Hongsheng Li. Visual cot: Advancing multi-modal language models with a comprehensive dataset and benchmark for chain-of-thought reasoning. NeurIPS, 2024. 





Zhenwei Shao, Mingyang Wang, Zhou Yu, Wenwen Pan, Yan Yang, Tao Wei, Hongyuan Zhang, Ning Mao, Wei Chen, and Jun Yu. Growing a twig to accelerate large vision-language models. In ICCV, 2025. 





Baifeng Shi, Ziyang Wu, Maolin Mao, Xin Wang, and Trevor Darrell. When do we not need larger vision models? In ECCV, 2024. 





Baifeng Shi, Boyi Li, Han Cai, Yao Lu, Sifei Liu, Marco Pavone, Jan Kautz, Song Han, Trevor Darrell, Pavlo Molchanov, et al. Scaling vision pre-training to 4k resolution. In CVPR, 2025a. 





Yuheng Shi, Minjing Dong, and Chang Xu. Harnessing vision foundation models for highperformance, training-free open vocabulary segmentation. In ICCV, 2025b. 





Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In CVPR, 2019. 





Zhi Tian, Chunhua Shen, Xinlong Wang, and Hao Chen. Boxinst: High-performance instance segmentation with box annotations. In CVPR, 2021. 





Peter Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Adithya Jairam Vedagiri IYER, Sai Charitha Akula, Shusheng Yang, Jihan Yang, Manoj Middepogu, Ziteng Wang, et al. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. NeurIPS, 2024a. 





Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. Eyes wide shut? exploring the visual shortcomings of multimodal llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9568–9578, 2024b. 





Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. 





Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024a. 





Wenbin Wang, Liang Ding, Minyan Zeng, Xiabin Zhou, Li Shen, Yong Luo, and Dacheng Tao. Divide, conquer and combine: A training-free framework for high-resolution image perception in multimodal large language models. arXiv preprint, 2024b. 





Xinlong Wang, Zhiding Yu, Shalini De Mello, Jan Kautz, Anima Anandkumar, Chunhua Shen, and Jose M Alvarez. Freesolo: Learning to segment objects without annotations. In CVPR, 2022. 





Penghao Wu and Saining Xie. V*: Guided visual search as a core mechanism in multimodal llms. arXiv preprint arXiv:2312.14135, 2023. 





Runpeng Yu, Xinyin Ma, and Xinchao Wang. Introducing visual perception token into multimodal large language model. In ICCV, 2025. 





Biao Zhang and Rico Sennrich. Root mean square layer normalization. NeurIPS, 2019. 





Jiarui Zhang, Mahyar Khayatkhoei, Prateek Chhikara, and Filip Ilievski. Mllms know where to look: Training-free perception of small visual details with multimodal llms. In ICLR, 2025a. 





Kaichen Zhang, Bo Li, Peiyuan Zhang, Fanyi Pu, Joshua Adrian Cahyono, Kairui Hu, Shuai Liu, Yuanhan Zhang, Jingkang Yang, Chunyuan Li, and Ziwei Liu. Lmms-eval: Reality check on the evaluation of large multimodal models, 2024a. URL https://arxiv.org/abs/2407. 12772. 





Linfeng Zhang, Jiebo Song, Anni Gao, Jingwei Chen, Chenglong Bao, and Kaisheng Ma. Be your own teacher: Improve the performance of convolutional neural networks via self distillation. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 3713–3722, 2019. 





Yi-Fan Zhang, Xingyu Lu, Shukang Yin, Chaoyou Fu, Wei Chen, Xiao Hu, Bin Wen, Kaiyu Jiang, Changyi Liu, Tianke Zhang, et al. Thyme: Think beyond images. arXiv preprint arXiv:2508.11630, 2025b. 





Yipeng Zhang, Yifan Liu, Zonghao Guo, Yidan Zhang, Xuesong Yang, Chi Chen, Jun Song, Bo Zheng, Yuan Yao, Zhiyuan Liu, et al. Llava-uhd v2: an mllm integrating high-resolution feature pyramid via hierarchical window transformer. arXiv preprint arXiv:2412.13871, 2024b. 





Xiangyu Zhao, Xiangtai Li, Haodong Duan, Haian Huang, Yining Li, Kai Chen, and Hua Yang. Mg-llava: Towards multi-granularity visual instruction tuning. arXiv preprint arXiv:2406.17770, 2024. 





Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. NeurIPS, 2023. 





Ziwei Zheng, Michael Yang, Jack Hong, Chenxiao Zhao, Guohai Xu, Le Yang, Chao Shen, and Xing Yu. Deepeyes: Incentivizing” thinking with images” via reinforcement learning. arXiv preprint arXiv:2505.14362, 2025. 





Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, Zhangwei Gao, Erfei Cui, Xuehui Wang, Yue Cao, Yangzhou Liu, Xingguang Wei, Hongjie Zhang, Haomin Wang, Weiye Xu, Hao Li, Jiahao Wang, Nianchen Deng, Songze Li, Yinan He, Tan Jiang, Jiapeng Luo, Yi Wang, Conghui He, Botian Shi, Xingcheng Zhang, Wenqi Shao, Junjun He, Yingtong Xiong, Wenwen Qu, Peng Sun, Penglong Jiao, Han Lv, Lijun Wu, Kaipeng Zhang, Huipeng Deng, Jiaye Ge, Kai Chen, Limin Wang, Min Dou, Lewei Lu, Xizhou Zhu, Tong Lu, Dahua Lin, Yu Qiao, Jifeng Dai, and Wenhai Wang. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models, 2025. 



## A MORE IMPLEMENTATION DETAILS.

Training Setting. The hyperparameters used to train our RPN were adapted from TwigVLM (Shao et al., 2025) and are detailed in Table 6. All models were trained on a server using four NVIDIA A6000 GPUs. Our training dataset consists of 152K samples, with 72K sourced from the GQA dataset (primarily natural images) and 80K from the OCR-VQA dataset (text-rich images). As a point of reference for computational cost, training the RPN for the LLaVA-1.5-7B model on our full 152K-sample dataset completes in around two hours. 


Table 6: Training hyperparameters for the RPN.


<table><tr><td>Config</td><td>Setting</td></tr><tr><td>optimizer</td><td>AdamW</td></tr><tr><td>weight decay</td><td>0.</td></tr><tr><td>optimizer momentum</td><td><eq>\beta_{1}, \beta_{2} = 0.9, 0.98</eq></td></tr><tr><td>batch size</td><td>128</td></tr><tr><td>learning rate schedule</td><td>cosine decay</td></tr><tr><td>peak learning rate</td><td>5e-5</td></tr><tr><td>warm-up strategy</td><td>linear</td></tr><tr><td>warm-up ratio</td><td>0.03</td></tr><tr><td>training epochs</td><td>1</td></tr></table>


Table 7: Model-specific layer configurations. ”Selected Attention Layers” refers to the layers from which attention maps were extracted for pseudo-label generation. ”Frozen Backbone Layers (B)” refers to the number of initial layers kept frozen, upon which the trainable RPN is built.


<table><tr><td>Model</td><td>Selected Attention Layers</td><td>Frozen Backbone Layers (B)</td></tr><tr><td>LLaVA-1.5-7B</td><td>14</td><td>15</td></tr><tr><td>LLaVA-1.5-13B</td><td>13–16</td><td>15</td></tr><tr><td>DeepSeek-VL-1.3B</td><td>7–14</td><td>9</td></tr><tr><td>DeepSeek-VL-7B</td><td>9–17</td><td>12</td></tr><tr><td>Qwen2.5-VL-3B</td><td>20–22</td><td>21</td></tr><tr><td>Qwen2.5-VL-7B</td><td>16–23</td><td>16</td></tr></table>

Layer Selection Rationale. The specific layers used for pseudo-label generation and for defining the RPN backbone were empirically chosen for each model architecture, with the final configurations detailed in Table 7. For generating pseudo-labels, our selection of attention maps is inspired by recent analyses (Kang et al., 2025b) which demonstrate that the middle layers of an MLLM contain the most potent visual grounding signals. These intermediate layers represent a ”sweet spot” where visual and textual representations are sufficiently fused for localization, yet before the model’s final layers become overly specialized for abstract reasoning and text generation, potentially weakening the direct visual signal. When a range of layers is specified, the attention maps were averaged to create a more stable and robust signal. The number of frozen backbone layers, B, determines the depth and richness of the features that the trainable RPN operates on. This choice represents a critical trade-off. A deeper backbone (larger B) provides more semantically rich features but increases the computational cost of the RoI prediction stage and can limit the learning capacity of a small RPN. Conversely, a shallower backbone is faster but may not provide features that are sufficiently aligned for accurate RoI prediction. The values in Table 7 were determined through ablation studies (see Table 4a) to find the optimal balance between performance and efficiency for each model. 

Algorithm Details. To provide a concrete and reproducible specification of our training procedure, we detail the complete end-to-end process in Algorithm 1. The algorithm consists of a main training loop and the core helper routines. The main training loop iterates through the dataset, performing two key stages for each sample. First, the ”teacher” stage (lines 3-10) uses the full, frozen MLLM to generate a pseudo-label. This involves auto-regressively generating a response, extracting the raw response-to-image attention map from a pre-selected middle layer, and then refining this map using our RemoveSinkTokens and AssignLabels helper functions. This produces the final denoised pseudo-label, $\bar { \bf M } _ { \mathrm { R o I } }$ . Second, the ”student” stage (lines 11-15) trains the RPN. The RPN performs a forward pass on the original input to predict a dense RoI map $\hat { { \bf M } } _ { \mathrm { R o I } }$ . A masked BCEWithLogits loss is then computed between the predicted map and the pseudo-label target, ensuring that ambiguous regions (where the mask is false) do not contribute to the gradient. Finally, the RPN’s parameters are updated via backpropagation. 

Algorithm 1 Self-Distilled RPN: training and helper routines
Require: Full MLLM L with L layers; frozen depth B; RPN depth R; pseudo-label layer l; dataset D; thresholds ( $\tau_{norm}$ , $\tau_{fg}$ , $\tau_{bg}$ ); optimizer for trainable RPN parameters
Ensure: Trained RPN on top of the frozen backbone
1: Initialization: Initialize RPN layers B+1:B+R from L; freeze layers 1:B; initialize optimizer.
2: for each ( $x_v, x_t$ ) ∈ D do
▷ // Teacher: generate pseudo-labels from the full MLLM
3: $Y_{pred} \leftarrow Decode(L, x_v, x_t)$ ▷ Auto-regressive response
4: $x_t^{full} \leftarrow Concat(x_t, Y_{pred})$ 5: $H_v^l, Q_r^l, K_v^l \leftarrow Features(L, x_v, x_t^{full}, l)$ 6: $A \leftarrow Softmax(Q_r^l(K_v^l)^{\top}/\sqrt{d})$ ▷ Softmax over visual tokens
7: $M_{RoI} \leftarrow MeanAcrossHeadsAndResp(A)$ 8: $M'_{RoI} \leftarrow RemoveSinkTokens(M_{RoI}, H_v^l, \tau_{norm})$ 9: $\bar{M}_{RoI} \leftarrow AssignLabels(M_{RoI}', \tau_{fg}, \tau_{bg})$ ▷ Values in {-1, 0, 1}
10: $mask \leftarrow (\bar{M}_{RoI} \neq -1)$ ▷ // Student: forward through frozen 1:B and trainable B+1:B+R
11: $H_v, H_{RoI} \leftarrow Features(L_{RPN}, x_v, x_t, B+R-1)$ 12: $Q_{RoI} \leftarrow LP_q(Norm(H_{RoI})), K_v \leftarrow LP_k(Norm(H_v))$ 13: $\hat{M}_{RoI} \leftarrow Q_{RoI}K_v^\top$ ▷ Logits for dense RoI map
14: loss ← BCEWithLogits( $\hat{M}_{RoI}[mask], \bar{M}_{RoI}[mask]$ )
15: Update(optimizer, loss)
Helper routines
1: function REMOVESINKTOKENS(M, H_v, $\tau_{norm}$ )
2: return M with entries set to 0 where $\|H_{v,j}\|_2 > \tau_{norm}$ 3: function ASSIGNLABELS(M, $\tau_{fg}, \tau_{bg}$ )
4: $a_{max} \leftarrow max_j M_j$ 5: $S_{fg} \leftarrow \{j : M_j \geq \tau_{fg} a_{max}\}$ 6: $B_{fg} \leftarrow MinEnclosingBox(S_{fg})$ 7: $S_{bg} \leftarrow \{j \notin B_{fg} : M_j \leq \tau_{bg} a_{max}\}$ 8: Build M with $M_j = 1$ for $j \in S_{fg}$ , $M_j = 0$ for $j \in S_{bg}$ , and -1 otherwise
9: return M 


Table 8: Ablation on the region proposal layers.


<table><tr><td>Setting</td><td>OCRBench</td><td>TextVQA</td><td>POPE</td><td>V*</td><td>Ave.</td></tr><tr><td>LLaVA-1.5-7B</td><td>31.4</td><td>46.1</td><td>85.9</td><td>50.3</td><td>53.4</td></tr><tr><td>B15R1</td><td><eq>35.8_{\uparrow 4.4}</eq></td><td><eq>54.7_{\uparrow 8.6}</eq></td><td><eq>87.1_{\uparrow 1.2}</eq></td><td><eq>56.0_{\uparrow 5.7}</eq></td><td><eq>58.4_{\uparrow 5.0}</eq></td></tr><tr><td>B15R2</td><td><eq>35.9_{\uparrow 4.5}</eq></td><td><eq>57.8_{\uparrow 11.7}</eq></td><td><eq>\mathbf{87.2}_{\uparrow 1.3}</eq></td><td><eq>63.4_{\uparrow 13.1}</eq></td><td><eq>61.1_{\uparrow 7.7}</eq></td></tr><tr><td>B15R3</td><td><eq>37.0_{\uparrow 5.6}</eq></td><td><eq>\mathbf{58.7}_{\uparrow 12.6}</eq></td><td><eq>87.1_{\uparrow 1.2}</eq></td><td><eq>\mathbf{67.5}_{\uparrow 17.2}</eq></td><td><eq>\mathbf{62.6}_{\uparrow 9.2}</eq></td></tr><tr><td>B15R4</td><td><eq>\mathbf{37.1}_{\uparrow 5.7}</eq></td><td><eq>58.4_{\uparrow 12.3}</eq></td><td><eq>87.1_{\uparrow 1.2}</eq></td><td><eq>\mathbf{67.5}_{\uparrow 17.2}</eq></td><td><eq>62.5_{\uparrow 9.1}</eq></td></tr></table>

## B MORE ABLATIONS

These experiments blow were conducted using default thresholds $( \tau _ { f g } = 0 . 2 , \tau _ { b g } = 0 . 1 )$ , as they were implemented before threshold ablation. 


Table 9: Ablation on the setting of Pre-smoothing and Post-smoothing.


<table><tr><td>Setting</td><td>OCRBench</td><td>TextVQA</td><td>POPE</td><td>V*</td><td>Ave.</td></tr><tr><td>LLaVA-1.5-7B</td><td>31.4</td><td>46.1</td><td>85.9</td><td>50.3</td><td>53.4</td></tr><tr><td>Pre-smoothing</td><td><eq>34.8_{\uparrow 3.4}</eq></td><td><eq>58.4_{\uparrow 8.3}</eq></td><td><eq>\mathbf{87.9}_{\uparrow 2.0}</eq></td><td><eq>60.7_{\uparrow 10.4}</eq></td><td><eq>60.5_{\uparrow 7.1}</eq></td></tr><tr><td>Post-smoothing</td><td><eq>\mathbf{37.0}_{\uparrow 5.6}</eq></td><td><eq>\mathbf{58.7}_{\uparrow 12.6}</eq></td><td><eq>87.1_{\uparrow 1.2}</eq></td><td><eq>\mathbf{67.5}_{\uparrow 17.2}</eq></td><td><eq>\mathbf{62.6}_{\uparrow 9.2}</eq></td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/e11b9957-d09a-473b-8ecb-edef7d68b919/92696238595a9b81f51109b20fe6863ec70ff4d74fef449b4b508b2330077800.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/e11b9957-d09a-473b-8ecb-edef7d68b919/8a9c7d4edcd0659cc97a9aea28488e7c9f2eab573883cfdc956943b12e0b7973.jpg)



Figure 6: Performance-Throughput Trade-off on the V-Star Benchmark. Each point on the plot corresponds to a different maximum number of visual tokens. Our approach achieves a superior trade-off. The x-axis is on a logarithmic scale for clarity.


On Region Proposal Layers. In Table 8, we present an ablation study on the depth of the trainable Region Proposal Network, varying the number of tunable layers (R). We observe that increasing the RPN’s depth from one to three layers yields a significant performance improvement (from 58.4% to 62.6% average score). Beyond this point, we found that the performance gains diminished, failing to justify the additional computational cost incurred during both training and inference. Therefore, to achieve the best balance between model capacity and efficiency, we selected R=3 as the default setting for our framework. 

On Pseudo-label Smoothing. Our standard pipeline applies a Gaussian filter to the RPN’s final predicted map, a step we term post-smoothing. We investigated an alternative approach, pre-smoothing, where the filter is instead applied directly to the pseudo-labels during their generation. As shown in Table 9, the post-smoothing approach consistently outperforms pre-smoothing across most benchmarks, achieving a +2.1% higher average score. We hypothesize that this performance gap stems from a distributional shift induced by pre-smoothing. Applying a filter to the sparse pseudo-labels makes them denser, effectively enlarging the target foreground area. While this might appear beneficial, it creates a significant discrepancy between the dense, artificially softened labels and the sharper, more sparse predictions natural to the self-initilized RPN. This distribution gap complicates the training process, particularly in our data-efficient setting. This difficulty is reflected in the final convergence loss, where the pre-smoothing strategy results in a 2.5x higher value than postsmoothing (0.05 vs. 0.02). We conclude that forcing the network to learn these artificially blurred targets increases ambiguity, ultimately degrading the precision of the RoI predictions. 

## C MORE COMPARISON AND ANALYSIS

## C.1 ANALYSIS OF THE PERFORMANCE-THROUGHPUT TRADE-OFF

More recent MLLMs, such as Qwen2.5-VL, adopt dynamic resolution techniques that handle images of various sizes by producing a varying number of visual tokens. A key concern is whether our RoIselection approach is more effective than simply increasing the native input resolution of the baseline model. To address this, we analyze the trade-off between performance and efficiency by scaling the input resolution for both the Qwen2.5-VL baseline and our proposed SD-RPN. The results of this analysis on the V-Star benchmark are presented in Fig. 6. The figure plots performance against throughput (tokens/s), where the ideal model would be in the top-right corner. It is evident that the trade-off curve for our method is consistently superior to that of the baseline. For example, on Qwen2.5-VL-7B, our method achieves a performance of 77.5 at 5.1 tokens/s using a 576-resolution input, a level the baseline only reaches at 4096 resolution with a much lower throughput of 1.0 tokens/s. This demonstrates that SD-RPN provides a significantly better balance between accuracy and computational cost, confirming the effectiveness of our approach over naive resolution scaling. 

## C.2 COMPARISON TO EXTERNAL GROUNDING MODELS

A practical concern regarding our SD-RPN is its computational cost, as it utilizes a portion of the MLLM. It may seem counter-intuitive, but our SD-RPN is, in fact, significantly more computationally efficient than using an external grounding model like Grounding-DINO or YOLO-World. The reason is that our RPN re-uses computations that the MLLM must perform anyway. An external model (e.g., Grounding-DINO) would require redundant computation in the form of a separate, full image encoding pass, which is computationally expensive. In contrast, our approach uses feature re-use: our RPN attaches to the MLLM’s middle layers and operates on features that are already computed during the MLLM’s initial forward pass. The only extra cost is passing these features through the RPN layers (e.g., 3 layers in our B15R3 configuration), which is a minimal addition. 

To provide a concrete analysis, we benchmarked our method (on LLaVA-1.5-7B) against a Grounding-DINO-Base model on a NVIDIA A6000 GPU. The results, presented in Table 10, are clear. Our RPN is over 20x faster in its specific prediction step (10.3ms vs 231.2ms) because it avoids a redundant, full-image encoding. This also leads to a much faster overall evaluation time on the V* benchmark. Our method is also significantly more accurate (62.6% vs 57.6% avg.). This is because our RPN is an integrated part of the MLLM and has access to its contextual understanding of the query. External models are excellent at finding literal objects (e.g., ”the bottom man”) but fail at abstract, reasoning-based queries that don’t name a specific object, such as the example in the last row of Figure 7. Our RPN, trained on the MLLM’s internal attention, excels at this. 


Table 10: Efficiency and Performance: Ours (SD-RPN) vs. External Grounding-DINO. The ”Grounding-DINO-B (1Box)” method crops one sub-image using the single most confident bounding box, while ”(2Box)” crops two sub-images using the top 2 boxes.


<table><tr><td>Method</td><td>RPN Time (ms)</td><td>V* Eval Time (s)</td><td>OCRBench</td><td>TextVQA</td><td>POPE</td><td>V*</td><td>Ave.</td></tr><tr><td>Ours (SD-RPN)</td><td>10.3</td><td>73</td><td>37.0</td><td>58.7</td><td>87.1</td><td>67.5</td><td>62.6</td></tr><tr><td>Grounding-DINO-B (2Box)</td><td>231.2</td><td>248</td><td>32.5</td><td>54.5</td><td>82.4</td><td>62.3</td><td>57.6</td></tr><tr><td>Grounding-DINO-B (1Box)</td><td>231.2</td><td>113</td><td>32.5</td><td>52.6</td><td>86.9</td><td>56.5</td><td>57.1</td></tr></table>

## C.3 DEEPER ANALYSIS ON SOURCES OF ATTENTION NOISE

In our work, we identify and actively mitigate two primary sources of noise from the raw attention maps used to generate pseudo-labels. First, there are Type 1: Sink Tokens. We build on recent studies (Darcet et al., 2023; Kang et al., 2025a) which identify ”sink tokens”—visual tokens that attract high attention despite being semantically irrelevant to the grounded object. Second, there are Type 2: Ambiguous Foreground/Background Signals. This is a more common, semantic, and task-dependent form of noise. As illustrated in Fig. 3 and described in Section 3.2, these are ”noisy maps” that exhibit ”erroneously high attention in background areas and incomplete activation across the true foreground object.” 

As the first type of sink token belongs to a well-defined source of noise, we now investigate the second type. Our empirical study finds that questions or images in the following three categories are prone to producing noisy attention maps, as illustrated in Fig. 7. The first source of noise is cluttered scenes or dense text (e.g., the first sample in Fig. 7); in these cases, the model’s attention can be fraught with noise as it diffuses across multiple irrelevant-but-plausible regions. The second type of noise is caused by abstract or reasoning-heavy queries. In practice, we find that a simple query (e.g., ”find the cat”) tends to produce clean attention. However, many VQA queries are abstract, such as the second sample in Fig. 7. For such queries, the attention map reflects the model’s complex reasoning process, which can appear noisy as it attends to multiple pieces of related (but not all relevant) information before settling on an answer. The last type of noise is related to incomplete activation, where the attention map only weakly or partially covers the true region of interest, as shown in the last sample of Fig. 7. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/e11b9957-d09a-473b-8ecb-edef7d68b919/6b8f081438b26f19bc018dc0db3790783be1b5954c9ed65dfc6e5b5852d49468.jpg)



Figure 7: Examples of noisy and sparse attention maps for abstract queries.


## C.4 COMPARISON WITH TOKEN COMPRESSION METHODS

We provide a detailed comparison of our SD-RPN with other SoTA methods designed for highresolution visual understanding in this section. To facilitate a fair and transparent comparison, Table 11 summarizes the performance of our method against several competitive models on key document and OCR benchmarks. The table also highlights critical architectural and data differences, such as the base Large Language Model (LLM), the size of the training dataset, and the maximum input resolution, all of which are significant factors that influence final performance. 


Table 11: Comparison with SoTA high-resolution methods on document and OCR benchmarks. <sup>#</sup>Denotes results obtained using OCR tokens. <sup>*</sup>Denotes results reported in the LLaVA-TokenPacker-HD paper (Li et al., 2025). The ”+152K” in the #Data column indicates the number of additional samples used for fine-tuning our SD-RPN on the corresponding baseline models.


<table><tr><td>Method</td><td>LLM</td><td>#Data</td><td>Max Res.</td><td>TextVQAval</td><td>OCRBenchtest</td><td>DocVQAtest</td><td>ChartQAtest</td></tr><tr><td>LLaVA-1.5</td><td>Vicuna-7B</td><td>1.2M</td><td>336×336</td><td>58.2#</td><td>31.4</td><td>22.2</td><td>18.1</td></tr><tr><td>LLaVA-1.5 + SD-RPN</td><td>Vicuna-7B</td><td>+152K</td><td>336×336</td><td>64.2#</td><td>37.0</td><td>35.1</td><td>20.1</td></tr><tr><td>Monkey (Li et al., 2024b)</td><td>Qwen-7B</td><td>1.4M</td><td>896×896</td><td>67.6</td><td>-</td><td>66.5</td><td>65.1</td></tr><tr><td>LLaVA-NeXT (Liu et al., 2024a)</td><td>Vicuna-7B</td><td>1.3M</td><td>672×672</td><td>64.9*</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Mini-Gemini-HD (Li et al., 2024a)</td><td>Vicuna-7B</td><td>2.7M</td><td>536×1536</td><td>68.4*</td><td>456*</td><td>65.0*</td><td>-</td></tr><tr><td>LLaVA-TokenPacker-HD (Li et al., 2025)</td><td>Vicuna-7B</td><td>2.7M</td><td>1088×1088</td><td>68.0*</td><td>452*</td><td>60.2*</td><td>-</td></tr><tr><td>DeepSeek-VL</td><td>Vicuna-7B</td><td>-</td><td>1024×1024</td><td>63.0</td><td>42.4</td><td>49.3</td><td>61.4</td></tr><tr><td>DeepSeek-VL + SD-RPN</td><td>Vicuna-7B</td><td>+152K</td><td>1024×1024</td><td>71.5</td><td>47.9</td><td>64.5</td><td>63.6</td></tr><tr><td>Qwen2.5-VL</td><td>Qwen2.5-7B</td><td>-</td><td>672×672</td><td>81.1</td><td>81.5</td><td>92.6</td><td>82.6</td></tr><tr><td>Qwen2.5-VL + SD-RPN</td><td>Qwen2.5-7B</td><td>+152K</td><td>672×672</td><td>83.5</td><td>82.9</td><td>94.4</td><td>85.5</td></tr></table>

The results in Table 11 lead to two primary conclusions. First, the overall performance of our method, like others, is strongly correlated with the capabilities of the underlying base MLLM. When applied to an earlier baseline such as LLaVA-1.5, our method’s absolute scores on benchmarks like DocVQA are understandably lower than SoTA methods built on more advanced foundations, which often leverage richer training sets and higher input resolutions. This performance difference is primarily a reflection of the base model’s capacity rather than a limitation of our enhancement module. Second, our SD-RPN module demonstrates consistent efficacy across base models of varying strengths. When integrated with a more competitive baseline like DeepSeek-VL-7B, our method achieves performance on par with or slightly exceeding other leading methods. To further validate this generalizability, we applied SD-RPN to the powerful Qwen2.5-VL model. Even on this SoTA baseline, our method delivers significant and consistent gains across all benchmarks, utilizing an identical and minimal fine-tuning recipe. 

This highlights a key advantage of our approach: SD-RPN provides a data-efficient and scalable pathway for enhancing the fine-grained perceptual capabilities of MLLMs. Unlike methods that necessitate extensive supervised fine-tuning on massive datasets, our self-distillation technique offers a resource-efficient solution for improving high-resolution understanding that can readily adapt to the continuous advancements in base MLLMs. 

## D PROMPT USAGE

Below are the prompts used for pseudo-label generation with LLaVA-1.5 and DeepSeek-VL. Each template follows the respective model’s standard input format. 

## LLaVA-1.5

<image> USER:{question} Answer the question using a single word or phrase. ASSISTANT: 

## DeepSeek-VL

<image_placeholder> USER:{question} Answer the question using a single word or phrase. Assistant: 

## E VISUALIZATION

To provide a deeper, qualitative understanding of our model’s advantages, Figure 8 presents a series of challenging visual question-answering examples. These samples, drawn from the V-Star, TextVQA, and DocVQA benchmarks, were specifically chosen because they require recognizing and reasoning about fine-grained details within cluttered scenes. The baseline model, which relies on processing the entire image at a global level, often fails to perceive these critical details. In contrast, our SD-RPN first employs its region proposal mechanism to identify the most salient RoI for a given question—visualized by the green bounding boxes for subsequent fine-grained perception. This two-stage approach allows our model to accurately perform tasks like detailed text extraction, object attribute identification, and spatial reasoning, achieving correct answers where the baseline consistently fails. 

In addition to the successful cases, we also provide a failure case analysis in Fig. 9. The failure cases can be broadly attributed to three main reasons: 

• Incomplete Activation. The top-left ”cup” case is a representative example. The RPN correctly identifies the measuring cup but only activates on a small portion of it. This partial RoI provides incomplete visual information to the MLLM, leading to an erroneous prediction. 

• Localization Error. The top-right ”book” example illustrates this type. The RPN is supposed to highlight the book’s cover to read the name but erroneously highlights an unrelated region. Given this misleading visual cue, the MLLM fails to produce the correct result. 

• Spatial Relationship Error. The bottom row shows errors of this type. Here, the RPN highlights all relevant objects correctly (e.g., ”bowl” and ”faucet”). However, the MLLM still produces the wrong result, suggesting that it struggles to preserve and reason about the precise spatial relationships between objects when they are processed as fine-grained crops. 

## F WHY PREDICTING ROI SCORES OUTPERFORMS USING RAW ATTENTION

Let X denote token-level features available to the region-of-interest (RoI) predictor, $Y \in \{ 0 , 1 \}$ is the latent foreground (FG) indicator, and $A \in [ 0 , 1 ]$ is the model’s response-to-image attention used as a noisy proxy for Y . Define the posterior $\begin{array} { r } { \eta ( \dot { \boldsymbol { x } } ) \mathrel { \mathop : } = \operatorname* { P r } ( Y = 1 \mid \boldsymbol { X } \stackrel { \cdot } { = } \boldsymbol { x } ) } \end{array}$ 

Regression view (continuous RoI scoring). We train a predictor h by minimizing the population squared loss 

$$
\mathcal {R} (h) = \mathbb {E} \big [ (h (X) - A) ^ {2} \big ].
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/e11b9957-d09a-473b-8ecb-edef7d68b919/3af61f5ce5451f211068a3e3d51d0a5ad75a37f80eee2f97801c3897c4cb45d1.jpg)



Figure 8: Qualitative comparison of our SD-RPN against the LLaVA-1.5-7B baseline on challenging samples from the V-Star, TextVQA, and DocVQA benchmarks. For each example, the top image displays the original input with the Region-of-Interest (RoI) predicted by our method, while the image below it visualizes the dense RoI map. These examples highlight SD-RPN’s ability to precisely localize and analyze fine-grained visual information (e.g., text, small objects), leading to correct answers where the baseline, which processes the full image, consistently fails.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/e11b9957-d09a-473b-8ecb-edef7d68b919/5afd1d1b83c4f37fd4d47869bdd3a6b594db8caa2366320d615db4c5ccd21e88.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/e11b9957-d09a-473b-8ecb-edef7d68b919/9731be16091b2beab9cbc9e78ce71210b0287041ea235552c9358d5fd76a795f.jpg)



Figure 9: Failure case analysis of SD-RPN on the LLaVA-1.5-7B baseline. We present examples where the baseline model succeeds, but our method fails. These failures can be categorized into three main types: incomplete activation (top-left), localization error (top-right), and spatial relationship error (bottom row).


By pointwise conditional minimization, the optimal predictor is the conditional expectation 

$$
h ^ {\star} (x) = \mathbb {E} [ A \mid X = x ].\tag{9}
$$

Under standard noise models, $h ^ { \star }$ is an affine function of $\eta ( x )$ : 

$$
\mathbb {E} [ A \mid X = x ] = \left\{ \begin{array}{l l} (1 - \rho_ {1} - \rho_ {0})   \eta (x) + \rho_ {0}, & \text { class - conditional   noise   (CCN) }, \\ \mu_ {0} + (\mu_ {1} - \mu_ {0})   \eta (x), & \text { additive   activation   model }. \end{array} \right.
$$

In both cases, $\mathbb { E } [ A \mid X = x ]$ is strictly increasing in $\eta ( x )$ when $\rho _ { 0 } + \rho _ { 1 } < 1$ or $\mu _ { 1 } ~ > ~ \mu _ { 0 }$ , so thresholding $h ^ { \star }$ recovers the Bayes decision boundary up to a constant shift. 

Noise reduction via conditional averaging. The raw attention signal A can be decomposed into its ”signal” component, defined as $h ^ { \star } ( \breve { X } ) \stackrel { \smile } { = } \mathbb { E } [ A \mid X ] .$ , and its ”noise” component, $\epsilon = \overset { \cdot } { A } - \mathbb { E } [ A \ |$ $X ]$ . By construction, the noise is zero-mean conditional on the features $( \dot { \mathbb { E } } [ \epsilon \mid X ] = 0 )$ . We can measure the quality of any estimator by its Mean Squared Error (MSE) with respect to this true underlying signal, $\dot { h } ^ { \star } ( X )$ . 

The MSE of the optimal predictor $h ^ { \star } ( X )$ with respect to the true signal is, trivially, zero: 

$$
\mathbb {E} \big [ (h ^ {\star} (X) - h ^ {\star} (X)) ^ {2} \big ] = 0.
$$

In contrast, the MSE of the raw attention signal A with respect to the true signal is: 

$$
\mathbb {E} \left[ (A - h ^ {\star} (X)) ^ {2} \right] = \mathbb {E} \left[ (A - \mathbb {E} [ A \mid X ]) ^ {2} \right] = \mathbb {E} \left[ \mathbb {E} \left[ (A - \mathbb {E} [ A \mid X ]) ^ {2} \mid X \right] \right] = \mathbb {E} [ \operatorname{Var} (A \mid X) ].
$$

This term, the expected conditional variance, represents the irreducible error inherent in the attention signal. As long as the attention is not a perfectly deterministic function of the features (i.e., Var(A | $\check { X ) } > 0 )$ , we have: 

$$
\mathbb {E} \big [ (h ^ {\star} (X) - h ^ {\star} (X)) ^ {2} \big ] <   \mathbb {E} \big [ (A - h ^ {\star} (X)) ^ {2} \big ].
$$

This inequality formally proves that the optimal predictor $h ^ { \star } ( X )$ is a strictly better, denoised estimate of the underlying signal than the raw attention A. Any learned predictor $\hat { h }$ that successfully approximates $h ^ { \star }$ will therefore also be a more stable and accurate predictor. 

Classification view (binary RoI selection). For symmetric class-conditional noise (symmetric CCN) with flip rate $\rho < { \frac { 1 } { 2 } }$ 

$$
\operatorname * {P r} (A = 1 \mid X = x) = (1 - 2 \rho) \eta (x) + \rho ,\tag{10}
$$

a strictly increasing transform of $\eta ( x )$ . Minimizing a classification-calibrated surrogate on A yields a scorer whose optimal threshold (shifted by $\rho )$ implements the clean Bayes rule. In contrast, selecting RoIs directly via A suffers from the conditional variance $\operatorname { V a r } ( A \mid { \dot { X } } )$ , leading to higher false positives/negatives, especially in low-margin regions. 

Implication for RoI prediction. A RoI head trained to regress A learns $\mathbb { E } [ A \mid X ]$ , which (i) is order-preserving with η(x) by the affine forms above, and (ii) enjoys reduced variance. Thresholding a learned predictor ${ \hat { h } } ( X )$ therefore yields more accurate and stable RoI proposals than thresholding raw attention A, matching our empirical gains. Our proposed Label Assignment strategy can be seen as a further refinement of this principle, where we structure the learning problem as a classification task on high-confidence tokens to make the training process even more robust to the noise in A. 

## G THE USE OF LARGE LANGUAGE MODELS

Throughout the preparation of this manuscript, we utilized a Large Language Model (LLM) as a general-purpose writing assistant to enhance the quality and clarity of our text. The LLM’s role was strictly limited to that of a linguistic refinement tool. We used it to help polish our initial drafts by: 

• Rephrasing sentences and paragraphs for improved readability, conciseness, and a more formal academic tone. 

• Correcting grammatical errors, spelling, and punctuation. 

• Improving the logical flow and transitions between sentences. 

• Assisting with the generation of LaTeX code for tables based on our provided experimental data. 

It is important to state that all core research ideas, the conceptualization of the proposed framework, the experimental design, the analysis of results, and the final scientific conclusions presented in this paper were conceived and articulated entirely by the human authors. The LLM did not contribute to the scientific ideation or the results of this work. The authors have reviewed, edited, and take full responsibility for all content in this manuscript. 