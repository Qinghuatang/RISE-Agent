# Multimodal Language Models See Better When They Look Shallower

Haoran Chen<sup>1,2,3</sup>*, Junyan Lin<sup>2,3</sup>*, Xinghao Chen<sup>2,3</sup>, Yue Fan Jianfeng Dong<sup>1</sup>, Xin Jin<sup>2,3</sup>, Hui Su<sup>5</sup>, Jinlan Fu<sup>6</sup>, Xiaoyu Shen<sup>2,3†</sup> 

<sup>1</sup>Zhejiang Gongshang University 

<sup>2</sup>Ningbo Key Laboratory of Spatial Intelligence and Digital Derivative <sup>3</sup>Institute of Digital Twin, Eastern Institute of Technology, Ningbo <sup>4</sup>Genmo.ai, <sup>5</sup>Meituan Inc., <sup>6</sup>National University of Singapore 

haoranchr321@gmail.com, xyshen@eitech.edu.cn 

## Abstract

Multimodal large language models (MLLMs) typically extract visual features from the final layers of a pretrained Vision Transformer (ViT). This widespread deep-layer bias, however, is largely driven by empirical convention rather than principled analysis. While prior studies suggest that different ViT layers capture different types of information—shallower layers focusing on fine visual details and deeper layers aligning more closely with textual semantics, the impact of this variation on MLLM performance remains underexplored. We present the first comprehensive study of visual layer selection for MLLMs, analyzing representation similarity across ViT layers to establish shallow, middle, and deep layer groupings. Through extensive evaluation of MLLMs (1.4B–7B parameters) across 10 benchmarks encompassing 60+ tasks, we find that while deep layers excel in semantic-rich tasks like OCR, shallow and middle layers significantly outperform them on fine-grained visual tasks in cluding counting, positioning, and object localization. Building on these insights, we propose a lightweight feature fusion method that strategically incorporates shallower layers, achieving consistent improvements over both single-layer and specialized fusion baselines. Our work offers the first principled study of visual layer selection in MLLMs, showing that MLLMs can often see better when they look shallower. 

## 1 Introduction

Multimodal Large Language Models (MLLMs) extend the capabilities of traditional Large Language Models (LLMs) by enabling joint reasoning over both visual and textual inputs (Hong et al., 2024; Bai et al., 2023; Chen et al., 2024a; Shen et al., 2024). Typically, these models integrate a pretrained Vision Transformer (ViT) to extract image features, which are then projected into the language embedding space of an LLM. This architecture enables unified multimodal understanding and powers a wide range of applications, including robotic navigation, medical diagnostics, and visual question answering (Alayrac et al., 2022; Chen et al., 2024b; Bai et al., 2023; Tong et al., 2024a; Li et al., 2025). 

While recent advancements have notably improved the language reasoning capabilities of MLLMs, the visual processing pipeline, particularly the selection of ViT layers for visual representation, remains insufficiently explored. In practice, MLLMs often default to using features from the deepest layers of ViT models. For instance, Qwen-VL (Bai et al., 2025) and InternVL-6B v1.2/1.5 use the final layer of CLIP-ViT (Radford et al., 2021), while other InternVL variants select the fourth-to-last layer (Chen et al., 2024b). The LLaVA series (Liu et al., 2023b,a, 2024a) relies on the penultimate layer. However, these choices are largely heuristics rather than systematic evaluation (Yao et al., 2024; Jiang et al., 2023; Tong et al., 2024b; Cao et al., 2024; Li et al., 2026). 

Previous work has shown that ViT layers encode a hierarchy of semantic information—from low-level edge detectors in shallow layers to abstract object representations in deeper layers (Gandelsman et al., 2024; Yao et al., 2024; Tong et al., 2024b). Yet, how these layer-wise representations affect MLLM performance remains poorly understood. This paper addresses this gap by systematically investigating which ViT layers provide the most effective visualfeaturesfor MLLMs. 

We begin by analyzing Layer-wise Representation Similarity (LRS) across CLIP-ViT’s hidden states using cosine similarity, revealing three semantically coherent layer groups: shallow (layers 1–12), middle (13–20), and deep (21–24) (Fig. 1). This categorization provides a foundation for structured layer selection and fusion. 

Building upon this foundation, we first systematically assess the efficacy of different deep vision layers. Our analysis reveals that while the penultimate layer does not universally achieve peak performance in every scenario, it demonstrates consistent superiority across all evaluated model scales (1.4B, 2.7B, and 7B parameters). This advantage stems from the penultimate layer’s unique balance of preserving fine-grained visual details while maintaining strong alignment with textual representations. Notably, the performance gap between the penultimate layer and other deep layers widens as model scale increases. This suggests that simply using larger LLMs cannot compensate for suboptimal visual feature selection, underscoring the critical importance of visual layer choice in MLLMs. 

Having established the penultimate layer’s strength among deep layers, we ask a more fundamental question: Can shallower ViT layers offer complementary or even superior information? Our analysis shows that shallow and middle layers outperform deep layers in approximately one-third of sub-tasks in the MME benchmark (Fu et al., 2024) (Fig. 3), particularly in tasks involving fine-grained localization and counting. For instance, layer 18 outperforms the penultimate layer by 20% on position tasks (Fig.9). Similar trends are observed in MMVet (Yu et al., 2023). Although shallow layers generally show lower average performance, they still excel on a significant subset of tasks (Fig.2). In contrast, deeper layers remain crucial for tasks with high-level semantic demands such as OCR. To assess robustness, we evaluate across three training data scales (665k, 737k, and 1M samples). Despite some fluctuations, our findings consistently demonstrate that shallow and middle layers carry underutilized yet valuable information. 

Motivated by these insights, we propose a simple yet effective fusion strategy that combines visual features from shallow, middle, and deep layers. Our method uses a single linear projection layer, keeping computational overhead minimal while achieving substantial performance gains. This minimalist approach offers a principled alternative to existing ad-hoc layer selection and fusion methods. Unlike prior works (Yao et al., 2024; Hong et al., 2024; Cao et al., 2024) that explore hierarchical feature fusion or LLM-aligned selection heuristically, our study provides the first systematic analysis of layerwise information variation within ViTs, grounded in both intrinsic representation structure and downstream performance. Our key contributions are summarized as follows: 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/0c52dca9-18d4-4e52-83c6-d86f87e7f8e1/6939ae1c96313c4ae63413893f228804a9b8206c69d9d647576e1f3d6d230fc0.jpg)



(a)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/0c52dca9-18d4-4e52-83c6-d86f87e7f8e1/b668f9f083fcd658ca07aa6afc651c5e45e7a8c8115d09ef0439bbdd4f1a7617.jpg)



(b)



Figure 1: (a) Average cosine similarity of visual representations across different layers in CLIP-ViT. (b) Layer-wise performance on OCR tasks. The results highlight three distinct representation regions and their influence on performance.


(1) We identify three semantically coherent groups of ViT layers (shallow, middle, deep) based on representation similarity. We show that shallow and middle layers, which are often overlooked, can outperform the commonly used deep layers (Sec. 4). 

(2) Through extensive experiments across different data sizes and model scales, we confirm the generalization of our findings. Even as gains diminish with scaling, shallow and middle layers continue to exhibit unique strengths over deep layers in certain sub-tasks (Sec. 5). 

(3) We design a linear-layer-based fusion method that integrates features from all three layer groups. It outperforms both specialized fusion designs (e.g., DenseConnector (Yao et al., 2024), MMFuser (Cao et al., 2024)) and standard practices in current MLLMs (e.g., using only the penultimate layer) (Sec. 6). 

## 2 Related Work

Visual Encoder in Multimodal LLMs Serving as the “eyes” of MLLMs, the vision encoder sets the upper bound of the model’s perceptual capabilities. CLIP, through image-text contrastive learning effectively aligns visual representation with text space and is widely adopted as the visual encoder in models such as LLaVA (Liu et al., 2023b,a), Qwen-VL (Wang et al., 2024), Flamingo (Alayrac et al., 2022), and BLIP (Li et al., 2023b). Other foundational vision models, such as DINOv2 (Oquab et al., 2023), SigLIP (Zhai et al., 2023), ConvNeXT (Liu et al., 2022), are also utilized to build MLLMs. In this paper, we select the widely used CLIP-ViT model as the focus of our layer-wise analysis. 

Visual Layer Selection Recent studies have explored incorporating shallow visual features within the ViT of multimodal language models, such as DenseConnector (Yao et al., 2024), MMFuser (Cao et al., 2024). Lin et al. (2025) have further investigated internal fusion strategies by integrating multiple visual layers with language representations, highlighting the critical role of visual layer selection in effective multimodal integration. 

Previous methods have largely relied on intuitive, heuristic-based strategies, such as evenly sampling layers. Although some approaches have explored the distinct characteristics of different ViT layers (Gandelsman et al., 2024), the specific roles of layers at different depths in multimodal tasks remain unclear. This study conducts a comprehensive analysis of layer-wise visual representations in MLLMs, aiming to inform the selection of visual layers and guide the design of future visual fusion strategies. 

## 3 Overall Setup

Problem Formulation MLLMs typically comprise three core components: a vision encoder, a connector that maps visual features to the language space, and a large language model (LLM) (Lin et al., 2024). This architecture empowers MLLMs to handle a diverse array of perception and reasoning tasks across both visual and textual modalities. 

Most modern MLLMs adopt a pre-trained CLIP-ViT (Radford et al., 2021) as their image encoder. A ViT encodes an image into a sequence of token embeddings through a stack of transformer blocks. Each block (or layer) progressively refines the visual representations, with earlier layers focusing on low-level spatial details and later layers capturing more abstract, semantic information. 

Formally, given an image I, the vision encoder produces a set of layer-wise outputs: 

$$
\mathbf {H} ^ {(1)}, \mathbf {H} ^ {(2)}, \dots , \mathbf {H} ^ {(L)} \quad \text { where } \quad \mathbf {H} ^ {(l)} \in \mathbb {R} ^ {T \times d}
$$

H<sup>(l)</sup> denotes the embedding at the l-th layer, T is the number of tokens, and d is the dimension. 

Despite the availability of rich multi-level features, most MLLMs select a single layer—often the penultimate or final one—to represent the entire image. This practice may overlook complementary signals from shallower layers that encode fine-grained visual details. In this work, we systematically investigate the impact of using different ViT layers for visual input and explore how selecting appropriate layers can improve MLLM performance across diverse tasks. 

Partitioning of Visual Representations To examine the behavioral patterns of different visual layers, we analyze the relationships between them based on cosine similarities. Inspired by prior findings (Sun et al., 2024) that LLMs exhibit several distinct representation spaces through such analysis, we similarly identify three significantly different representation spaces within CLIP-ViT. 

As shown in Fig.1a, three distinct representation spaces emerge among the visual layers (see Appendix A for computational details). Experiments on OCR and TextVQA (Fig. 1b) further indicate that shallow layers contribute little to performance, which rises substantially in the middle layers and peaks in the deep layers. Layers within the same representation space exhibit similar behaviors. 

Based on behavioral similarity, we categorize the 24 CLIP-ViT visual layers into three groups: shallow layers (1 to 12), middle layers (13 to 20), and deep layers (21 to 24). 

Implementation Details We employ CLIP ViT-L/14 (336px) (Radford et al., 2021) as the visual encoder and 1.4B MobileLLaMA (Chu et al., 2024) as the language model for efficiency analysis, with a one-layer MLP serving as the connector. Training follows a two-phase strategy aligned with LLaVA (Liu et al., 2023b). AdamW optimizer with a cosine annealing scheduler is used, with learning rates of 1e-3 (phase one) and 2e-5 (phase two), and batch sizes of 256 and 128. Training on four NVIDIA A100 80GB GPUs takes 2 hours for phase one and 8 hours for phase two. We adopt the LLaVA 1.5 (Liu et al., 2023b) dataset, comprising 558K image-caption pairs which is carefully filtered from CC3M (Sharma et al., 2018) for pre-training and 665K conversational instances for instruction tuning. Unless explicitly noted, the experimental setup remains the same. 

Evaluation Benchmarks To comprehensively explore and evaluate various visual representations, we classified the benchmarks into four categories following previous work (Tong et al., 2024a): General tasks, OCR tasks, Vision-centric tasks, and Hallucination tasks. 

The General tasks category assesses basic vision-language reasoning abilities, including MME (Fu et al., 2024) (yes/no questions on attributes like existence and color), MMBench (Liu et al., 2024c) (multiple-choice across diverse aspects), SEEDBench (Li et al., 2023a) (spatial and temporal reasoning), and GQA (Hudson and Manning, 2019a) (complex real-world VQA). The OCR category evaluates a model’s ability to recognize textual content from images, featuring TextVQA (Singh et al., 2019) and OCRBench (Liu et al., 2024d). The Vision-centric category emphasize fine-grained perception and localization, including CVBench (Tong et al., 2024a) (evaluating spatial relations and depth), RealWorldQA (real-world QA), MMVet (Yu et al., 2023) (general multimodal assessment), and RefCOCO (Yu et al., 2016) (visual grounding). Finally, the Hallucination category includes POPE (Li et al., 2023c), which evaluates whether MLLMs generate false or invented content not grounded in the image. 

<table><tr><td rowspan="2">Layers</td><td colspan="5">General</td><td colspan="2">OCR</td><td colspan="6">Vision-Centric</td><td>Hallu</td></tr><tr><td><eq>MME^P</eq></td><td><eq>MME^C</eq></td><td>MMB</td><td>SEEDB</td><td>GQA</td><td>TVQA</td><td>OCRB</td><td>CVB</td><td><eq>CVB^{2D}</eq></td><td><eq>CVB^{3D}</eq></td><td>RWD</td><td>MMVet</td><td>RefCOCO</td><td>POPE</td></tr><tr><td>1</td><td>750.1</td><td>211.4</td><td>0</td><td>25.30</td><td>40.55</td><td>7.99</td><td>24</td><td>40.14</td><td>34.87</td><td>45.42</td><td>38.17</td><td>9.9</td><td>5.73</td><td>70.21</td></tr><tr><td>2</td><td>790.2</td><td>212.5</td><td>0.34</td><td>25.76</td><td>41.24</td><td>7.96</td><td>23</td><td>40.21</td><td>34.34</td><td>46.08</td><td>37.12</td><td>10.1</td><td>5.36</td><td>71.01</td></tr><tr><td>3</td><td>742.7</td><td>219.2</td><td>0.17</td><td>25.00</td><td>41.76</td><td>8.10</td><td>28</td><td>42.69</td><td>37.89</td><td>47.50</td><td>36.08</td><td>10.4</td><td>6.87</td><td>72.33</td></tr><tr><td>4</td><td>788.4</td><td>239.6</td><td>0</td><td>25.53</td><td>42.26</td><td>8.50</td><td>23</td><td>42.69</td><td>36.71</td><td>48.67</td><td>37.25</td><td>10.2</td><td>7.61</td><td>72.34</td></tr><tr><td>5</td><td>813.2</td><td>220.7</td><td>0.17</td><td>25.26</td><td>42.74</td><td>8.22</td><td>21</td><td>41.30</td><td>33.69</td><td>48.92</td><td>36.86</td><td>10.9</td><td>8.25</td><td>72.91</td></tr><tr><td>6</td><td>838.8</td><td>227.8</td><td>0</td><td>25.23</td><td>43.16</td><td>8.26</td><td>24</td><td>41.69</td><td>35.97</td><td>47.42</td><td>36.86</td><td>11.5</td><td>9.31</td><td>75.18</td></tr><tr><td>7</td><td>815.6</td><td>235.7</td><td>0</td><td>25.74</td><td>44.90</td><td>8.75</td><td>25</td><td>43.02</td><td>37.12</td><td>48.92</td><td>37.39</td><td>10.6</td><td>11.10</td><td>75.44</td></tr><tr><td>8</td><td>857.7</td><td>237.5</td><td>0</td><td>25.48</td><td>46.14</td><td>8.77</td><td>25</td><td>41.43</td><td>36.85</td><td>46.00</td><td>36.99</td><td>11.2</td><td>10.79</td><td>76.20</td></tr><tr><td>9</td><td>889.7</td><td>232.8</td><td>0.17</td><td>27.72</td><td>47.02</td><td>9.05</td><td>28</td><td>40.53</td><td>36.23</td><td>44.83</td><td>37.12</td><td>13.0</td><td>10.06</td><td>77.84</td></tr><tr><td>10</td><td>903.4</td><td>228.2</td><td>0.17</td><td>26.61</td><td>48.39</td><td>9.03</td><td>30</td><td>41.8</td><td>36.19</td><td>47.42</td><td>37.39</td><td>11.4</td><td>13.50</td><td>77.46</td></tr><tr><td>11</td><td>935.3</td><td>224.3</td><td>0.52</td><td>26.58</td><td>49.85</td><td>10.65</td><td>32</td><td>42.51</td><td>37.27</td><td>47.75</td><td>36.86</td><td>14.2</td><td>12.14</td><td>79.24</td></tr><tr><td>12</td><td>980.1</td><td>232.1</td><td>0.09</td><td>26.85</td><td>50.39</td><td>16.58</td><td>70</td><td>41.81</td><td>36.7</td><td>46.92</td><td>38.56</td><td>12.6</td><td>11.20</td><td>80.63</td></tr><tr><td>13</td><td>964.0</td><td>252.5</td><td>0.09</td><td>26.33</td><td>51.14</td><td>18.12</td><td>91</td><td>41.71</td><td>35.75</td><td>47.67</td><td>37.25</td><td>11.6</td><td>12.50</td><td>81.39</td></tr><tr><td>14</td><td>984.2</td><td>265.4</td><td>0.69</td><td>34.07</td><td>51.83</td><td>22.86</td><td>130</td><td>42.69</td><td>36.12</td><td>49.25</td><td>39.08</td><td>13.8</td><td>14.37</td><td>81.97</td></tr><tr><td>15</td><td>1042.8</td><td>227.5</td><td>0.17</td><td>28.98</td><td>52.89</td><td>25.79</td><td>155</td><td>43.85</td><td>36.37</td><td>51.33</td><td>37.12</td><td>13.6</td><td>13.77</td><td>83.11</td></tr><tr><td>16</td><td>1069.5</td><td>225.4</td><td>0</td><td>27.95</td><td>52.81</td><td>28.08</td><td>166</td><td>43.26</td><td>36.61</td><td>49.92</td><td>38.04</td><td>13.7</td><td>15.41</td><td>84.32</td></tr><tr><td>17</td><td>1074.8</td><td>230.4</td><td>0.26</td><td>32.81</td><td>53.86</td><td>28.25</td><td>200</td><td>47.26</td><td>39.43</td><td>55.08</td><td>39.22</td><td>15.4</td><td>18.49</td><td>84.46</td></tr><tr><td>18</td><td>1088.7</td><td>237.1</td><td>29.38</td><td>52.06</td><td>54.37</td><td>31.44</td><td>200</td><td>47.29</td><td>41.17</td><td>53.42</td><td>39.48</td><td>14.3</td><td>17.04</td><td>84.26</td></tr><tr><td>19</td><td>945.1</td><td>236.8</td><td>20.02</td><td>44.64</td><td>48.32</td><td>18.27</td><td>121</td><td>45.69</td><td>37.21</td><td>54.17</td><td>35.95</td><td>13.2</td><td>18.22</td><td>81.47</td></tr><tr><td>20</td><td>1118.2</td><td>232.1</td><td>26.03</td><td>51.72</td><td>54.83</td><td>32.05</td><td>211</td><td>47.29</td><td>40.32</td><td>54.25</td><td>38.82</td><td>16.3</td><td>18.49</td><td>84.76</td></tr><tr><td>21</td><td>1041.4</td><td>212.5</td><td>0.95</td><td>35.42</td><td>49.47</td><td>28.10</td><td>190</td><td>44.37</td><td>39.32</td><td>49.42</td><td>39.87</td><td>14.5</td><td>17.09</td><td>81.91</td></tr><tr><td>22</td><td>1123.6</td><td>238.9</td><td>23.28</td><td>49.60</td><td>54.52</td><td>30.84</td><td>211</td><td>44.37</td><td>36.73</td><td>52.00</td><td>39.87</td><td>17.3</td><td>16.32</td><td>84.79</td></tr><tr><td>23</td><td>1142.7</td><td>245.0</td><td>35.31</td><td>52.84</td><td>54.61</td><td>33.73</td><td>233</td><td>44.26</td><td>38.02</td><td>50.50</td><td>45.36</td><td>18.0</td><td>17.08</td><td>84.00</td></tr><tr><td>24</td><td>1114.1</td><td>243.5</td><td>32.65</td><td>51.09</td><td>53.61</td><td>30.63</td><td>197</td><td>46.68</td><td>39.78</td><td>53.58</td><td>43.92</td><td>16.1</td><td>17.08</td><td>83.65</td></tr></table>


Table 1: Performance across layers 1–24. ${ \bf M M E } ^ { P }$ and $\mathbf { M M E } ^ { C }$ represent the MME perception and cognition tasks respectively. SEEDB, GQA, OCRB and CVB refer to SEEDBench, General QA tasks, OCRBench and CVBench, with $\mathrm { C V B ^ { 2 D } }$ and $\mathrm { C V B } ^ { 3 { \bar { D } } }$ indicating the 2D/3D subtasks of CVBench, respectively. RWD stands for RealWorldQA. This table provides a detailed analysis of all 24 layers, highlighting that many optimal performances arefound in the middle layers, which are marked in bold.


## 4 Experiment: Layer-wise Exploration

Previous studies have primarily used techniques such as linear probing and attention head decomposition to analyze CLIP-ViT representations (Gandelsman et al., 2024). While these methods reveal what types of information are present in different ViT layers, they do not assess whether such information can be effectively utilized by MLLMs. The mere presence of information in a particular layer does not guarantee its usefulness when integrated into an MLLM. In contrast, our work goes beyond probing for representational content—we systematically evaluate how each ViT layer contributes to downstream MLLM performance. To this end, we conduct a layerwise exploration by individually connecting each visual layer to the language model, training the corresponding MLLM, and benchmarking its task performance. The layerwise performance is shown in Tab. 1 and Fig. 2. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/0c52dca9-18d4-4e52-83c6-d86f87e7f8e1/dae492b843f884156bb27b088cf5a84a9002b0af486b9dfab8fbe14033cda027.jpg)



Figure 2: Averaged performance of layers 1 to 24 across various tasks. General represents tasks from MME, MMBench, GQA, and SEEDBench. OCR includes includes TextVQA and OCRBench. CVB corresponds to CVBench, whereas VC∗ includes RefCOCO, RealWorldQA, and MMVet. Results show that the final layer underperforms the penultimate layer, and middle layers sometimes surpass deeper ones.


## 4.1 Deep-to-Deep Layer Comparison

A common practice is to use deep layers from ViT as input to the MLLM. In this section, we investigate the effectiveness of this approach. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/0c52dca9-18d4-4e52-83c6-d86f87e7f8e1/223b3cf510d42c7eca1ffa71a1659c6daa7a1cb2e7d733ec8cdeb67fc52fa3b9.jpg)



(a) MME


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/0c52dca9-18d4-4e52-83c6-d86f87e7f8e1/9e83eed7c37f3f129cbc5ebf4be58f43e9d1b6afc8784685b6e951d1a85e57c2.jpg)



(b) MMVet


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/0c52dca9-18d4-4e52-83c6-d86f87e7f8e1/92d54b5225ae8c0a0ac7f0c3a49f4a3043994a7bd9e5d1917707475eb9ec5a7d.jpg)



(c) MMBench


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/0c52dca9-18d4-4e52-83c6-d86f87e7f8e1/303e5f33cedee0c2edb7c86e4fe9685e93a95f629de803ef458a0d9e2514f2e7.jpg)



(d) SEEDBench



Figure 3: Layer-wise performance distribution across four benchmarks: (a) MME, (b) MMVet, (c) MMBench, and (d) SEEDBench. The x-axis corresponds to layer indices and the y-axis indicates the sub-tasks index (see Tab. 12 for details). Top-performing layers for each sub-task are highlighted with color-coded markers: (1st place), (2nd place), and (3rd place).


The final layer is not the optimal choice: As shown in Tab. 1 and Fig. 2, the final layer does NOT perform the best on any benchmark. For general tasks, a noticeable performance drop is observed at the final layer, with OCR tasks exhibiting particularly severe degradation. A similar trend is evident in POPE. However, vision-centric tasks partially show this decline. Overall, these results indicate that the final layer is NOT the optimal choice for representation across tasks. 

The underlying reason might lie in the CLIP model’s training mechanism, where supervision primarily focuses on aligning the final layer [CLS] token with text embeddings. The [CLS] token in the final layer is optimized by CLIP’s contrastive loss, making it highly specialized for the image-text matching task. However, this optimization process can significantly suppress local image features in the final layer, which can harm performance when finer-grained details are required. 

Penultimate layer as the optimal choice. As shown in Fig. 2, the penultimate layer consistently achieves the best performance across tasks. Notably, it outperforms other deep layers on General, OCR, and vision-centric tasks. This superiority stems from its ability to retain rich visual information while maintaining strong text alignment, which is second only to the final layer. Such a balance offers an optimal trade-off between visual expressiveness and semantic alignment, making it particularly well-suited for multimodal tasks. 

## 4.2 Deep-to-Shallower Layer Comparison

Afterwards, we investigate the effects of using shallow and middle layers in MLLMs. We highlight the following key observations. 

Deep layers are essential for OCR. As shown in Fig. 1, shallow layers provide negligible text information for the LLM. A clear boundary exists between the shallow and middle layers, with layer 12 marking the transition point. Layers before this point fail to contribute meaningfully to text processing, while a sharp performance gain occurs immediately afterward. This might be attributed to two essential requirements for OCR tasks: 

1. Rich fine-grained visual features in visual representation: In OCR tasks, a strong perception of details is often required. Therefore, these fine-grained details must be embedded in the representation with sufficiently strong signals to be effectively utilized by the LLM. 

2. Well textually aligned visual features: Despite containing rich visual details, shallow layers lack intrinsic alignment with textual representations, limiting their usefulness for OCR tasks. Fig. 1a confirms this with notably low cosine similarity between shallow and deep layers. This discrepancy poses a challenge for the connector, which can only align features originating from the deep (text) space or adjacent middle layers. 

Limited impact of representation quality on cognitive tasks. Cognitive tasks, such as “Code Reasoning” and “Numerical Calculation”, require both perception and high-level reasoning capabilities. Interestingly, we observe that even the shallow layers, which generally yield lower quality visual representations can rival or even outperform deeper layers in tasks under the MME-Cognitive. layer 3 achieves superior performance on “Code Reasoning”, “Numerical Calculation’, and “Text Recognition” compared to both middle and deep layers. (see Tab.7 and Tab.9 in Appendix) Thesefindings suggest that,for cognitive tasks, visualfeature quality is not the primary limiting factor. 

Potential of middle layers We first conduct an investigation into the middle representation spaces. The performance of these two spaces, as shown in Tab. 1, several key insights emerge from the results: 

(1) The middle layer has the potential to perform best: Although the middle layer’s information has not been fully processed, it still achieves the best performance on one-third of the bench marks. Specifically, compared to the penultimate layer, layer 14 achieves a 20-point higher score on MME-Cognitive, layer 18 outperforms by 3% on CVBench, layer 17 surpasses by 1.4% on Ref-COCO, and layer 20 exceeds by 0.2% on GQA. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/0c52dca9-18d4-4e52-83c6-d86f87e7f8e1/f0f1ef574dfc3eb49fa21a4189ade8d34227e5930d9dd2a457f0c4e00bbfd0a0.jpg)



Figure 4: Radar charts comparing the performance of Layers 23 and 24 across four different tasks under three LLM scales: 1.4B, 2.7B, and 7B. The results consistently show that the penultimate layer outperforms thefinal layer in all tasks. This trend remains stable across different model scales.


(2) The middle layers generally perform better on vision-centric tasks: Fig. 3a illustrates the performance of different layers across subtasks in the MME dataset, showing that position and existence tasks benefit more from middle layer representations. As depicted in Fig. 3b, the penultimate layer achieves top performance in only three out of eleven subtasks, whereas the shallow and middle layers yield optimal results in seven. Similarly, in Fig. 3c, one-third of the best performing results, such as those in spatial relations, physical relations, and cross fine-grained perception originate from the shallow and middle layers. A comparable trend is observed in SEEDBench (Fig. 3d), where middle layers produce optimal results in nearly half of the subtasks, including Instance Attribute, Instance Location, Instance Interaction, and Text Recognition. 

The hallucination problem is more pronounced in shallow layers but is effectively mitigated in the middle layers. As shown in Tab. 1, POPE results indicate that hallucination issues are most prominent in the shallow representation space, with minimal variation between the middle and deep layers. Notably, in the middle representation space, half of the layers outperform the penultimate layer on the POPE. This phenomenon likely stems from the fact that the challenge of this task lies more in visual perception than in semantic comprehension. In Sec.5.3, we provide a detailed analysis showing that further experiments with larger LLMs consistently support this finding. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/0c52dca9-18d4-4e52-83c6-d86f87e7f8e1/bf5656b730c3b0092a358f43dfe66ae929282cc9bfe5b33278040b79a758acac.jpg)



Figure 5: Proportion of subtasks achieving their best performance at the penultimate layer on MME and SEEDBench, demonstrating a clear upward trend.


## 5 Effect of Data and Model Scale

To further assess the generality of our findings, we extend our experiments to larger model scales and training datasets in this section and analyze the resulting performance trends. 

## 5.1 Settings

More training Data. We investigate the impact of training data scales using the same training procedure detailed in Sec. 3. In the first stage, we use the LLaVA 558k dataset (Liu et al., 2023b). For the second stage, we evaluate three dataset configurations: (1) LLaVA 665K, (2) Cambrian-1 737K (Tong et al., 2024a), an expansion of the 665K dataset with additional OCR data, and (3) a custom 1M dataset that builds on the 737K dataset by incorporating data specifically curated for vision-centric tasks. The dataset composition can be found in Appendix B. 

Scaling LLM sizes. Building on the original 1.4B experiments, we extend our study to include MobileLLaMA (Chu et al., 2024) 2.7B and Vicuna v1.5 7B. These LLMs are selected due to their similar architectures, making them well-suited for investigating the impact of different LLM sizes. 

Due to computational constraints, we do not conduct a full layer-wise analysis across different data scales. Instead, leveraging insights from Sec 4.2, we select representative layers from the shallow (layer 3), middle (layer 18), and deep (layers 23 and 24) representation spaces to examine how variations in data scale affect model performance. 

## 5.2 Deep-to-Deep Layer Comparison

As shown in Tab. 2, our findings indicate that the key conclusions in Sec. 4.2 remain valid across different training data scales. As data scales up, we uncover the following key insights: 

<table><tr><td rowspan="2">Data Scale</td><td rowspan="2">Layers</td><td colspan="5">General</td><td colspan="2">OCR</td><td colspan="4">Vision-Centric</td><td>Hallu</td></tr><tr><td><eq>MME^P</eq></td><td><eq>MME^C</eq></td><td>MMB</td><td>SEEDB</td><td>GQA</td><td>TVQA</td><td>OCRB</td><td>CVB</td><td><eq>CVB^{2D}</eq></td><td><eq>CVB^{3D}</eq></td><td>RWQA</td><td>POPE</td></tr><tr><td rowspan="4">665k</td><td>3</td><td>742.7</td><td>219.3</td><td>0.17</td><td>25.00</td><td>41.76</td><td>8.10</td><td>28</td><td>42.69</td><td>37.89</td><td>47.50</td><td>36.08</td><td>72.33</td></tr><tr><td>18</td><td>1088.8</td><td>237.1</td><td>29.38</td><td>52.06</td><td>54.37</td><td>31.44</td><td>200</td><td>47.29</td><td>41.17</td><td>53.42</td><td>39.48</td><td>84.26</td></tr><tr><td>23</td><td>1142.8</td><td>245.0</td><td>35.31</td><td>52.84</td><td>54.61</td><td>33.73</td><td>233</td><td>44.26</td><td>38.02</td><td>50.50</td><td>45.36</td><td>84.00</td></tr><tr><td>24</td><td>1114.1</td><td>243.6</td><td>32.65</td><td>51.09</td><td>53.61</td><td>30.63</td><td>197</td><td>46.68</td><td>39.78</td><td>53.58</td><td>43.92</td><td>83.65</td></tr><tr><td rowspan="4">737k</td><td>3</td><td>845.9</td><td>225.4</td><td>0.26</td><td>26.33</td><td>44.12</td><td>8.34</td><td>27</td><td>42.36</td><td>35.81</td><td>48.92</td><td>37.52</td><td>74.98</td></tr><tr><td>18</td><td>1093.1</td><td>226.4</td><td>43.04</td><td>56.33</td><td>56.98</td><td>35.98</td><td>270</td><td>48.87</td><td>46.57</td><td>51.17</td><td>43.40</td><td>86.18</td></tr><tr><td>23</td><td>1163.7</td><td>230.0</td><td>48.37</td><td>55.55</td><td>56.77</td><td>36.41</td><td>265</td><td>48.09</td><td>47.59</td><td>48.58</td><td>41.83</td><td>86.22</td></tr><tr><td>24</td><td>1121.7</td><td>258.6</td><td>46.05</td><td>55.43</td><td>56.34</td><td>36.09</td><td>255</td><td>43.63</td><td>38.01</td><td>49.25</td><td>44.44</td><td>85.09</td></tr><tr><td rowspan="4">1M</td><td>3</td><td>871.8</td><td>215.4</td><td>13.40</td><td>40.21</td><td>45.67</td><td>8.03</td><td>26</td><td>43.04</td><td>37.74</td><td>48.33</td><td>39.08</td><td>74.08</td></tr><tr><td>18</td><td>1145.9</td><td>213.2</td><td>42.44</td><td>56.72</td><td>57.74</td><td>35.68</td><td>267</td><td>54.08</td><td>56.33</td><td>51.83</td><td>43.14</td><td>84.06</td></tr><tr><td>23</td><td>1214.4</td><td>249.3</td><td>52.92</td><td>58.58</td><td>57.91</td><td>37.24</td><td>263</td><td>53.48</td><td>53.96</td><td>53.00</td><td>43.27</td><td>84.03</td></tr><tr><td>24</td><td>1192.0</td><td>245.0</td><td>47.34</td><td>57.62</td><td>57.21</td><td>36.45</td><td>264</td><td>47.88</td><td>42.84</td><td>52.92</td><td>44.97</td><td>84.58</td></tr></table>


Table 2: Performance comparison of visual representations across different data scales, demonstrating the consistency of our key findings. Even as gains diminish with scaling, middle layers continue to exhibit unique strengths over deep layers in OCRBench, SEEDBench, GQA and CVBench.


The penultimate layer remains the optimal choice in deep space regardless of LLM size As shown in Fig. 4, the penultimate layer consistently outperforms the final layer across LLMs of 1.4B, 2.7B, and 7B, reinforcing our findings. This indicates that CLIP-ViT’s final-layer visual degradation, driven by its training paradigm, cannot be offset by a stronger LLM. 

Furthermore, increasing LLM size does not yield significant improvements in POPE performance, indicating that hallucination bottlenecks in MLLMs stem primarily from the quality of the visual representation. In contrast, vision-centric tasks benefit more from scaling LLM size, indicating that even for tasks grounded in visual understanding, strong perception alone does not suffice, as robust reasoning capabilities remain essential. 

The more data and the larger the model, the more the deep layers benefit. As both the training data scale and LLM size increase, a clear trend emerges in Fig. 5: the proportion of subtasks where the penultimate layer achieves the best performance consistently grows. 

Since the visual encoder remains frozen during training, this suggests that compared to the middle layers, fine-grained information is less explicitly preserved in the deep layers. In other words, finegrained details in the middle layers are more readily utilized by the LLM, whereas those in the deep layers are harder to extract, requiring larger amounts of data to activate the LLM to effectively capture these fine-grained features. 

## 5.3 Deep-to-Shallower Layer Comparison

We observe that while conclusions from small models may not fully generalize to larger ones, the main findings still hold, as detailed below: 

The potential of shallower layers persists across model and data scales The consistency of the previous conclusion is validated under larger training data and increased LLM size. As detailed in Appendix 9 and 10, under the 2.7B model, the penultimate layer fails to outperform shallower layers on several MME subtasks, such as Count, Position, and Existence. Similar patterns emerge in SEEDBench and persist with the 7B model, where shallower layers (e.g., Layer 18) achieve better results on tasks like Spatial Relation. On the 665K dataset, layer 18 outperforms the penultimate layer by up to 3% on CVBench and maintains a slight advantage on OCR and vision-centric tasks. This trend persists on the 737K dataset, where layer 18 continues to lead on SEEDBench and GQA. Although the performance gap narrows on the 1M dataset, the middle layer still surpasses the deep layer on both OCRBench and CVBench. 

Limited gains on OCR tasks despite data and LLM scaling. Layer 3 exhibits comparable performance on OCR tasks both before and after incorporating OCR-specific training data, and across models of different scales (2.7B vs. 7B). This suggests that increasing task-specific data alone cannot overcome the inherent limitations of shallow representations. The lack of improvement can be attributed to their poor alignment with the textural feature space required for OCR understanding. In contrast, the middle layers, though only partially aligned, still exhibit performance gains under additional supervision. 

## 6 Visual Feature Fusion

Building on the above-mentioned findings that deeper layers are not universally optimal and that shallower layers offer valuable complementary information, we explore the most effective way to enhance visual representation by combining visual features from multiple layers. To be specific, we employ a simple fusion strategy to merge features from different layers and conduct a preliminary study on various layer combinations, aiming to highlight the potential benefits of layer fusion. 

<table><tr><td rowspan="2">Models</td><td colspan="5">General</td><td colspan="2">OCR</td><td colspan="4">Vision-Centric</td><td colspan="2">Hallu</td></tr><tr><td><eq>MME^P</eq></td><td><eq>MME^C</eq></td><td>MMB</td><td>SEEDB</td><td>GQA</td><td>TextVQA</td><td>OCRB</td><td>CVB</td><td><eq>CVB^{2D}</eq></td><td><eq>CVB^{3D}</eq></td><td>RWQA</td><td>POPE</td><td>Win</td></tr><tr><td>Baseline(23)</td><td>1142.8</td><td>245.0</td><td>35.31</td><td>52.84</td><td>52.84</td><td>33.73</td><td>233</td><td>44.26</td><td>38.02</td><td>50.50</td><td>45.36</td><td>84.00</td><td>9/10</td></tr><tr><td>DenseConnector</td><td>1145.0</td><td>253.2</td><td>47.85</td><td>57.16</td><td>56.92</td><td>37.54</td><td>257</td><td>45.60</td><td>35.83</td><td>54.92</td><td>45.10</td><td>84.95</td><td>7/10</td></tr><tr><td>MMFuser</td><td>1149.5</td><td>238.9</td><td>49.65</td><td>56.21</td><td>56.59</td><td>35.43</td><td>245</td><td>45.70</td><td>36.89</td><td>54.50</td><td>44.83</td><td>84.53</td><td>8/10</td></tr><tr><td>Ours*</td><td>1157.2</td><td>236.1</td><td>49.22</td><td>57.23</td><td>57.35</td><td>37.70</td><td>265</td><td>44.56</td><td>36.53</td><td>52.58</td><td>45.75</td><td>84.82</td><td>-</td></tr></table>


Table 3: Study on different layer fusion strategies. “Ours” represents $\mathcal { L } _ { 5 } = 2 3 , 1 8 , 3$ . “Win” denotes the proportion of datasets where our method achieves superior performance. Our method outperforms DC and MMFuser on 7 and 8 out of 10 benchmarks.


## 6.1 Method

The equation below shows the simplest visual feature fusion mechanism, 

$$
f = \operatorname{Concat} \left(\mathcal {H} ^ {(i)} \mid i \in \mathcal {L}\right)\tag{1}
$$

where $\mathcal { L }$ denotes the set of selected layers, and each $\mathcal { H } ^ { ( i ) }$ represents the feature representation extracted from layer l with a dimension of $N \times D$ . Here, N is the number of visual tokens, and D is the feature dimensionality of each token. The concatenation function Concat( ) merges these representations along the feature dimension, producing an output $f$ of size $N \times ( D \times | \mathcal { L } | )$ ), where $| { \mathcal { L } } |$ denotes the number of concatenated layers. The resulting f is then fed into the Connector. Subsequently, we explore various layer combinations. 

## 6.2 Exp-I: Ablation of Fusion Layer Selection

We select representative layers from shallow, middle, and deep layers for the fusion ablation study to preliminarily explore the effect of different representation spaces in fusion methods. We systematically construct different layer combinations to analyze their impact on fusion performance. 

Multiple stages bring generalization: Fig. 6 illustrates six different configurations, ranging from using only the end stage to incorporating all stages. Compared to two-layer fusion $( \mathcal { L } _ { 1 } )$ and three-layer fusion covering two stages $( \mathcal { L } _ { 3 } )$ , three-layer fusion $( \mathcal { L } _ { 2 , 3 } )$ that spans all stages $( \mathcal { L } _ { 2 }$ and Ours {23, 3, 18}) leads to more consistent performance improvements. Notably, $\mathcal { L } _ { 2 }$ performs worse than Ours {23, 3, 18} on OCR tasks. This is likely because the first layer feature is too raw, making them less suitable for extracting the low-level visual cues required in OCR. By incorporating more stable representations from multiple stages, models can achieve the most robust performance across tasks. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/0c52dca9-18d4-4e52-83c6-d86f87e7f8e1/940a6cd8cb5ff8478dd5f0507a93749e9bad5733c1b877a4ef2e8f2887ec91e5.jpg)



Figure 6: Performance comparison of different layer fusion combinations on four tasks. $\mathcal { L } _ { 1 } { - } \mathcal { L } _ { 4 }$ denote representative strategies for layer selection. “Ours” is $\mathcal { L } = \{ 2 3 , \mathsf { \hat { 3 } } , 1 8 \}$


## 6.3 Exp-II: Fusion Method Comparison

To investigate the impact of different layer fusion strategies, we compare our selected layers {23, 3, 18} and a simple concatenation strategy, against other carefully designed fusion methods. 

Less is more: We compare two state-of-the-art ad-hoc methods, DenseConnector (DC) and MM-Fuser with our method. As shown in Tab. 3, our method outperforms DC and MMFuser on 7 and 8 across 10 widely used benchmarks, respectively. These results suggest that complex fusion strategies may be unnecessary, as the simplest concatenation already meets the performance requirements. 

## 7 Conclusion

This study presents a comprehensive layer-wise analysis, revealing that shallow and middle representation spaces can surpass the performance of deep layers. Evaluations across diverse data and model scales further substantiate our findings. Furthermore, we introduce a straightforward yet highly effective fusion strategy for visual feature integration, delivering substantial improvements over the baseline. Our findings offer a foundation for advancing future research in fusion methodologies. 

## Limitations

Due to the high computational cost of layer-wise analysis, we adopt a linear probing-inspired strategy: most experiments are conducted on the 1.4B model, with selective validation on the 2.7B and 7B variants. However, our study does not extend to larger-scale LLMs. In terms of visual encoders, we focus exclusively on CLIP-ViT-L/14, given its widespread adoption, and leave the exploration of alternative backbones to future work. 

Moreover, vision–language fusion strategies can be broadly classified into internal and external methods. Our analysis focuses only on external fusion approaches and does not directly compare them with internal alternatives. We ensure consistent experimental conditions across all settings, which enables a fair assessment of visual representation quality. However, our current design does not explore how different connector architectures may affect performance. We regard this as a valuable direction for future research. 

## Acknowledgement

We are grateful to EIT and IDT High Performance Computing Center for providing computational resources that greatly facilitated this research. This work was financially supported by the 2035 Key Research and Development Program of Ningbo City (Grant No. 2025Z034), the Pioneer and Leading Goose R&D Program of Zhejiang (Grant No. 2024C01110), and the National Natural Science Foundation of China (Grant No. 62472385). 

## References



Manoj Acharya, Kushal Kafle, and Christopher Kanan. 2019. Tallyqa: Answering complex counting questions. In Proceedings of the AAAI conference on artificial intelligence, volume 33, pages 8076–8084. 





Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, and 1 others. 2022. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35:23716– 23736. 





Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, and 1 others. 2023. Qwen technical report. arXiv preprint arXiv:2309.16609. 





Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, and 1 others. 2025. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923. 





Yue Cao, Yangzhou Liu, Zhe Chen, Guangchen Shi, Wenhai Wang, Danhuai Zhao, and Tong Lu. 2024. Mmfuser: Multimodal multi-layer feature fuser for fine-grained vision-language understanding. arXiv preprint arXiv:2410.11829. 





Kaibing Chen, Dong Shen, Hanwen Zhong, Huasong Zhong, Kui Xia, Di Xu, Wei Yuan, Yifei Hu, Bin Wen, Tianke Zhang, Changyi Liu, Dewen Fan, Huihui Xiao, Jiahong Wu, Fan Yang, Size Li, and Di Zhang. 2024a. Evlm: An efficient visionlanguage model for visual understanding. Preprint, arXiv:2407.14177. 





Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, and 1 others. 2024b. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24185–24198. 





Xiangxiang Chu, Limeng Qiao, Xinyu Zhang, Shuang Xu, Fei Wei, Yang Yang, Xiaofei Sun, Yiming Hu, Xinyang Lin, Bo Zhang, and Chunhua Shen. 2024. Mobilevlm v2: Faster and stronger baseline for vision language model. Preprint, arXiv:2402.03766. 





Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, Yunsheng Wu, and Rongrong Ji. 2024. Mme: A comprehensive evaluation benchmark for multimodal large language models. Preprint, arXiv:2306.13394. 





Yossi Gandelsman, Alexei A. Efros, and Jacob Steinhardt. 2024. Interpreting CLIP’s image representation via text-based decomposition. In The Twelfth International Conference on Learning Representations. 





Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. 2017. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 6904–6913. 





Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxiao Dong, Ming Ding, and 1 others. 2024. Cogagent: A visual language model for gui agents. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14281–14290. 





Drew A. Hudson and Christopher D. Manning. 2019a. Gqa: A new dataset for real-world visual reasoning and compositional question answering. Preprint, arXiv:1902.09506. 





Drew A Hudson and Christopher D Manning. 2019b. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceed ings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6700–6709. 





Dongsheng Jiang, Yuchen Liu, Songlin Liu, Jin’e Zhao, Hao Zhang, Zhen Gao, Xiaopeng Zhang, Jin Li, and Hongkai Xiong. 2023. From clip to dino: Visual encoders shout in multi-modal large language models. arXiv preprint arXiv:2310.08825. 





Justin Johnson, Bharath Hariharan, Laurens Van Der Maaten, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick. 2017. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2901–2910. 





Kushal Kafle, Brian Price, Scott Cohen, and Christopher Kanan. 2018. Dvqa: Understanding data visualizations via question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5648–5656. 





Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. 2014. Referitgame: Referring to objects in photographs of natural scenes. In Proceedings ofthe 2014 conference on empirical methods in natural language processing (EMNLP), pages 787– 798. 





Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. 2016. A diagram is worth a dozen images. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11– 14, 2016, Proceedings, Part IV 14, pages 235–251. Springer. 





Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, and 1 others. 2017. Visual genome: Connecting language and vision using crowdsourced dense image annotations. International journal of computer vision, 123:32–73. 





Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. 2023a. Seed-bench: Benchmarking multimodal llms with generative comprehension. Preprint, arXiv:2307.16125. 





Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. 2023b. Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. In International conference on machine learning, pages 19730–19742. PMLR. 





Xu Li, Yi Zheng, Haotian Chen, Xiaolei Chen, Yuxuan Liang, Chenghang Lai, Bin Li, and Xiangyang Xue. 2026. Instruction-guided fusion of multi-layer visual features in large vision-language models. Pattern Recognition, 170:111932. 





Yanshu Li, Hongyang He, Yi Cao, Qisen Cheng, Xiang Fu, and Ruixiang Tang. 2025. M2iv: Towards efficient and fine-grained multimodal in-context learning in large vision-language models. arXiv preprint arXiv:2504.04633. 





Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. 2023c. Evaluating object hallucination in large vision-language models. Preprint, arXiv:2305.10355. 





Junyan Lin, Haoran Chen, Yue Fan, Yingqi Fan, Xin Jin, Hui Su, Jinlan Fu, and Xiaoyu Shen. 2025. Multi-layer visual feature fusion in multimodal llms: Methods, analysis, and best practices. Preprint, arXiv:2503.06063. 





Junyan Lin, Haoran Chen, Dawei Zhu, and Xiaoyu Shen. 2024. To preserve or to compress: An in-depth study of connector selection in multimodal large language models. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, pages 5666–5680, Miami, Florida, USA. Association for Computational Linguistics. 





Haotian Liu, Chunyuan Li, Yuheng Li, and Lee. 2023a. Improved baselines with visual instruction tuning. 





Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. 2024a. Llavanext: Improved reasoning, ocr, and world knowledge. 





Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023b. Visual instruction tuning. 





Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2024b. Visual instruction tuning. Advances in neural information processing systems, 36. 





Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. 2024c. Mmbench: Is your multi-modal model an all-around player? Preprint, arXiv:2307.06281. 





Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xucheng Yin, Cheng lin Liu, Lianwen Jin, and Xiang Bai. 2024d. Ocrbench: On the hidden mystery of ocr in large multimodal models. Preprint, arXiv:2305.07895. 





Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. 2022. A convnet for the 2020s. Preprint, arXiv:2201.03545. 





Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan L Yuille, and Kevin Murphy. 2016. Generation and comprehension of unambiguous object descriptions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 11–20. 





Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. 2019. Ok-vqa: A visual question answering benchmark requiring external knowl edge. In Conference on Computer Vision and Pattern Recognition (CVPR). 





Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. 2021. Docvqa: A dataset for vqa on document images. In Proceedings ofthe IEEE/CVF winter con ference on applications of computer vision, pages 2200–2209. 





Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and Anirban Chakraborty. 2019. Ocr-vqa: Visual question answering by reading text in images. In 2019 international conference on document analysis and recognition (ICDAR), pages 947–952. IEEE. 





Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, and 1 others. 2023. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193. 





Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, and 1 others. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR. 





Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi. 2022. A-okvqa: A benchmark for visual question answering using world knowledge. In European Conference on Computer Vision, pages 146–162. Springer. 





ShareGPT. 2023. https://sharegpt.com/. 





Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. 2018. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In Proceedings ofACL. 





Haozhan Shen, Kangjia Zhao, Tiancheng Zhao, Ruochen Xu, Zilun Zhang, Mingwei Zhu, and Jianwei Yin. 2024. Zoomeye: Enhancing multimodal llms with human-like zooming capabilities through tree-based image exploration. arXiv preprint arXiv:2411.16044. 





Oleksii Sidorov, Ronghang Hu, Marcus Rohrbach, and Amanpreet Singh. 2020. Textcaps: a dataset for image captioning with reading comprehension. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II 16, pages 742–758. Springer. 





Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. 2019. Towards vqa models that can read. Preprint, arXiv:1904.08920. 





Qi Sun, Marc Pickett, Aakash Kumar Nain, and Llion Jones. 2024. Transformer layers as painters. arXiv preprint arXiv:2407.09298. 





Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Yang, and 1 others. 2024a. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. arXiv preprint arXiv:2406.16860. 





Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. 2024b. Eyes wide shut? exploring the visual shortcomings of multimodal llms. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9568–9578. 





Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, and 1 others. 2024. Qwen2- vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191. 





Huanjin Yao, Wenhao Wu, Taojiannan Yang, YuXin Song, Mengxi Zhang, Haocheng Feng, Yifan Sun, Zhiheng Li, Wanli Ouyang, and Jingdong Wang. 2024. Dense connector for mllms. arXiv preprint arXiv:2405.13800. 





Licheng Yu, Patrick Poirson, Shan Yang, Alexander C Berg, and Tamara L Berg. 2016. Modeling context in referring expressions. In ECCV2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part II 14, pages 69–85. Springer. 





Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. 2023. Mm-vet: Evaluating large multimodal models for integrated capabilities. Preprint, arXiv:2308.02490. 





Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. 2023. Sigmoid loss for language image pre-training. Preprint, arXiv:2303.15343. 



## Appendix

We provide some additional information as supplementary material. This material is divided into four sections: 

• A detailed analysis and formal computation process of visual representations is provided in Appendix A. 

• Experimental details covering dataset composition, evaluation protocols, visualization of fusion performance, and analyses on model factors such as LLM choice, data scale, layer selection, and feature fusion are provided in Appendix B. 

• We provide additional details of our reasoning and rebuttal explanations in Appendix C. 

• Ethics statement is provided in Appendix D. 

## A Visual Representations

## A.1 Formal Computation of LRS

To examine the behavioral patterns of different visual layers, we introduce Layer-wise Representation Similarity (LRS) to analyze the relationships between various visual layers. 

Specifically, we compute the cosine similarity between hidden states of ViT layers, where values closer to 1 indicate higher similarity. The formalized division process is detailed in the following equation: 

We define the hidden state matrix H of CLIP-ViT as: 

$$
H \in \mathbb {R} ^ {L \times d}
$$

where L is the number of layers, and each row $H _ { i } \in \mathbb { R } ^ { d }$ represents the hidden state of the i-th layer. The cosine similarity matrix S is computed as: 

$$
S = \left| \frac {H H ^ {T}}{\| H \| _ {2} \| H ^ {T} \| _ {2}} \right|
$$

where $H H ^ { T }$ computes the pairwise dot prod ucts between layer hidden states. The denominator normalizes the values using the $L _ { 2 } .$ -norm, defined as: 

$$
\| H _ {i} \| _ {2} = \sqrt {\sum_ {k = 1} ^ {d} (H _ {i} ^ {(k)}) ^ {2}}
$$

ensuring that values lie within [ 1, 1]. The absolute value guarantees that all elements are within the range [0, 1]. Then, we compute the average of S across four tasks. 

We also explore the variations in visual representation spaces across four tasks. As illustrated in Figure 7, the partitioning of visual representations is minimally influenced by the nature of the tasks. In other words, the shallow, middle, and deep representation spaces exhibit remarkable stability, maintaining consistent structures across various tasks. 

## A.2 Replacing the Penultimate Layer

As shown in Table 4, when the penultimatelayer visual representations are replaced by those from other layers of the visual encoder without additional training, layers 20 to 24 (belonging to the deep representation space) do not suffer catastrophic performance degradation. Moreover, among the two layers closest to the penultimate one, layer 22 exhibits more stable performance than the final layer. 

<table><tr><td>Layer</td><td>MME-P</td><td>MME-C</td><td>OCRB</td><td>TextVQA</td><td>RefCOCO</td></tr><tr><td>24</td><td>1153.5</td><td>306.0</td><td>266</td><td>41.16</td><td>47.56</td></tr><tr><td>23</td><td>1509.9</td><td>365.3</td><td>314</td><td>46.10</td><td>49.04</td></tr><tr><td>22</td><td>1451.1</td><td>366.7</td><td>304</td><td>44.76</td><td>47.46</td></tr><tr><td>21</td><td>1368.8</td><td>293.2</td><td>287</td><td>41.59</td><td>40.47</td></tr><tr><td>20</td><td>1259.2</td><td>265.7</td><td>271</td><td>39.11</td><td>44.31</td></tr><tr><td>19</td><td>1183.3</td><td>267.1</td><td>240</td><td>36.76</td><td>42.83</td></tr><tr><td>18</td><td>1083.7</td><td>237.8</td><td>205</td><td>32.18</td><td>36.04</td></tr><tr><td>17</td><td>993.6</td><td>255.7</td><td>156</td><td>27.92</td><td>31.07</td></tr><tr><td>16</td><td>901.0</td><td>256.7</td><td>116</td><td>23.37</td><td>19.96</td></tr><tr><td>15</td><td>790.0</td><td>253.9</td><td>94</td><td>17.96</td><td>14.85</td></tr></table>


Table 4: Performance metrics across different layers on various benchmarks for non-training methods are presented. Specifically, MME-P denotes MME Perception, MME-C corresponds to MME Cognition, and OCRB represents OCR-Bench. The performance on RefCOCO is evaluated using Intersection over Union (IOU) as the metric.


## B Experiment Details

## B.1 Composition of Three Scale Datasets

As shown in Table 5, the following datasets are incorporated to enhance the model’s capabilities across multiple multimodal tasks, with all datasets accessed and utilized strictly under their licenses. 

• AI2D (Allen Institute for AI Diagram Dataset) (Kembhavi et al., 2016) AI2D is designed for visual reasoning and diagram understanding, featuring annotated diagrams with textual descriptions and Q&A pairs. It is particularly useful for multimodal reasoning and visual question answering (VQA) tasks. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/0c52dca9-18d4-4e52-83c6-d86f87e7f8e1/00ae7e1f235c718dd298d23d20847884940da006fb538585b218186073fe84d4.jpg)



(a) General


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/0c52dca9-18d4-4e52-83c6-d86f87e7f8e1/4079408290d3310c54605217e759244d39f5afbb2348260aeb396379b6829b52.jpg)



(b) OCR


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/0c52dca9-18d4-4e52-83c6-d86f87e7f8e1/26e4a3bf0054b10d04f49c45677a05ee45ab24c6bb74ce18b8eb457da507ec66.jpg)



(c) CV-Centric


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/0c52dca9-18d4-4e52-83c6-d86f87e7f8e1/10dcfa51e877c533211d81ecdfae53998ee9d9bcab2324101a619368acb498d1.jpg)



(d) Hallucination



Figure 7: A visualization of the average cosine similarity of visual representations across different layers in CLIP-ViT for four tasks, namely General, OCR, CV-Centric, and Hallucination. Values closer to 1 indicate greater similarity.


<table><tr><td>Data</td><td>Size</td></tr><tr><td>LLaVA (Liu et al., 2024b)</td><td>158K</td></tr><tr><td>+ ShareGPT (ShareGPT, 2023)</td><td>40K</td></tr><tr><td>+ VQAv2 (Goyal et al., 2017)</td><td>83K</td></tr><tr><td>+ GQA (Hudson and Manning, 2019b)</td><td>72K</td></tr><tr><td>+ OKVQA (Marino et al., 2019)</td><td>9K</td></tr><tr><td>+ OCRVQA (Mishra et al., 2019)</td><td>80K</td></tr><tr><td>+ A-OKVQA (Schwenk et al., 2022)</td><td>66K</td></tr><tr><td>+ TextCaps (Sidorov et al., 2020)</td><td>22K</td></tr><tr><td>+ RefCOCO (Kazemzadeh et al., 2014; Mao et al., 2016)</td><td>48K</td></tr><tr><td>+ VG (Krishna et al., 2017)</td><td>86K</td></tr><tr><td>LLaVA-1.5 (Liu et al., 2023a)</td><td>665K</td></tr><tr><td>+ AI2D (Kembhavi et al., 2016)</td><td>16K</td></tr><tr><td>+ DocVQA (Mathew et al., 2021)</td><td>15K</td></tr><tr><td>+ DVQA (Kafle et al., 2018)</td><td>13K</td></tr><tr><td>Cambrian-737k (Tong et al., 2024a)</td><td>737K</td></tr><tr><td>+ CLEVR (Johnson et al., 2017)</td><td>215k</td></tr><tr><td>+ TallyQA (Acharya et al., 2019)</td><td>77K</td></tr><tr><td>Customized-1M</td><td>1M</td></tr></table>


Table 5: The mixture detail of fine-tuning dataset for LLaVA-1.5 665K, Cambrian-1 737K and customized 1M.


• DocVQA (Document Visual Question Answering) (Mathew et al., 2021) DocVQA focuses on visual question answering over document images, where questions pertain to scanned documents, OCR-recognized text, and textual reasoning. This dataset is valuable for document comprehension, text recognition, and multimodal reasoning. 

• DVQA (Diagrammatic Visual Question Answering) (Kafle et al., 2018) DVQA is designed for visual question answering over diagrams and charts, covering questions related to bar charts, pie charts, and scientific illustrations. It evaluates the model’s ability to read structured visual information and perform reasoning based on graphical representations. 

• CLEVR (Compositional Language and Elementary Visual Reasoning) (Johnson et al., 2017) CLEVR is a synthetic dataset for visual reasoning, containing 3D-rendered scenes with structured questions that require reasoning based on attributes, object relationships, and compositional logic. It is widely used to assess a model’s capability in compositional and multi-step reasoning. 

• TallyQA (Acharya et al., 2019) TallyQA is a dataset specifically designed for object counting tasks, where questions require the model to accurately count objects in an image. It evaluates the model’s ability to attend to relevant objects, integrate global and local in formation, and perform numerical reasoning. 

## B.2 Evaluation Metrics

We provide a comprehensive explanation of the evaluation methods, categorizing them into three distinct types based on the evaluation metrics: 

• For benchmarks such as MME-Perception, MME-Cognition, OCRBench, and MMVet, we adopt the common approach of directly using the dataset-defined scores. We follow this established approach to maintain consistency and comparability in evaluations. 

• Using Accuracy directly as the evaluation metric. This applies to benchmarks such as MMBench, SEEDBench, GQA, TextVQA, CVBench, RealworldQA, and POPE. 

• In evaluating the RefCOCO dataset, we use CIDEr (Consensus-based Image Description Evaluation) as the primary evaluation metric. 

To facilitate evaluation, we use lmms-eval as our primary evaluation tool. Evaluations must be conducted on the official platform for the MMVet dataset by uploading the necessary data. Regarding the CVBench 3D tasks, where models generally exhibit weaker instruction-following performance, we employ the DeepSeek API as the judge. This tool provides results consistent with GPT-4o but is significantly more cost-effective. 

## B.3 The impact of LLMs

Additional experiments are conducted on different sizes of large language models to investigate their impact on visual information processing. We validate our conclusions on LLMs of 2.7b and 7b sizes. Due to computational resource constraints, we selected representative layers from the three representation spaces to conduct experiments on subtasks of MME and SEEDBench. As shown in Table 6, the penultimate layer does not consistently achieve the best performance on MME. The commonly used penultimate layer achieves optimal performance on 6 out of 14 subtasks, while other layers, such as Layers 3, 18, and 24, demonstrate superior performance on the remaining subtasks. This observation aligns with prior findings, suggesting that middle layers can exhibit superior performance over deeper layers on certain CV-centric tasks. Notably, layer 18 outperforms the penultimate layer in tasks such as Count, Position, and Existence. 

As illustrated in Table 7, while performance varies slightly across the subtasks of SEEDBench, the penultimate layer achieves the best performance on only 3 out of 9 subtasks. These results provide strong empirical evidence that shallow and middle layers can outperform deeper layers on specific subtasks. 

## B.4 The impact of Data Scale

The conclusion remains valid across different data scales. Under the 737k data scale, half of the subtasks in the MME dataset achieve optimal performance using the penultimate layer. However, for tasks like Count, Position, and Existence, the middle visual representation layer (Layer 18) demonstrates either superior or comparable performance. Similarly, results under the 1M data scale also show that half of the optimal performances are achieved on layers other than the penultimate one. 

The results for SEEDBench subtasks, as presented in Table 10, further support this observation. At the 737k data scale, Layer 18 from the middle representation space achieves the best performance on 5 out of 9 subtasks, while the penultimate layer excels in only 3 subtasks. Likewise, under the 1M data scale, half of the subtasks continue to achieve their best performance on layers other than the penultimate one. These findings consistently demonstrate across varying data scales that shallow and middle layers have the potential to outperform deep layers in certain scenarios. 

## B.5 Layer Selections and Feature Fusion

In the shallow layer, we select layers 1 and 3 as representatives. Layer 1, being the most chaotic, primarily captures early-stage visual features, while layer 3 is still in a chaotic state but performs relatively well. In the middle layer, we choose layers 18 and 17, as they achieve the first and second-best performance within this representation space. For the deep layer, we select layer 23, as it demonstrates the highest overall performance. The baseline configuration considers only layer 23 as the visual representation. 

Full version of comparison study as shown in Figure 8. We evaluate four state-of-the-art fusion methods, including three from DenseConnector (DC) (Yao et al., 2024) and one from MMFuser (Cao et al., 2024). As shown in Table 3, our method outperforms STI, SCI, and DCI on 8, 7, and 7 benchmarks, respectively. Compared to MMFuser (Cao et al., 2024), our approach demonstrates superior performance on 8 benchmarks. These results highlight the significant potential of visual feature fusion strategies in enhancing MLLMs and offering guidance for developing future fusion strategies. 

<table><tr><td>Model Size</td><td colspan="4">2.7b</td><td colspan="4">7b</td></tr><tr><td>Layers</td><td>3</td><td>18</td><td>23</td><td>24</td><td>3</td><td>18</td><td>23</td><td>24</td></tr><tr><td>Code Reasoning</td><td>52.50</td><td>47.50</td><td>47.50</td><td>40.00</td><td>50.00</td><td>40.00</td><td>42.50</td><td>45.00</td></tr><tr><td>Artwork</td><td>53.00</td><td>65.00</td><td>65.75</td><td>64.50</td><td>50.00</td><td>69.25</td><td>71.00</td><td>70.75</td></tr><tr><td>Celebrity</td><td>46.76</td><td>49.12</td><td>64.12</td><td>58.82</td><td>51.76</td><td>59.41</td><td>74.71</td><td>74.41</td></tr><tr><td>Numerical Calculation</td><td>50.00</td><td>50.00</td><td>42.50</td><td>25.00</td><td>47.50</td><td>45.00</td><td>37.50</td><td>37.50</td></tr><tr><td>Text Translation</td><td>50.00</td><td>50.00</td><td>50.00</td><td>50.00</td><td>65.00</td><td>65.00</td><td>47.50</td><td>67.50</td></tr><tr><td>Count</td><td>50.00</td><td>65.00</td><td>61.67</td><td>58.33</td><td>56.67</td><td>85.00</td><td>85.00</td><td>80.00</td></tr><tr><td>Color</td><td>53.33</td><td>83.33</td><td>86.67</td><td>88.33</td><td>78.33</td><td>91.67</td><td>91.67</td><td>91.67</td></tr><tr><td>Commonsense Reasoning</td><td>52.86</td><td>60.71</td><td>64.29</td><td>62.14</td><td>57.86</td><td>69.29</td><td>73.57</td><td>72.86</td></tr><tr><td>Position</td><td>48.33</td><td>71.67</td><td>71.67</td><td>71.67</td><td>61.67</td><td>71.67</td><td>75.00</td><td>80.00</td></tr><tr><td>OCR</td><td>50.00</td><td>67.50</td><td>72.50</td><td>65.00</td><td>55.00</td><td>77.50</td><td>75.00</td><td>70.00</td></tr><tr><td>Landmark</td><td>59.50</td><td>75.25</td><td>80.25</td><td>76.75</td><td>66.25</td><td>78.00</td><td>86.00</td><td>84.50</td></tr><tr><td>Scene</td><td>73.75</td><td>87.75</td><td>87.75</td><td>89.00</td><td>80.00</td><td>85.50</td><td>85.50</td><td>87.50</td></tr><tr><td>Existence</td><td>83.33</td><td>98.33</td><td>96.67</td><td>95.00</td><td>81.67</td><td>96.67</td><td>96.67</td><td>95.00</td></tr><tr><td>Posters</td><td>37.41</td><td>59.18</td><td>65.65</td><td>64.29</td><td>51.02</td><td>74.83</td><td>81.63</td><td>83.33</td></tr></table>


Table 6: Performance of LLaVA architectures with 2.7B and 7B LLMs on MME subtasks, evaluated across four layers from three representative spaces.


<table><tr><td>Model Size</td><td colspan="4">2.7b</td><td colspan="4">7b</td></tr><tr><td>Layers</td><td>3</td><td>18</td><td>23</td><td>24</td><td>3</td><td>18</td><td>23</td><td>24</td></tr><tr><td>Scene Understanding</td><td>36.04</td><td>68.84</td><td>68.87</td><td>69.79</td><td>50.70</td><td>73.50</td><td>73.91</td><td>73.59</td></tr><tr><td>Instance Identity</td><td>32.33</td><td>62.26</td><td>62.92</td><td>62.59</td><td>41.40</td><td>67.78</td><td>70.29</td><td>70.56</td></tr><tr><td>Instance Attribute</td><td>40.35</td><td>62.19</td><td>59.07</td><td>60.46</td><td>50.48</td><td>69.09</td><td>68.70</td><td>68.21</td></tr><tr><td>Instance Location</td><td>37.53</td><td>52.25</td><td>49.69</td><td>53.78</td><td>43.46</td><td>61.04</td><td>61.45</td><td>59.71</td></tr><tr><td>Instance Counting</td><td>25.70</td><td>43.07</td><td>47.57</td><td>45.20</td><td>33.02</td><td>56.89</td><td>57.13</td><td>57.29</td></tr><tr><td>Spatial Relation</td><td>33.03</td><td>42.47</td><td>40.64</td><td>43.99</td><td>41.55</td><td>52.05</td><td>49.62</td><td>51.45</td></tr><tr><td>Instance Interaction</td><td>34.02</td><td>64.95</td><td>54.64</td><td>64.95</td><td>48.45</td><td>63.92</td><td>67.01</td><td>71.13</td></tr><tr><td>Visual Reasoning</td><td>35.05</td><td>67.07</td><td>72.51</td><td>72.21</td><td>53.47</td><td>75.23</td><td>78.85</td><td>77.04</td></tr><tr><td>Text Recognition</td><td>44.71</td><td>21.18</td><td>21.18</td><td>24.71</td><td>38.82</td><td>34.12</td><td>47.06</td><td>43.53</td></tr></table>


Table 7: Performance of LLaVA architectures with 2.7B and 7B LLMs on SEEDBench subtasks, evaluated across four layers from three representative spaces


## C Detailed Discussion

## C.1 Alignment Guideline

What defines a good visual representation in multimodal models? Firstly, an ideal visual representation must simultaneously provide rich visual information and effectively align with textual modal. An MLLM can only correctly answer queries when the information corresponding to the given instruction is explicitly embedded within the visual representation. However, when the visual representation fails to deliver the necessary information, the model becomes prone to hallucination problems. Secondly, alignment with the textual modality is essential for a large language model to understand and process information from a different modality. This alignment ensures that the rich visual content is effectively leveraged. In a word, the CLIP series models currently offer the best trade-off between these two dimensions. 

## C.2 Comparison with Prior Work

Although (Liu et al., 2023b; Yao et al., 2024; Cao et al., 2024; Li et al., 2026) have briefly considered features from different layers, none have conducted a systematic or in-depth analysis of layer selection or the functional roles of visual layers in MLLMs. 

• (Liu et al., 2023b) presents a limited comparison between the penultimate and final layers on ScienceQA, without broader analysis. In contrast, we generalize this observation across over 10 benchmarks and various model and data scales, and offer an interpretation in Section 4.1 grounded in $\mathrm { C L I P } \mathrm { s }$ contrastive learning objective. 

<table><tr><td rowspan="2">Models</td><td colspan="5">General</td><td colspan="2">OCR</td><td colspan="4">Vision-Centric</td><td>Hallu</td><td rowspan="2">Win</td></tr><tr><td><eq>MME^P</eq></td><td><eq>MME^C</eq></td><td>MMB</td><td>SEEDB</td><td>GQA</td><td>TextVQA</td><td>OCRB</td><td>CVB</td><td><eq>CVB^{2D}</eq></td><td><eq>CVB^{3D}</eq></td><td>RWQA</td><td>POPE</td></tr><tr><td>Baseline(23)</td><td>1142.8</td><td>245.0</td><td>35.31</td><td>52.84</td><td>52.84</td><td>33.73</td><td>233</td><td>44.26</td><td>38.02</td><td>50.50</td><td>45.36</td><td>84.00</td><td>9/10</td></tr><tr><td>+ 18</td><td><eq>1148.5^{5.7\uparrow}</eq></td><td><eq>228.9^{16.1\downarrow}</eq></td><td><eq>46.91^{11.6\uparrow}</eq></td><td><eq>57.01^{4.2\uparrow}</eq></td><td><eq>56.80^{4\uparrow}</eq></td><td><eq>37.66^{3.9\uparrow}</eq></td><td><eq>273^{40\uparrow}</eq></td><td><eq>44.73^{0.5\uparrow}</eq></td><td><eq>35.79^{2.2\downarrow}</eq></td><td><eq>53.67^{3.2\uparrow}</eq></td><td><eq>45.49^{0.1\uparrow}</eq></td><td><eq>84.51^{0.5\uparrow}</eq></td><td>8/10</td></tr><tr><td>+ 1+18</td><td><eq>1155.4^{12.6\uparrow}</eq></td><td><eq>246.8^{1.8\uparrow}</eq></td><td><eq>48.54^{13.2\uparrow}</eq></td><td><eq>56.75^{3.9\uparrow}</eq></td><td><eq>56.68^{3.8\uparrow}</eq></td><td><eq>36.53^{2.8\uparrow}</eq></td><td><eq>236^{3\uparrow}</eq></td><td><eq>45.65^{1.4\uparrow}</eq></td><td><eq>36.21^{1.8\downarrow}</eq></td><td><eq>55.08^{4.6\uparrow}</eq></td><td><eq>46.93^{1.6\uparrow}</eq></td><td><eq>84.56^{0.6\uparrow}</eq></td><td>7/10</td></tr><tr><td>+ 17+18</td><td><eq>1182.5^{39.7\uparrow}</eq></td><td><eq>220.7^{24.3\downarrow}</eq></td><td><eq>48.80^{13.5\uparrow}</eq></td><td><eq>56.68^{3.8\uparrow}</eq></td><td><eq>56.48^{3.6\uparrow}</eq></td><td><eq>38.29^{4.6\uparrow}</eq></td><td><eq>263^{30\uparrow}</eq></td><td><eq>45.38^{1.1\uparrow}</eq></td><td><eq>36.25^{1.8\downarrow}</eq></td><td><eq>54.50^{4\uparrow}</eq></td><td><eq>44.71^{0.7\downarrow}</eq></td><td><eq>85.50^{1.5\uparrow}</eq></td><td>6/10</td></tr><tr><td>DC-STI</td><td><eq>1142.4^{0.4\downarrow}</eq></td><td><eq>218.9^{26.1\downarrow}</eq></td><td><eq>48.02^{12.7\uparrow}</eq></td><td><eq>57.23^{4.4\uparrow}</eq></td><td><eq>56.86^{4.0\uparrow}</eq></td><td><eq>36.42^{2.7\uparrow}</eq></td><td><eq>226^{7\downarrow}</eq></td><td><eq>43.83^{0.4\downarrow}</eq></td><td><eq>35.10^{2.9\downarrow}</eq></td><td><eq>52.58^{2.1\uparrow}</eq></td><td><eq>44.44^{0.9\downarrow}</eq></td><td><eq>86.38^{2.4\uparrow}</eq></td><td>8/10</td></tr><tr><td>DC-SCI*</td><td><eq>1166.5^{23.7\uparrow}</eq></td><td><eq>241.8^{3.2\downarrow}</eq></td><td><eq>48.71^{13.4\uparrow}</eq></td><td><eq>57.26^{4.4\uparrow}</eq></td><td><eq>56.61^{3.8\uparrow}</eq></td><td><eq>36.70^{3.0\uparrow}</eq></td><td><eq>241^{8\uparrow}</eq></td><td><eq>43.19^{1.1\downarrow}</eq></td><td><eq>34.96^{3.1\downarrow}</eq></td><td><eq>51.42^{0.9\uparrow}</eq></td><td><eq>44.44^{0.9\downarrow}</eq></td><td><eq>84.45^{0.5\uparrow}</eq></td><td>7/10</td></tr><tr><td>DC-DCI</td><td><eq>1145.0^{2.2\uparrow}</eq></td><td><eq>253.2^{8.2\uparrow}</eq></td><td><eq>47.85^{12.5\uparrow}</eq></td><td><eq>57.16^{4.3\uparrow}</eq></td><td><eq>56.92^{4.1\uparrow}</eq></td><td><eq>37.54^{3.8\uparrow}</eq></td><td><eq>257^{24\uparrow}</eq></td><td><eq>45.60^{1.3\uparrow}</eq></td><td><eq>35.83^{2.2\downarrow}</eq></td><td><eq>54.92^{4.4\uparrow}</eq></td><td><eq>45.10^{0.3\downarrow}</eq></td><td><eq>84.95^{1.0\uparrow}</eq></td><td>7/10</td></tr><tr><td>MMFuser</td><td><eq>1149.5^{6.7\uparrow}</eq></td><td><eq>238.9^{6.1\downarrow}</eq></td><td><eq>49.65^{14.3\uparrow}</eq></td><td><eq>56.21^{3.4\uparrow}</eq></td><td><eq>56.59^{3.8\uparrow}</eq></td><td><eq>35.43^{1.7\uparrow}</eq></td><td><eq>245^{12\uparrow}</eq></td><td><eq>45.70^{1.4\uparrow}</eq></td><td><eq>36.89^{1.1\downarrow}</eq></td><td><eq>54.50^{4.0\uparrow}</eq></td><td><eq>44.83^{0.5\downarrow}</eq></td><td><eq>84.53^{0.5\uparrow}</eq></td><td>8/10</td></tr><tr><td>Ours*</td><td><eq>1157.2^{14.4\uparrow}</eq></td><td><eq>236.1^{8.9\downarrow}</eq></td><td><eq>49.22^{13.9\uparrow}</eq></td><td><eq>57.23^{4.4\uparrow}</eq></td><td><eq>57.35^{4.5\uparrow}</eq></td><td><eq>37.70^{4.0\uparrow}</eq></td><td><eq>265^{32\uparrow}</eq></td><td><eq>44.56^{0.3\uparrow}</eq></td><td><eq>36.53^{1.5\downarrow}</eq></td><td><eq>52.58^{2.1\uparrow}</eq></td><td><eq>45.75^{0.4\uparrow}</eq></td><td><eq>84.82^{0.8\uparrow}</eq></td><td>-</td></tr></table>


Table 8: Study on different layer fusion strategies. The results reveal that nearly all fusion methods significantly outperform the baseline, with performance variations depending on the combination of different layers. $( \ast ) \cdot \mathrm { D C - S C I } ^ { \prime }$ is the same as $\mathcal { L } _ { 4 }$ and ’Ours’ represents $\mathcal { L } _ { 5 }$ . ’Win’ denotes the proportion of datasets where our method achieves superior performance.


<table><tr><td>Data Scale</td><td colspan="4">737k</td><td colspan="4">1M</td></tr><tr><td>Layers</td><td>3</td><td>18</td><td>23</td><td>24</td><td>3</td><td>18</td><td>23</td><td>24</td></tr><tr><td>Code Reasoning</td><td>50.00</td><td>47.50</td><td>45.00</td><td>42.50</td><td>47.50</td><td>47.50</td><td>45.00</td><td>47.50</td></tr><tr><td>Artwork</td><td>51.00</td><td>59.25</td><td>64.50</td><td>61.75</td><td>53.75</td><td>65.00</td><td>68.00</td><td>66.50</td></tr><tr><td>Celebrity</td><td>48.82</td><td>55.88</td><td>64.12</td><td>62.65</td><td>52.35</td><td>62.06</td><td>68.53</td><td>65.59</td></tr><tr><td>Numerical Calculation</td><td>47.50</td><td>37.50</td><td>35.00</td><td>47.50</td><td>50.00</td><td>30.00</td><td>45.00</td><td>47.50</td></tr><tr><td>Text Translation</td><td>50.00</td><td>50.00</td><td>50.00</td><td>50.00</td><td>50.00</td><td>50.00</td><td>50.00</td><td>50.00</td></tr><tr><td>Count</td><td>55.00</td><td>55.00</td><td>55.00</td><td>50.00</td><td>58.33</td><td>58.33</td><td>60.00</td><td>60.00</td></tr><tr><td>Color</td><td>60.00</td><td>75.00</td><td>76.67</td><td>75.00</td><td>61.67</td><td>78.33</td><td>80.00</td><td>80.00</td></tr><tr><td>Commonsense Reasoning</td><td>54.29</td><td>56.43</td><td>60.71</td><td>61.43</td><td>52.14</td><td>57.14</td><td>62.14</td><td>58.57</td></tr><tr><td>Position</td><td>51.67</td><td>70.00</td><td>63.33</td><td>70.00</td><td>50.00</td><td>70.00</td><td>73.33</td><td>70.00</td></tr><tr><td>OCR</td><td>50.00</td><td>55.00</td><td>57.50</td><td>52.50</td><td>55.00</td><td>52.50</td><td>55.00</td><td>55.00</td></tr><tr><td>Landmark</td><td>63.25</td><td>71.00</td><td>77.50</td><td>74.25</td><td>61.75</td><td>72.50</td><td>77.50</td><td>76.00</td></tr><tr><td>Scene</td><td>69.25</td><td>83.50</td><td>85.00</td><td>84.25</td><td>74.00</td><td>84.50</td><td>84.00</td><td>85.75</td></tr><tr><td>Existence</td><td>85.00</td><td>96.67</td><td>96.67</td><td>96.67</td><td>80.00</td><td>96.67</td><td>96.67</td><td>95.00</td></tr><tr><td>Posters</td><td>38.44</td><td>50.34</td><td>55.44</td><td>56.80</td><td>37.41</td><td>52.72</td><td>55.44</td><td>54.76</td></tr></table>


Table 9: Experimental results on MME subtasks under data scales of 737k and 1M, with the model settings consistent with those described in the main text.


• (Yao et al., 2024) uses attention maps to suggest that different layers attend to different regions of interest. However, such analysis is inherently qualitative and subjective, lacking a principled explanation of why these differences arise or how they relate to downstream tasks. 

• (Cao et al., 2024) proposes a shallow-to-deep fusion strategy but does not explain the underlying rationale for selecting specific layers, and its choices are based on empirical heuristics (e.g., 1/3, 2/3, 3/3), with no analysis of representations itself. 

• While (Li et al., 2026) focuses on fusion design, it lacks consideration of the role of visual layers in MLLMs. Our work provides novel insights, such as the unexpected correlation between inter-layer similarity and downstream performance. Moreover, they analyze only 1/5 of ViT layers, without considering data scale or model size. 

In contrast, our work is the first to systematically group ViT layers based on representation similarity, and to establish clear correlations between layer types and multimodal task performance across 60+ tasks and multiple model sizes (1.4B–7B) and data scales (665k-1M). 

## D Ethics Statement

This work made limited use of ChatGPT for language polishing and code debugging support. All scientific contributions, analyses, and conclusions are those of the authors. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/0c52dca9-18d4-4e52-83c6-d86f87e7f8e1/19fb03c033e6bfe82d939ac27196d5a0365aa02afe4a498139a6c27670d5d9c1.jpg)


<table><tr><td>Data Scale</td><td colspan="4">737k</td><td colspan="4">1M</td></tr><tr><td>Layers</td><td>3</td><td>18</td><td>23</td><td>24</td><td>3</td><td>18</td><td>23</td><td>24</td></tr><tr><td>Scene Understanding</td><td>26.85</td><td>67.35</td><td>68.24</td><td>67.57</td><td>44.02</td><td>65.71</td><td>68.97</td><td>67.57</td></tr><tr><td>Instance Identity</td><td>23.65</td><td>56.36</td><td>55.60</td><td>55.87</td><td>35.06</td><td>57.89</td><td>61.82</td><td>58.77</td></tr><tr><td>Instance Attribute</td><td>26.03</td><td>59.93</td><td>58.36</td><td>57.69</td><td>44.27</td><td>61.93</td><td>61.45</td><td>60.59</td></tr><tr><td>Instance Location</td><td>26.69</td><td>50.20</td><td>47.75</td><td>48.26</td><td>39.37</td><td>48.67</td><td>49.69</td><td>49.08</td></tr><tr><td>Instance Counting</td><td>27.34</td><td>41.93</td><td>40.13</td><td>41.19</td><td>32.04</td><td>42.38</td><td>44.91</td><td>45.97</td></tr><tr><td>Spatial Relation</td><td>29.68</td><td>41.86</td><td>38.81</td><td>38.96</td><td>37.90</td><td>42.92</td><td>39.57</td><td>39.57</td></tr><tr><td>Instance Interaction</td><td>21.65</td><td>46.39</td><td>51.55</td><td>49.48</td><td>38.14</td><td>50.52</td><td>51.55</td><td>52.58</td></tr><tr><td>Visual Reasoning</td><td>27.79</td><td>62.54</td><td>67.98</td><td>69.18</td><td>43.20</td><td>60.42</td><td>68.88</td><td>67.07</td></tr><tr><td>Text Recognition</td><td>21.18</td><td>34.12</td><td>48.24</td><td>43.53</td><td>41.18</td><td>17.65</td><td>55.29</td><td>42.35</td></tr></table>


Table 10: Experimental results on SEEDBench subtasks under data scales of 737k and 1M, with the model settings consistent with those described in the main text.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/0c52dca9-18d4-4e52-83c6-d86f87e7f8e1/534e10e972fe2b80252dc341daabfb1471d450d0432e347be02ffaa7ffbb6832.jpg)



what is written in the image?


Penultimate: the image reads "ula" on a white background. Others: the image is written in the form of a street sign, which reads "zula." 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/0c52dca9-18d4-4e52-83c6-d86f87e7f8e1/f4643c909a71e19ee0eb9aaf00045c221067f9cedeeb93fe6bd2c9cb82f8ce66.jpg)


What color does Moldova show in the graph? 

(a)OCR 

Penultimate: moldova shows up as a green line in the graph. Others: moldova shows up in the graph as purple. 

Users: 

In the picture, which direction is the cat facing? 

Penultimate: upward Others: facing the camera 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/0c52dca9-18d4-4e52-83c6-d86f87e7f8e1/77553d6cb1bb4d99303edadb06d849f74914f46231be34deb2c25f678acf8f1d.jpg)


How many apples are there in the image? And how many bananas are there? 

Penultimate: 2 apples and 1 bananas Others: 1 apples and 1 bananas (b)object_localization 


Figure 8: Case study illustrating four examples where the penultimate layer provides incorrect answers, but these errors can be resolved by using shallow and middle layers. In all four cases, Layer 18 of CLIP-ViT as the visual representation could successfully provide the correct answers.


<table><tr><td>Model</td><td>MMB</td><td>SEEDB</td><td>GQA</td><td>OCRB</td><td>RefCOCO</td></tr><tr><td>Baseline (Vicuna-7B)</td><td>65.1</td><td>65.2</td><td>63.2</td><td>320</td><td>52.3</td></tr><tr><td>DenseConnector (Yao et al., 2024)</td><td>63.3</td><td>66.6</td><td>62.9</td><td>323</td><td>-</td></tr><tr><td>MMFuser (Cao et al., 2024)</td><td>67.5</td><td>60.8</td><td>62.8</td><td>-</td><td>-</td></tr><tr><td>Instruction-Guided Fusion (Li et al., 2026)</td><td>66.9</td><td>68.3</td><td>63.1</td><td>-</td><td>-</td></tr><tr><td>Ours</td><td>67.4</td><td>67.7</td><td>63.5</td><td>325</td><td>54.0</td></tr></table>


Table 11: Performance comparison on 7B-scale LLMs. Dense2Connector and MMFuser are retrained under the same settings, while results of Instruction-Guided Fusion are copied from (Li et al., 2026). The results highlight the importance of visual layer selection.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/0c52dca9-18d4-4e52-83c6-d86f87e7f8e1/90213f8bc20d1f86f60cea9a059df7680057b7f4d498c2a7731b4e008a49e90f.jpg)



Figure 9: Heatmap showing the relative performance of CLIP-ViT layers 1-24 on MME subtasks. Consistent with the experimental settings in the main text, the results demonstrate that while deep layers generally achieve the best performance, shallow and middle layers can surpass deep layers on specific tasks, providing further support for our findings.


<table><tr><td>Index</td><td>MME</td><td>MMVet</td><td>MMBench</td><td>SEEDBench</td></tr><tr><td>1</td><td>Code Reasoning</td><td>Rec_OCR_Spat_Math</td><td>Attribute Comparison</td><td>Instance Attribute</td></tr><tr><td>2</td><td>Count</td><td>Rec_Spat</td><td>Attribute Recognition</td><td>Instance Location</td></tr><tr><td>3</td><td>Numerical Calculation</td><td>Rec_OCR_Know_Gen</td><td>Physical Property</td><td>Instance Interaction</td></tr><tr><td>4</td><td>Position</td><td>OCR_Spat</td><td>Spatial Relation</td><td>Text Recognition</td></tr><tr><td>5</td><td>Existence</td><td>OCR_Know_Spat</td><td>Image-Text Understanding</td><td>Instance Identity</td></tr><tr><td>6</td><td>Scene</td><td>Rec</td><td>Fine-grained Perception (Cross-instance)</td><td>Scene Understanding</td></tr><tr><td>7</td><td>OCR</td><td>OCR_Gen_Spat</td><td>Image Style</td><td>Instance Counting</td></tr><tr><td>8</td><td>Poster</td><td>Rec_Know_Gen</td><td>Physical Relation</td><td>Spatial Relation</td></tr><tr><td>9</td><td>Color</td><td>Rec_Know</td><td>Image Scene</td><td>Visual Reasoning</td></tr><tr><td>10</td><td>Celebrity</td><td>Rec_OCR_Gen_Spat</td><td>Celebrity Recognition</td><td>—</td></tr><tr><td>11</td><td>Landmark</td><td>OCR</td><td>Identity Reasoning</td><td>—</td></tr><tr><td>12</td><td>Artwork</td><td>Rec_OCR_Gen</td><td>Image Emotion</td><td>—</td></tr><tr><td>13</td><td>Commonsense Reasoning</td><td>Rec_OCR</td><td>Image Topic</td><td>—</td></tr><tr><td>14</td><td>—</td><td>—</td><td>Natural Relation</td><td>—</td></tr><tr><td>15</td><td>—</td><td>—</td><td>Object Localization</td><td>—</td></tr><tr><td>16</td><td>—</td><td>—</td><td>OCR</td><td>—</td></tr><tr><td>17</td><td>—</td><td>—</td><td>Social Relation</td><td>—</td></tr><tr><td>18</td><td>—</td><td>—</td><td>Attribute Recognition</td><td>—</td></tr><tr><td>19</td><td>—</td><td>—</td><td>Coarse Perception</td><td>—</td></tr><tr><td>20</td><td>—</td><td>—</td><td>Fine-grained Perception (Single-instance)</td><td>—</td></tr><tr><td>21</td><td>—</td><td>—</td><td>Relation Reasoning</td><td>—</td></tr><tr><td>22</td><td>—</td><td>—</td><td>Action Recognition</td><td>—</td></tr><tr><td>23</td><td>—</td><td>—</td><td>Function Reasoning</td><td>—</td></tr><tr><td>24</td><td>—</td><td>—</td><td>Future Prediction</td><td>—</td></tr><tr><td>25</td><td>—</td><td>—</td><td>Logical Reasoning</td><td>—</td></tr></table>


Table 12: Subtask-to-Index Mapping Table. Each column corresponds to the substask taxonomy defined in MME (Fu et al., 2024), MMVet (Yu et al., 2023), MMBench (Liu et al., 2024c) and SEEDBench (Li et al., 2023a).
