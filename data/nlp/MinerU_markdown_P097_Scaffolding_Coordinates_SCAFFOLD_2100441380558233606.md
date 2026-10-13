# Scaffolding Coordinates to Promote Vision-Language Coordination in Large Multi-Modal Models

Xuanyu Lei<sup>1,2</sup>, Zonghan Yang<sup>2</sup>, Xinrui Chen<sup>2</sup>, Peng Li<sup>1,3</sup>*, Yang Liu<sup>1,2,3,4</sup>* 

<sup>1</sup>Institute for AI Industry Research (AIR), Tsinghua University, Beijing, China <sup>2</sup>Dept. of Comp. Sci. & Tech., Institute for AI, Tsinghua University, Beijing, China <sup>3</sup>Shanghai Artificial Intelligence Laboratory, Shanghai, China <sup>4</sup>Jiangsu Collaborative Innovation Center for Language Competence, Jiangsu, China leixy24@mails.tsinghua.edu.cn 

lipeng@air.tsinghua.edu.cn, liuyang2011@tsinghua.edu.cn 

## Abstract

State-of-the-art Large Multi-Modal Models (LMMs) have demonstrated exceptional capabilities in vision-language tasks. Despite their advanced functionalities, the performances of LMMs are still limited in challenging scenarios that require complex reasoning with multiple levels of visual information. Existing prompting techniques for LMMs focus on either improving textual reasoning or leveraging tools for image preprocessing, lacking a simple and general visual prompting scheme to promote vision-language coordination in LMMs. In this work, we propose SCAFFOLD prompting that scaffolds coordinates to promote vision-language coordination. Specifically, SCAFFOLD overlays a dot matrix within the image as visual information anchors and leverages multi-dimensional coordinates as textual positional references. Extensive experiments on a wide range of challenging visionlanguage tasks demonstrate the superiority of SCAFFOLD over the textual Chain-of-Thought prompting. Our code is released in https: //github.com/THUNLP-MT/Scaffold. 

## 1 Introduction

Large Multi-Modal Models (LMMs) like GPT-4V (Achiam et al., 2023) and Gemini (Gemini et al., 2023) have demonstrated impressive zero-shot capabilities in processing diverse visual-language tasks. Leveraging the advanced reasoning ability of the language model component, early attempts have been made to deploy LMMs in realistic scenarios, such as autonomous driving (Wen et al., 2023) and anomaly detection (Cao et al., 2023). 

However, current LMMs display limited performance when conducting complex reasoning over multiple levels of visual information (Yang et al., 2023b; Wu et al., 2023a; Wu and Xie, 2023). For example, in a spatial reasoning task (Liu et al., 2023a), an LMM needs to verify or generate the statement by elucidating the relationship between different sources of visual information, and aligning its internal workings with textual expressions. Challenges for LMMs arise in orchestrating precise visual perception with accurate language understanding and generation. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/fa1f492d-3249-430e-bc39-42ccac2509e2/5ca78d4eb24bc52512295489600f19e608b049a6e0faa0f15a0417c192fcca79.jpg)



Figure 1: The overall framework of SCAFFOLD. SCAF-FOLD overlays a dot matrix onto the input image with Cartesian coordinates labeled aside the dots. The coordinates are also briefed in the textual prompt, which steers the LMM to leverage the dots on the image as a scaffold and promotes vision-language coordination.


To enhance vision-language coordination, prior efforts for LMMs can be divided into two categories: instruction tuning and prompting. Instruction tuning uses high-quality image-text pairs of either general purposes (Dai et al., 2023; Liu et al., 2023c,b; Xu et al., 2023) or specialized domains (Zhang et al., 2023; Chen et al., 2023a, 2024) to facilitate additional training of LMMs for improved performance. Nevertheless, instructiontuning consumes heavy loads of computing resources and thus sacrifices flexibility in methodology. Prompting for LMMs, on the contrary, steers the improved functioning of LMMs in a nonparametric manner. While techniques for language model prompting like Chain-of-Thought (Wei et al., 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/fa1f492d-3249-430e-bc39-42ccac2509e2/15ed112ea1c9181825b159dadcdcfb02c7fe686a4d85a176a329b3c1453e935d.jpg)



Figure 2: An example of SCAFFOLD in double-image vision-language tasks, with correct textual reasoning illustrated in green and incorrect textual reasoning illustrated in red. Note that only the original images and questions are visible to GPT-4V; SCAFFOLD adds the coordinates on images and the corresponding textual prompt guidance.


2022) apply to LMMs as well, the elicited intermediate reasoning steps mainly take place with the condition of the textual prompts (Wu et al., 2023b). As textual prompting techniques are actively being exploited, few endeavors have been made into visual prompting, which steers the precise visual perception of LMMs for vision-language coordination. 

The challenge of visual prompting for LMMs lies in the mismatch of semantic granularity between visual and textual information. While each word is explicitly separated in a textual sentence, different identities in an image are not isolated with clear boundaries. Recent works on visual prompting include leveraging tools to narrow the semantic granularity gap between visual and textual inputs. Mitra et al. (2023) treat the LMM itself as a scene graph extractor to generate visual information in the textual format. Self-operatingcomputer<sup>1</sup> project uses object detection models to draw bounding boxes for better grounding. Setof-mark prompting (Yang et al., 2023a) leverages advanced image segmentation models to construct object segmentation overlays on the input image. However, tool usage inevitably results in additional resource burdens and potentially erroneous information. Additional expenses from external model deployment and scenario-specific pipeline configurations limit the feasibility in practical applications of such methods. An alternative avenue of recent efforts is visual search (Wu and Xie, 2023; Nasiriany et al., 2024), where the solution to a complex visual task is cast as an iterative search process. Nevertheless, the iterative queries of LMMs throughout the search process entail considerable expenses, limiting the practical value. Therefore, it remains elusive whether a simple and general visual prompting scheme exists to promote vision-language coordination in LMMs. 

In this work, we present SCAFFOLD, a simple and versatile visual prompting scheme to promote the coordination between vision and language in LMMs. As shown in Fig 1, SCAFFOLD overlays a dot matrix onto the input image, and each dot is labeled with its multi-dimensional Cartesian coordinate. The dot matrix on the image forms the scaffold that indicates relative visual positions for LMMs. The overlaid coordinates are also included in the textual prompt, which explicitly strengthens the connection between visual and textual information for LMMs. The LMMs are thus steered to leverage the coordinates to solve different visionlanguage tasks. In this way, SCAFFOLD provides a scaffold to promote vision-language coordination in LMMs. Extensive experiments on spatial reasoning, compositional reasoning, fine-grained grounding, and hallucination benchmarks demonstrate the superiority of SCAFFOLD over GPT-4V with the textual CoT prompting. We also show that the performance of SCAFFOLD can be further enhanced with region cropping, which reveals the promising future of active perception enabled by SCAFFOLD. 

## 2 Related Work

Large Multi-Modal Models (LMMs). State-ofthe-art LMMs like GPT-4V (Achiam et al., 2023) and Gemini (Gemini et al., 2023) have excelled in general vision-language tasks (Wu et al., 2023a; Yang et al., 2023b; Fu et al., 2023b). The integration of visual capabilities in LMMs with advanced language proficiency and instruction-following skills pave the way for versatile visual interactive agents, both in digital (He et al., 2024; Zheng et al., 2024) and embodied environments (Wake et al., 2023; Chen et al., 2023b). 

GPT-4V Evaluation. As a leading LMM, GPT-4V (Achiam et al., 2023) has significantly expanded the boundaries of LMM capabilities, motivating researchers to systematically explore its strengths and weaknesses (Wu et al., 2023a; Yang et al., 2023b). Despite its proficiency, researchers have proposed challenging benchmarks that reveal large performance gap between GPT-4V and humans, including MMVP (Tong et al., 2024), MMMU (Yue et al., 2023), Mementos (Wang et al., 2024), V* Bench (Wu and Xie, 2023), Contextual (Wadhawan et al., 2024), etc. Extensive evaluations indicate that plenty of room exists for state-of-theart LMMs to improve their certain visual-language capabilities. 

Multi-Modal Prompting Methods. Prompting methods focus on unlocking model potentials by carefully constructing model inputs. Chain-of-Thought (Wei et al., 2022; Kojima et al., 2022, CoT) and its variants (Yao et al., 2023; Besta et al., 

2023) have successfully elicited reasoning capabilities in language models. However, in multimodal contexts such as compositional reasoning, the original CoT is less effective (Mitra et al., 2023). Consequently, numerous multi-modal prompting methods have been developed for specific visual capabilities. For instance, Compositional CoT (Mitra et al., 2023) for compositional reasoning, Spatial CoT (Chen et al., 2024) for spatial understanding, Set-of-Marks prompting (Yang et al., 2023a) for visual grounding. However, these methods tend to be tailored for specific capabilities, calling for simple and general visual prompting schemes. 

## 3 Methodology

In this section, we introduce SCAFFOLD prompting for vision-language coordination in LMMs. 

## 3.1 Visual Perspective of SCAFFOLD: Dot Matrices and Coordinates

Visually, we enhance each input image with a uniformly distributed rectangular dot matrix, where each dot is labeled with multi-dimensional coordinates. These dots serve as visual positional anchors, while their coordinates are utilized as textual references in textual responses. 

Visual Anchor Implementation. We select uniform rectangular dot matrices as our visual anchor due to their simplicity, flexibility in textual description, and potential adaptability to image sequences. With the size of the image as $H \times W$ , and the coordinate matrix containing $X \times Y$ dots in total, the horizontal interval between neighboring dots is set as $\frac { W } { Y + 1 }$ , and the vertical interval between neighboring dots is set as $\frac { H } { X + 1 }$ . Finally, we calculate the position of dot (x, y) as follows: 

$$
\operatorname{Position} (x, y) = \left(x \times \frac {H}{X + 1}, y \times \frac {W}{Y + 1}\right).\tag{1}
$$

Unlike grids, which divide images into separate regions and may disrupt continuous visual content, dot matrices offer a less intrusive overlay. We also incorporate the original image as the additional input to alleviate the potential interference between the visual content and the dot matrix to be overlaid. The dot matrix therefore serves as the information enhancement for vision-language coordination. 

Coordinates Implementation. For our approach, we use multi-dimensional Cartesian coordinates due to its simplicity and clarity. For a single image with an overlaid dot matrix of size $h \times w$ , we assign two-dimensional coordinates $( x , y )$ to each dot, representing its relative visual position. Here, the xcoordinate ascends from 1 to h within each column, while the y-coordinate ascends from 1 to w within each row. For image sequences, we extend these coordinates to three-dimensional (t, x, y). The tcoordinate remains constant within each image but increases sequentially across the sequence, allowing for differentiation between images and enhancing temporal perception. 

In comparison, we also consider other coordinate options and identify their limitations. Absolute pixel coordinates, for example, consume excessive space and are complicated to perceive and apply ac curately. Furthermore, one-dimensional Cartesian coordinates and alphabetic coordinates fall short in providing adequate positional information. 

Other Factors. 1. Matrix Size. The matrix should be visually clear and provide ample space for multidimensional coordinates. For simplicity, we employ a 6 × 6 matrix for general vision-language tasks. 2. Matrix Density. Without prior visual knowledge, we choose rectangular dot matrices with a uniform density for general vision-language tasks, providing LMMs equal assistance when reasoning across different regions. 3. Matrix Color. The coordinates are designed to be recognizable by LMMs using their OCR capabilities. Consequently, we color each dot in either black or white according to its contrast against the background. 

## 3.2 Textual Perspective of SCAFFOLD: Description and Guidelines

To complement the coordinates-overlaid visual inputs, we prepend textual guidance to task instructions for LMMs. This includes a brief description of the dot matrices and coordinates, accompanied by several general guidelines for their effective use, as detailed in Appendix A.1. The characteristics of these descriptions and guidelines are as follows: (1) Conciseness: The textual guidance is deliberately brief and clear, ensuring easy comprehension. (2) Generality: Designed to be universally applicable, these guidelines are not specific to any particular scenario, making them suitable for a wide range of vision-language tasks. (3) Extensibility: The guidelines are semantically independent, allowing for the seamless addition of more tailored instructions based on different scenarios. (4) Compositionality: The prepended texts can be easily combined with other prompting methods, such as zeroshot or compositional CoT (Kojima et al., 2022; Mitra et al., 2023). 

## 4 Experiments

To demonstrate the effectiveness of SCAFFOLD, we conduct extensive experiments on top of GPT-4V on a range of challenging vision-language tasks, including Spatial Reasoning, Compositional Reasoning, Fine-Grained Visual Understanding and Hallucination. Specifically, we perform systematic evaluations on 11 benchmarks. We set the temperature of GPT-4V as zero in our experiments. We also conduct extended studies of SCAFFOLD with other LMMs like Gemini (Gemini et al., 2023) in Sec. 4.4, and comparison with Set-of-Mark (SoM) prompting (Yang et al., 2023a) in Sec. 4.5. More experimental details can be found in Appendix B. 

## 4.1 Benchmarks

This subsection briefly introduces the benchmarks used for evaluation. Due to the limited budget, for some datasets, we sample a subset for experiments. Spatial Reasoning evaluates LMM capability to infer spatial relationships between objects. Selected benchmarks are as follows. 1. MME (Position split) (Fu et al., 2023a) is a subset of the MME comprehensive evaluation suite for LMMs to infer object positions. 2. Visual Spatial Reasoning (VSR) (Liu et al., 2023a) challenges LMMs with 66 types of spatial relations to verify spatial propositions. 3. EgoThink (Spatial split) (Cheng et al., 2023) tests the spatial reasoning ability of LMMs from a first-person perspective. 

Compositional Reasoning requires LMMs to identify object attributes and their interrelations. Selected benchmarks are as follows. 1. Winoground (Thrush et al., 2022) is a challenging benchmark that necessitates compositional reasoning of LMMs to match images with captions, reformulated as binary choice questions for our evaluation. 2. WHOOPS! VQA (Bitton-Guetta et al., 2023) involves compositional reasoning over commonsense-defying images. 3. CLEVR (Johnson et al., 2017) is designed for assessing compositional reasoning in program-generated scenes. 

Fine-Grained Visual Understanding requires LMMs to perform visual search and precisely perceive fine-grained visual details. Selected benchmarks are as follows. 1. V* Bench (Wu and Xie, 2023) requires LMMs to identify and reason with fine-grained visual details in high-resolution images. 2. Spotting Differences <sup>2</sup> is our newlycollected dataset challenging LMMs to find and pinpoint differences between two similar images, with further details in Appendix C.3. 

<table><tr><td>Crucial Capability</td><td>Dataset</td><td>Size</td><td>Metric</td><td>Standard</td><td>CoT</td><td>SCAFFOLD (Ours)</td></tr><tr><td rowspan="3">Spatial Reasoning</td><td>MME (Position)</td><td>60</td><td>Accuracy</td><td>51.7</td><td>51.7</td><td>75.0 (+23.3)</td></tr><tr><td>VSR</td><td>200</td><td>Accuracy</td><td>67.8</td><td>70.4</td><td>74.4 (+6.6)</td></tr><tr><td>EgoThink (Spatial)</td><td>50</td><td>LLM as Judge</td><td>66.0</td><td>74.0</td><td>76.0 (+10.0)</td></tr><tr><td rowspan="3">Compositional Reasoning</td><td>Winoground</td><td>100</td><td>Text Score</td><td>64.0</td><td>72.5</td><td>72.5 (+8.5)</td></tr><tr><td>WHOOPS! VQA</td><td>200</td><td>BEM Score</td><td>58.6</td><td>57.6</td><td>62.7 (+4.1)</td></tr><tr><td>CLEVR</td><td>200</td><td>LLM as Judge</td><td>43.5</td><td>43.0</td><td>48.0 (+4.5)</td></tr><tr><td rowspan="2">Fine-Grained Visual Understanding</td><td>V* Bench</td><td>238</td><td>Accuracy</td><td>27.2</td><td>30.8</td><td>44.6 (+17.4)</td></tr><tr><td>Spotting Differences</td><td>50</td><td>Accuracy</td><td>13.0</td><td>14.0</td><td>19.0 (+6.0)</td></tr><tr><td rowspan="3">Hallucination</td><td>POPE (Adversarial)</td><td>100</td><td>Accuracy</td><td>79.0</td><td>80.0</td><td>86.0 (+7.0)</td></tr><tr><td>HallusionBench (Hard)</td><td>504</td><td>Accuracy</td><td>45.6</td><td>48.8</td><td>53.0 (+7.4)</td></tr><tr><td>Mementos</td><td>100</td><td>LLM as Judge</td><td>33.3</td><td>33.5</td><td>36.1 (+2.8)</td></tr><tr><td>Overall</td><td>All</td><td>1,802</td><td>Average</td><td>50.0</td><td>52.4</td><td>58.8 (+8.8)</td></tr></table>


Table 1: Results of SCAFFOLD on 11 challenging vision-language benchmarks on top of GPT-4V, with the highest score bold.


<table><tr><td>Dataset</td><td>Metric</td><td>Standard</td><td>CoT</td><td>SCAFFOLD</td></tr><tr><td>WINO</td><td>Text Score</td><td>88.0</td><td>90.0</td><td>92.0</td></tr><tr><td>VSR</td><td>Accuracy</td><td>76.0</td><td>84.0</td><td>84.0</td></tr><tr><td>POPE</td><td>Accuracy</td><td>86.0</td><td>86.0</td><td>88.0</td></tr><tr><td>V* Bench</td><td>Accuracy</td><td>66.0</td><td>60.0</td><td>42.0</td></tr></table>


Table 2: Results of SCAFFOLD on top of Gemini-1.5 on 4 challenging vision-language benchmarks, with the highest score in bold. Despite the unknown differences in the black-box Gemini model, SCAFFOLD has shown consistent improvement on 3 out of 4 benchmarks.


Hallucination measures the tendency of LMMs to generate hallucinatory or illusory perceptions. Selected benchmarks are as follows. 1. POPE (Adversarial Subset) (Li et al., 2023b) assesses object hallucination by querying the existence of specific objects. 2. HallusionBench (Guan et al., 2023) consists of meticulously crafted images to measure hallucination and visual illusion in LMMs. 3. Mementos (Wang et al., 2024) evaluates LMM to conduct precise reasoning over image sequences and measures their performances in terms of object and behavior hallucinations. 

## 4.2 Baselines

This section presents the baseline prompting methods used in our experiments: 1. Standard Prompting utilizes original images and user instructions as inputs for LMMs, establishing a straightforward baseline without any prompt optimization. 2. CoT (Kojima et al., 2022) guides LMMs to perform step-by-step reasoning before outputting the final answer. The prompt text “Let’s think step by step." is prepended to task descriptions. 

<table><tr><td>Dataset</td><td>Metric</td><td>Standard</td><td>SoM</td><td>SCAFFOLD</td></tr><tr><td>WINO</td><td>Text Score</td><td>70.0</td><td>72.0</td><td>81.0</td></tr><tr><td>VSR</td><td>Accuracy</td><td>64.0</td><td>80.0</td><td>73.0</td></tr><tr><td>POPE</td><td>Accuracy</td><td>73.0</td><td>80.0</td><td>84.0</td></tr><tr><td>V* Bench</td><td>Accuracy</td><td>24.0</td><td>34.0</td><td>45.0</td></tr></table>


Table 3: Performance comparison with Set-of-Mark Prompting (Yang et al., 2023a) on 4 challenging visionlanguage benchmarks on top of GPT-4V, with the highest score in bold. SCAFFOLD outperforms SoM on 3 out of 4 benchmarks.


## 4.3 Results and Analyses

As presented in Table 1, the results demonstrate that SCAFFOLD significantly enhances the visual capabilities of LMMs, surpassing zero-shot CoT (Kojima et al., 2022) in most evaluated benchmarks. With standard prompting and CoT prompting averaging 50.0 and 52.4 respectively, SCAFFOLD successfully obtains an overall improvement of 8.8. The integration of SCAFFOLD and CoT also leads to further improvement (detailed in Appendix E.1). 

Spatial Reasoning: SCAFFOLD notably enhances the spatial reasoning capabilities of LLMs across the three benchmarks, with an average improvement of 13.3. Fig. 3 illustrates how SCAFFOLD enabled GPT-4V adeptly identifies crucial objects and records accurate positional information using two-dimensional coordinates, leading to the correct assessment of spatial relations through the numerical analysis of their x-coordinates. 

Compositional Reasoning: With SCAFFOLD, GPT-4V demonstrates improved abilities in compositional reasoning with an average improvement of 5.7, showing enhanced perception of key visual elements and smoother reasoning processes. As Fig. 7 in Appendix D shows, SCAFFOLD associates crucial objects with their textual positions, assisting 

Question: Determine whether thefollowing statement is true orfalse: The bench is at the left side of the potted plant. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/fa1f492d-3249-430e-bc39-42ccac2509e2/0064aad51af281abac0f0f17c35ebbe5e17c6f5a7b9cb11b3d9f91bf88ecfa53.jpg)


Ground-Truth: True. GPT-4V + CoT: The statement is false. In the image provided, the bench is on the right side of the potted plant, not the left. Therefore, the statement is false. GPT-4V + SCAFFOLD: The bench is located near coordinates (3,2) through (6,4), and the potted plant is near coordinates (3,6) through (6,6). Given the arrangement of the coordinates, the bench is to the left of the potted plant. true. 

Figure 3: An example from the Visual Spatial Reasoning (VSR) dataset. SCAFFOLD facilitates better visionlanguage coordination than other methods. 

GPT-4V to accurately identify and localize significant visual details. 

Fine-Grained Visual Understanding: On benchmarks such as V* Bench (Wu and Xie, 2023) and Spotting Differences, SCAFFOLD has markedly improved the capabilities of LMMs in conducting visual search and localizing fine-grained visual details, with an average improvement of 11.7. Taking Fig. 8 in Appendix D as an example, with the support of coordinates, GPT-4V effectively pinpoints and documents the precise locations of target visual elements, contributing to its accurate perception of the target attribute. Additionally, we notice that without coordinates, GPT-4V is more easily to give up and apologize for its search failure. 

Hallucination: With an average improvement of 5.7, utilizing coordinates as a scaffold enables GPT-4V to recognize objects within a scene and further accurately describe their positions, guiding its textual reasoning to focus on precise visual information. With an example shown in Fig. 9 in Appendix D, GPT-4V with coordinates is capable of precisely capturing visual details and preventing hallucinating non-existent objects, promoting accurate visual grounding. 

## 4.4 Effect of SCAFFOLD Atop Other LMMs

Given the impressive performance gains on top of GPT-4V, we continue to apply SCAFFOLD to Gemini-1.5-Flash (Gemini et al., 2023) to explore its generalization ability. Due to limited resources, we selected one dataset from each crucial capability and randomly sampled 50 samples from each. The results, detailed in Table 2, demonstrate that SCAFFOLD can still remarkably boost LMM performances in 3 out of 4 datasets. We also surprisingly notice that in the V* Bench, the standard prompting outperforms both CoT and SCAFFOLD. The results may come from two reasons: (i) Being a close-sourced LMM system, Gemini models may conduct specific optimizations for visual search in high-resolution images, which could hinder the effects of external prompting schemes. (ii) Sometimes the LMM tends to directly output the answer without leveraging the coordinates, without fully fulfilling the power of SCAFFOLD. 

As for open-sourced LMMs, we apply SCAF-FOLD on LLaVa-1.5 (Liu et al., 2023c) and find that it fails to leverage coordinates for textual reasoning, lacking both instruction following and visual understanding capabilities. To enable open-sourced models with SCAFFOLD may require instruction fine-tuning, which we leave for future research. 

## 4.5 Comparison with Set-of-Mark Prompting

To further validate the effectiveness of S , we compare SCAFFOLD with Set-of-Mark Prompting (Yang et al., 2023a), which leverages external segmentation models as tools. More experiment details are included in Appendix F. The results on 4 challenging benchmarks, as detailed in Table 3, illustrate the superior performance of SCAFFOLD on most benchmarks. Without external model deployment, SCAFFOLD, being significantly simpler, can achieve comparable performances with Set-of-Mark prompting, demonstrating its effectiveness and broad potential applications. 

## 5 Integration with Active Perception

In complex visual environments, humans would proactively engage with their surroundings to enhance scene understanding, like zooming in or changing perspectives. Similarly, we recognize that LMMs should possess such capabilities in realistic scenarios and propose that SCAFFOLD can function as a scaffold for effective active perception. 

To validate this, we integrate SCAFFOLD with active perception in the experiments on the direct_attributes subset of V* Bench (Wu and Xie, 2023), which requires LMMs to perceive finegrained details in high-resolution images. This challenge encompasses both the localization of target objects and the identification of their attributes under resolution constraints. Consequently, we adopt two metrics to measure LMM performance, including Not Found Rate representing the percentage of invalid responses, and Success Rate representing the percentage of correct responses. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/fa1f492d-3249-430e-bc39-42ccac2509e2/c32c607882be4fd79d1deced8b2bb45893698a34d6434e7da2dfc1c885329096.jpg)



Figure 4: The procedure of combined SCAFFOLD and active perception techniques on V* Bench.


<table><tr><td>Method</td><td>N.F.R. ↓ (%)</td><td>S.R. ↑ (%)</td></tr><tr><td>Standard</td><td>72.2</td><td>21.7</td></tr><tr><td>CoT</td><td>71.3</td><td>21.7</td></tr><tr><td>SCAFFOLD</td><td>26.2</td><td>31.3</td></tr><tr><td>SCAFFOLD + A.P. (Re-use)</td><td>14.8</td><td>45.2</td></tr><tr><td>SCAFFOLD + A.P. (Re-overlay)</td><td>13.0</td><td>47.8</td></tr></table>


Table 4: Results of SCAFFOLD + active perception on V* Bench (Wu and Xie, 2023) direct_attributes subset, where A.P. denotes active perception detailed in Sec. 5, N.F.R. denotes Not Found Rate and S.R. as Success Rate.


Depicted in Fig. 4, our combined method unfolds in two phases: an initial visual search to locate the target details, then cropping the image around the pinpointed coordinates to focus on the target attributes. As for processing the cropped images, we develop two types of variants: Re-use the coordinates overlaid in the original image, and Re-overlay the cropped image with a new set of X × Y dot matrices following Eq. (1). 

The results, presented in Table 4, reveal a performance enhancement of 14.1% and 16.5% compared with SCAFFOLD alone, underscoring the utility of coordinates in facilitating active perception. Furthermore, the results reveal two notable performance leaps. The initial improvement (CoT → SCAFFOLD) is attributed to the use of coordinates, significantly reducing the Not Found Rate and thereby aiding in the visual search process. The subsequent gain (SCAFFOLD → SCAFFOLD + A.P.) results from the combined implementation of active perception, which enables LMMs to discern target attributes within the cropped regions accurately. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/fa1f492d-3249-430e-bc39-42ccac2509e2/7af8cc9dd6001194c31db3322320d98247a457d0eff1c23f0edaaec69662babd.jpg)



Figure 5: Impact of matrix sizes. Better accuracies are illustrated in darker green.


## 6 Ablation Studies

We conduct extensive ablation studies on key factors such as matrix size and textual-only prompt guidance to validate and further explore SCAF-FOLD. Additional ablation studies, e.g., the integration of SCAFFOLD and CoT and the effect of matrix color, are included in Appendix E. 

## 6.1 Experimental Setup

Due to limited GPT-4V access quota, we each sample 50 questions from Visual Spatial Reasoning (Liu et al., 2023a), Winoground (Thrush et al., 2022), and POPE (Adversarial Subset) (Li et al., 2023b), creating an ablation subset of 150 samples. Overall accuracy per question is adopted as the metric and GPT-4V is used for our experiments. Additionally, for stable results, we run each experi-

<table><tr><td>Method</td><td>WNG</td><td>VSR</td><td>POPE</td><td>Overall</td></tr><tr><td>Standard</td><td>70.0</td><td>64.0</td><td>73.0</td><td>69.0</td></tr><tr><td>Text-Only Guideline</td><td>75.0</td><td>63.0</td><td>76.0</td><td>71.3</td></tr><tr><td>SCAFFOLD</td><td>81.0</td><td>73.0</td><td>84.0</td><td>79.3</td></tr></table>


Table 5: Results of the textual-only guidance baseline in ablation experiments, where WNG denotes the Winoground (Thrush et al., 2022) dataset and POPE denotes the POPE (Li et al., 2023b) Adversarial subset. ment twice and report average accuracy.


## 6.2 Effect of Matrix Size

The matrix size h and w may influence the precision of textual reference and the granularity of visual information. Consequently, we incorporate matrices of difference sizes varying from $3 \times 3$ to $7 \times 7$ and measure their performances. 

Fig. 5 depicts the performance variations with different matrix sizes, suggesting 6 × 6 as the optimal size for our ablation dataset. Additionally, the sizes in the upper right section tend to perform better than those in the lower left section. It may be due to the sampled images usually possessing equal or larger widths than heights, suggesting matrix sizes may ideally align with image sizes. 

Additionally, the 6 × 6 size did not yield the best results across all three subsets, hinting at potential improvements by customizing matrix sizes for specific tasks. It may be beneficial to automatically and dynamically adjust the matrix size and we leave this open problem to future research. 

## 6.3 Ablation with Textual-Only Guidance

Since SCAFFOLD consists of both visual and textual guidance, we add an instruction-only ablation experiment with only textual guidance, to validate the effectiveness of the dots. The textual guideline we used for the baseline is as follows: 

```txt
You need to complete a task. When identifying key objects in the image, you can first generate approximate locations of the objects then complete the task. 
```

The results, detailed in Table 5, show that the text-only guidelines can slightly improve the performance but the improvement is marginal compared with SCAFFOLD. It further demonstrates the effectiveness of the coordinates. 

## 7 Discussion

The effect of resolution and ratios of the tested images. In our work, we have conducted experiments on 11 challenging datasets with diverse image ratios and resolutions to examine the flexibility and generalizability of SCAFFOLD. To show the diversity of the image inputs, we collect all the images in our experiments and illustrate the statistics about their width-height ratios and resolutions. The distribution is visualized in Fig 6. As the distribution shows, SCAFFOLD remains effective across images with varying ratios and resolutions. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/fa1f492d-3249-430e-bc39-42ccac2509e2/538da915a43bbe7b45e287427f928fe1b441368931b79afb6010ff9eb95c0e70.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/fa1f492d-3249-430e-bc39-42ccac2509e2/72605cf381052ca9afc48e40f67a14e51c127ea58a0969524c907b777ca943a4.jpg)



Figure 6: The distribution of the image ratios and sizes in our experiments.


<table><tr><td>Method</td><td>WINO</td><td>VSR</td><td>POPE</td><td>Overall</td></tr><tr><td>Standard</td><td>70.0</td><td>64.0</td><td>73.0</td><td>69.0</td></tr><tr><td>1-shot, Standard</td><td>76.0</td><td>66.0</td><td>76.0</td><td>72.7</td></tr><tr><td>1-shot, SCAFFOLD, w/o text inst.</td><td>79.0</td><td>70.0</td><td>81.0</td><td>76.7</td></tr><tr><td>0-shot, SCAFFOLD, w/ text inst.</td><td>81.0</td><td>73.0</td><td>84.0</td><td>79.3</td></tr></table>


Table 6: Results of few-shot explorations, where WNG denotes the Winoground (Thrush et al., 2022) dataset and POPE denotes the POPE (Li et al., 2023b) Adversarial subset.


Potential usage of in-context examples. SCAF-FOLD leverages general textual guidelines to steer LMMs to reason via the coordinates. Being a zeroshot method, SCAFFOLD does not require users to carefully curate samples for each task. However, in-context examples are also effective in eliciting LMM capabilities, we explore few-shot approaches and the result are detailed in Table 6. 

In our one-shot experiments, we curate one example from each dataset and remove the textual guidelines in our standard implementation (denoted as 1-shot, SCAFFOLD, w/o text inst. in the table). The example contains a user query and a curated response which utilizes coordinates to better perceive the image. Unsurprisingly, we found that GPT-4V can leverage the capabilities of the coordinates in 1-shot setting without textual instructions and exhibit remarkable improvement compared with the 1-shot standard setting (without coordinates). Additionally, due to simplicity and effectiveness, we still recommend our standard zero-shot implementation. 

## 8 Conclusion

In this work, we propose SCAFFOLD, a simple and general visual prompting method that utilizes scaffolding coordinates to promote vision-language coordination in LMMs. Extensive experiments show that SCAFFOLD successfully elicits LMM capabilities in several challenging vision-language tasks. 

## Limitations

Here we discuss two limitations of this work. 

(1) To automatically adjust dot matrix attributes. In this work, for simplicity and clarity, we adopt matrices of size 6 × 6 in our implementation. However, our ablation study in Section 6 suggests that a one-size-fits-all matrix size can yield good, but not the best results across all datasets. Given the diversity of visual tasks and the varying granularity of information in different scenes, it stands to reason that tailoring the matrix attributes, such as size and coordinates format, to the specific requirements of each task or even each sample could improve performance. Addressing the dynamic and automatic adjustment of these attributes to better suit different scenarios remains an area for future exploration. 

(2) To enhance precision in visual localization. By integrating dot matrices with coordinates onto images, we aimed to facilitate improved visionlanguage coordination by associating key objects with their closest coordinates. However, our observations indicate that, particularly in complex or clustered scenes, GPT-4V occasionally struggles to accurately associate textual reasoning with the nearest coordinates. This challenge underscores the need for LMMs to achieve improved visual localization and grounding capabilities in intricate environments. With SCAFFOLD, we expect the future of LMMs and visual prompting techniques to be further improved in terms of visual localization. 

## Acknowledgements

This work is supported by the National Key R&D Program of China (2022ZD0160502), the National Natural Science Foundation of China (No. 61925601, 62276152), and funding from Wuxi Research Institute of Applied Technologies, Tsinghua University under Grant 20242001120. We would like to thank An Liu and Zeyuan Yang for their valuable advice. 

## References



Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774. 





Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Michal Podstawski, Hubert Niewiadomski, Piotr Nyczyk, et al. 2023. Graph of thoughts: Solving elaborate problems with large language models. arXiv preprint arXiv:2308.09687. 





Nitzan Bitton-Guetta, Yonatan Bitton, Jack Hessel, Ludwig Schmidt, Yuval Elovici, Gabriel Stanovsky, and Roy Schwartz. 2023. Breaking common sense: Whoops! a vision-and-language benchmark of synthetic and compositional images. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 2616–2627. 





Yunkang Cao, Xiaohao Xu, Chen Sun, Xiaonan Huang, and Weiming Shen. 2023. Towards generic anomaly detection and understanding: Large-scale visuallinguistic model (gpt-4v) takes the lead. arXiv preprint arXiv:2311.02782. 





Boyuan Chen, Zhuo Xu, Sean Kirmani, Brian Ichter, Danny Driess, Pete Florence, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. 2024. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. arXiv preprint arXiv:2401.12168. 





Chi Chen, Ruoyu Qin, Fuwen Luo, Xiaoyue Mi, Peng Li, Maosong Sun, and Yang Liu. 2023a. Position-enhanced visual instruction tuning for multimodal large language models. arXiv preprint arXiv:2308.13437. 





Liang Chen, Yichi Zhang, Shuhuai Ren, Haozhe Zhao, Zefan Cai, Yuchi Wang, Peiyi Wang, Tianyu Liu, and Baobao Chang. 2023b. Towards end-to-end embodied decision making via multi-modal large language model: Explorations with gpt4-vision and beyond. arXiv preprint arXiv:2310.02071. 





Sijie Cheng, Zhicheng Guo, Jingwen Wu, Kechen Fang, Peng Li, Huaping Liu, and Yang Liu. 2023. Can vision-language models think from a first-person perspective? arXiv preprint arXiv:2311.15596. 





Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. 2023. Instructblip: Towards general-purpose vision-language models with instruction tuning. arXiv preprint arXiv:2305.06500. 





Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. 2023a. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394. 





Chaoyou Fu, Renrui Zhang, Haojia Lin, Zihan Wang, Timin Gao, Yongdong Luo, Yubo Huang, Zhengye Zhang, Longtian Qiu, Gaoxiang Ye, et al. 2023b. A challenger to gpt-4v? early explorations of gemini in visual expertise. arXiv preprint arXiv:2312.12436. 





Gemini, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. 2023. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805. 





Tianrui Guan, Fuxiao Liu, Xiyang Wu Ruiqi Xian Zongxia Li, Xiaoyu Liu Xijun Wang, Lichang Chen Furong Huang Yaser Yacoob, and Dinesh Manocha Tianyi Zhou. 2023. Hallusionbench: An advanced diagnostic suite for entangled language hal lucination & visual illusion in large vision-language models. arXiv preprint arXiv:2310.14566. 





Hongliang He, Wenlin Yao, Kaixin Ma, Wenhao Yu, Yong Dai, Hongming Zhang, Zhenzhong Lan, and Dong Yu. 2024. Webvoyager: Building an end-toend web agent with large multimodal models. arXiv preprint arXiv:2401.13919. 





Justin Johnson, Bharath Hariharan, Laurens Van Der Maaten, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick. 2017. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2901–2910. 





Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, and Ross Girshick. 2023. Segment anything. arXiv:2304.02643. 





Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large language models are zero-shot reasoners. Advances in neural information processing systems, 35:22199– 22213. 





Feng Li, Hao Zhang, Peize Sun, Xueyan Zou, Shilong Liu, Jianwei Yang, Chunyuan Li, Lei Zhang, and Jianfeng Gao. 2023a. Semantic-sam: Segment and recognize anything at any granularity. arXiv preprint arXiv:2307.04767. 





Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. 2023b. Eval uating object hallucination in large vision-language models. arXiv preprint arXiv:2305.10355. 





Fangyu Liu, Guy Emerson, and Nigel Collier. 2023a. Visual spatial reasoning. Transactions ofthe Associationfor Computational Linguistics, 11:635–651. 





Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. 2023b. Improved baselines with visual instruction tuning. arXiv preprint arXiv:2310.03744. 





Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023c. Visual instruction tuning. In NeurIPS. 





Chancharik Mitra, Brandon Huang, Trevor Darrell, and Roei Herzig. 2023. Compositional chain-of-Thought prompting for large multimodal models. arXiv preprint arXiv:2311.17076. 





Soroush Nasiriany, Fei Xia, Wenhao Yu, Ted Xiao, Jacky Liang, Ishita Dasgupta, Annie Xie, Danny Driess, Ayzaan Wahid, Zhuo Xu, Quan Vuong, Tingnan Zhang, Tsang-Wei Edward Lee, Kuang-Huei Lee, Peng Xu, Sean Kirmani, Yuke Zhu, Andy Zeng, Karol Hausman, Nicolas Heess, Chelsea Finn, Sergey Levine, and Brian Ichter. 2024. Pivot: Iterative visual prompting elicits actionable knowledge for vlms. arXiv preprint arXiv:2402.07872. 





Tristan Thrush, Ryan Jiang, Max Bartolo, Amanpreet Singh, Adina Williams, Douwe Kiela, and Candace Ross. 2022. Winoground: Probing vision and language models for visio-linguistic compositionality. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5238– 5248. 





Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. 2024. Eyes wide shut? exploring the visual shortcomings of multimodal llms. arXiv preprint arXiv:2401.06209. 





Rohan Wadhawan, Hritik Bansal, Kai-Wei Chang, and Nanyun Peng. 2024. Contextual: Evaluating contextsensitive text-rich visual reasoning in large multimodal models. arXiv preprint arXiv:2401.13311. 





Naoki Wake, Atsushi Kanehira, Kazuhiro Sasabuchi, Jun Takamatsu, and Katsushi Ikeuchi. 2023. Gpt-4v (ision) for robotics: Multimodal task planning from human demonstration. arXiv preprint arXiv:2311.12015. 





Xiyao Wang, Yuhang Zhou, Xiaoyu Liu, Hongjin Lu, Yuancheng Xu, Feihong He, Jaehong Yoon, Taixi Lu, Gedas Bertasius, Mohit Bansal, et al. 2024. Mementos: A comprehensive benchmark for multimodal large language model reasoning over image sequences. arXiv preprint arXiv:2401.10529. 





Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. 2022. Chain-of-Thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems, 35:24824–24837. 





Licheng Wen, Xuemeng Yang, Daocheng Fu, Xiaofeng Wang, Pinlong Cai, Xin Li, Tao Ma, Yingxuan Li, Linran Xu, Dengke Shang, et al. 2023. On the road with gpt-4v (ision): Early explorations of visual-language model on autonomous driving. arXiv preprint arXiv:2311.05332. 





Penghao Wu and Saining Xie. 2023. V*: Guided visual search as a core mechanism in multimodal llms. arXiv preprint arXiv:2312.14135, 17. 





Yang Wu, Shilong Wang, Hao Yang, Tian Zheng, Hongbo Zhang, Yanyan Zhao, and Bing Qin. 2023a. An early evaluation of gpt-4v (ision). arXiv preprint arXiv:2310.16534. 





Yifan Wu, Pengchuan Zhang, Wenhan Xiong, Barlas Oguz, James C Gee, and Yixin Nie. 2023b. The role of chain-of-Thought in complex vision-language reasoning task. arXiv preprint arXiv:2311.09193. 





Zhiyang Xu, Ying Shen, and Lifu Huang. 2023. Multi-Instruct: Improving multi-modal zero-shot learning via instruction tuning. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 11445– 11465, Toronto, Canada. Association for Computational Linguistics. 





Jianwei Yang, Hao Zhang, Feng Li, Xueyan Zou, Chunyuan Li, and Jianfeng Gao. 2023a. Set-of-mark prompting unleashes extraordinary visual grounding in gpt-4v. arXiv preprint arXiv:2310.11441. 





Zhengyuan Yang, Linjie Li, Kevin Lin, Jianfeng Wang, Chung-Ching Lin, Zicheng Liu, and Lijuan Wang. 2023b. The dawn of lmms: Preliminary explorations with gpt-4v (ision). arXiv preprint arXiv:2309.17421, 9(1):1. 





Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L Griffiths, Yuan Cao, and Karthik Narasimhan. 2023. Tree of thoughts: Deliberate problem solving with large language models. arXiv preprint arXiv:2305.10601. 





Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. 2023. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. arXiv preprint arXiv:2311.16502. 





Shilong Zhang, Peize Sun, Shoufa Chen, Min Xiao, Wenqi Shao, Wenwei Zhang, Kai Chen, and Ping Luo. 2023. Gpt4roi: Instruction tuning large language model on region-of-interest. arXiv preprint arXiv:2307.03601. 





Boyuan Zheng, Boyu Gou, Jihyung Kil, Huan Sun, and Yu Su. 2024. Gpt-4v (ision) is a generalist web agent, if grounded. arXiv preprint arXiv:2401.01614. 





Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. 2023. Judging llm-as-a-judge with mt-bench and chatbot arena. arXiv preprint arXiv:2306.05685. 



## A Prompts

This section exhibits the prompts used in SCAF-FOLD implementation and our experiments. 

## A.1 SCAFFOLD implementation

In SCAFFOLD implementation, we use the following textual guidelines to describe the effective use of coordinates. 

Single-Image Setting. In the single-image setting, we label all the dots with two-dimensional coordinates and deliver both the original image and the coordinates-overlaid image to the model. Consequently, we use the following guidelines. 

I will provide you with two images of the same scene. The second image is overlaid with a dot matrix of the shape of HEIGHT * WIDTH to help you with your task, and each dot is labeled with two-dimensional coordinates (x,y). 1. When you mention any key objects in the image, first output their nearest coordinates then identify them. 2. You use the coordinates to determine the spatial relationships of the objects. Within each column, the x-coordinate increases from top to bottom, and within each row, the y-coordinate increases from left to right. 3. You can search and reason region by region with the help of the dots. 4. Finally, conclude your answer in format [[AN-SWER]], such as [[A]], [[B]], [[C]] or [[D]]. 

Note that the fourth guideline serves as a constraint for specific output formats and may vary among different tasks. 

Double-Images Setting. In a double-images setting, we label all the dots with three-dimensional coordinates, with the first coordinate serving to distinguish between two images. Consequently, we use the following guidelines. 

Two images are provided, each overlaid with a grid of dots arranged in a matrix with dimensions h by w. Each dot on this grid is assigned a unique set of three-dimensional coordinates labeled as (t, x, y). The first coordinate, ’t’, serves to distinguish between the two images: ’1’ is assigned to the first image on the left, and ’2’ to the second image on the right. The other two coordinates, ’x’ and ’y’, are used to specify the dot’s spatial location within its respective image. This labeling system is designed to assist you in identifying and referring to specific points within each image. 1. When you mention any key objects in the image, first output their nearest coordinates then identify them. 2. You use the coordinates to determine the spatial relationships of the objects. within each column, the x-coordinate increases from top to bottom, and Within each row, the y-coordinate increases from left to right. 3. You can search and reason region by region with the help of the dots. 4. Finally, you must conclude your answer in format [[ANSWER]], such as [[A]] or [[B]]. 

Note that the fourth guideline serves as a constraint for specific output formats and may vary among different tasks. 

Image-Sequence Setting. In the image sequence setting, we label all the dots with three-dimensional coordinates. For simplicity and efficiency, we only deliver the coordinates-overlaid image to the model. Consequently, we use the following guidelines. 

A sequence of images is provided, each overlaid with a grid of dots arranged in a matrix with dimensions HEIGHT by WIDTH. Each dot on this grid is assigned a unique set of three-dimensional coordinates labeled as (t, x, y). The first coordinate, ’t’, serves to distinguish between the images, for instance, ’1’ is assigned to the first image, and ’8’ to the last image. The other two coordinates, ’x’ and ’y’, are used to specify the dot’s spatial location within its respective image. This labeling system is designed to assist you in identifying and referring to specific points within each image. 

1. When you mention any key objects in the image, first output their nearest coordinates then identify them. 

2. You use the coordinates to determine the temporal and spatial relationships of the objects. Within the image sequence, the t-coordinate increases as time grows; within each column, the x-coordinate increases from top to bottom; within each row, the y-coordinate increases from left to right. 

3. You can search and reason region by region with the help of the dots. 

4. you need to keep your descriptions concise and clear. 

Note that the fourth guideline serves as a constraint for specific output formats and may vary among different tasks. 

## A.2 Integration Experiments

We list the prompts used in the integration experiments. 

SCAFFOLD + Active Perception Firstly, we guide the LMM to visually search the image and localize target objects using the following prompt. 

Based on the question: question you should first identify key objects in the question and link them with their nearest coordinate, and finally conclude the coordinates in format [[(x,y)]] in the end(you don’t need to answer the question). 

Secondly, we crop the image based on the out put coordinates and guide the LMM to answer the question based on the cropped images in the second turn of the conversation. 

Here are the cropped images from the scene according to your selected coordinates, you can take a closer look and answer the question. If I don’t provide cropped images or the target does not exist in the cropped image, please visually search the original image and answer the question. Question: QUESTION Options: OPTIONS 

## B Experimental details

This section details the setting of our experiments. The model version adopted in GPT series is gpt-4vvision-preview(referred as GPT-4V in our paper) to purely study its vision abilities without the intervention of external tools in more advanced models. The model version adopted in Gemini series is Gemini-1.5-Flash(referred as Gemini-1.5 in our paper) for its advanced vision-language capabilities and test efficiency. In our experiments, we set the temperature to zero to increase stability of the results. 

## C Benchmarks

This section details the benchmarks used in our experiments to evaluate our method. The benchmarks are divided into four categories and elaborated respectively. 

## C.1 Spatial Reasoning

Spatial reasoning is a crucial capability of LMMs to determine spatial relationships between objects in the image. To evaluate the effectiveness of our method to elicit spatial reasoning capabilities, we select several challenging datasets including MME (Fu et al., 2023a) Position split, Visual Spatial Reasoning (Liu et al., 2023a) dataset, Ego-Think (Cheng et al., 2023) Spatial split. Note that we only select a subset related to spatial reasoning in comprehensive evaluation benchmarks like MME (Fu et al., 2023a) and EgoThink (Cheng et al., 2023) because the other subsets don’t constitute spatial reasoning and our GPT-4V access quota is limited. The details of the benchmarks are as follows. 

## MME (Position) (Fu et al., 2023a)

Dataset Introduction. The MME (Fu et al., 2023a) benchmark is a comprehensive evaluation suite designed to measure the perception and recognition capabilities of LMMs in 14 tasks. To evaluate spatial reasoning capabilities, we only select Position split from 14 subtasks. It contains 60 questions about the spatial relationship among the objects in the scene, such as left, above, etc.. 

Metric. All questions are formatted as general interrogative sentences, which can be answered with either "yes" or "no". Therefore, we guide the model to output the final answer within double square brackets [[]], and then use accuracy as the metric to evaluate the model’s performance. 

## Visual Spatial Reasoning (VSR) (Liu et al., 2023a)

Dataset Introduction. The VSR (Liu et al., 2023a) dataset is designed to comprehensively evaluate LMM capabilities to perform spatial reasoning in images, containing 66 types of spatial descriptions in language. Each sample provides an image and a corresponding spatial description in terms of two individual objects in the scene. Because of the limited GPT-4V (Achiam et al., 2023) access quota, we randomly sample 200 samples from the dataset for evaluation. 

Metric. The task is to determine the correctness of the given spatial description, which can be answered with either "true" or "false". Therefore, we guide the model to output the final answer within double square brackets [[]], and then use accuracy as the metric to evaluate the model’s performance. EgoThink (Spatial) (Cheng et al., 2023) 

Dataset Introduction. The Egothink (Cheng et al., 2023) dataset is intended for first-perspective vision-language problem-solving capabilities. To evaluate spatial reasoning capabilities, we only select Spatial split from it. The Spatial split contains 50 questions about the spatial relationship among the objects, particularly requiring LMMs to perceive and reason from the first perspective. 

Metric. The task is to answer spatial questions from the first perspective. We adopt the evaluation script in EgoThink official implementation <sup>3</sup>, which uses GPT-4 (Achiam et al., 2023) as the judge model to score the answers. 

## C.2 Compositional Reasoning

Compositional reasoning represents the capability of LMMs to perceive and reason in terms of objects’ attributes and their relationships, significant for visual perception. To evaluate the effectiveness of our method to elicit compositional reasoning capabilities, we select several challenging datasets including Winoground (Thrush et al., 2022), WHOOPS! (Bitton-Guetta et al., 2023) and 

CLEVR (Johnson et al., 2017). 

## Winoground (Thrush et al., 2022)

Dataset Introduction. Winoground (Thrush et al., 2022) proposes a novel dataset that challenges LMMs to correctly match two images and two captions. The captions use the same words but in a different order, requiring a precise understanding of both images and captions. The challenging dataset is a suitable benchmark for compositional reasoning. 

Metric. Given two images and two captions, we compose a sample into four binary-choice questions for the effective evaluation of LMMs, includ ing choosing the correct caption given two images and choosing the correct image given two captions respectively. Finally, we adopt group score to measure LMM performance: only when the model answers all four questions correctly is it considered completely correct for this sample. 

## WHOOPS! (Bitton-Guetta et al., 2023)

Dataset Introduction. The WHOOPS! (Bitton-Guetta et al., 2023) is designed to challenge LMMs to perform compositional reasoning in terms of purposefully commonsense-defying images. It remains challenging for LMMs to recognize and interpret these unconventional images. Furthermore, several tasks are posed over the dataset, and we select the visual question-answering task. 

Metric. We adopt the BEM score in WHOOPS! official implementation <sup>4</sup> to evaluate LMM performances. 

## CLEVR (Johnson et al., 2017)

Dataset Introduction. The CLEVR (Johnson et al., 2017) is designed as a standard evaluation suite for compositional vision-language reasoning with program-rendered images and correctly generated annotations. Containing objects like a metal cube or red sphere, the dataset poses questions in terms of object existence, object attributes, and object relationships. For effective evaluation, we randomly sampled 200 samples in the dataset. 

Metric. We guide the LMMs to answer the questions in an open-ended generative manner. Due to the complexity of the question and the answer, we adopt GPT-4 (Achiam et al., 2023) as a judge model to determine the correctness of answers. Our judge prompt is as follows, adapted from MT-Bench (Zheng et al., 2023). 

[Instruction] Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider correctness and helpfulness. You will be given a reference answer and the assistant’s answer. Begin your evaluation by comparing the assistant’s answer with the reference answer. Identify and correct any mistakes. The assistant has access to an image along with questions but you will not be given images. Therefore, please consider only how the answer is close to the reference answer. If the assistant’s answer is not exactly the same as or similar to the answer, then he must be wrong. Be as objective as possible. Discourage uninformative answers. Also, equally, treat short and long answers and focus on the correctness of answers. After providing your explanation, you must rate the response with either 0, 0.5, or 1 by strictly following this format: "[[rating]]", for example: "Rating: [[0.5]]". 

[Question] question 

[The Start of Reference Answer] ground_truth [The End of Reference Answer] 

[The Start of Assistant’s Answer] answer [The End of Assistant’s Answer] 

## C.3 Fine-Grained Visual Understanding

Fine-grained Visual Understanding represents the capability of LMMs to precisely capture, perceive, and describe certain visual details in the scene. To evaluate the effectiveness of our method to elicit Fine-grained Visual Understanding, we select several challenging datasets including V* Bench (Wu and Xie, 2023) and Spotting Differences. 

## V* Bench (Wu and Xie, 2023)

Dataset Introduction. The V* Bench (Wu and Xie, 2023) is designed to challenge LMMs to perform visual search and identify fine-grained visual details in high-resolution images. The challenges lie in the necessity of visual search, while GPT-4V sometimes fails to capture visual details, responding with I’m sorry, I couldn’tfind XXX in the image. Note that we conduct our evaluation in an openended generative manner, GPT-4V sometimes fails to identify targets and refuses to choose an option. Metric. The questions are multiple-choice questions. Due to the generative nature of current LMMs, we pose a question to the target LMM and let it generate an open-ended response. Finally, we guide the model to output the final answer within double square brackets [[]], and then use accuracy as the metric to evaluate the model’s performance. Spotting Differences 5 

Dataset Introduction. The dataset was designed to challenge the spatial analysis capabilities and object localization abilities of LLMs. Inspired and derived by the "Spot the Difference" web game, each level consists of two images. The goal is to locate the differences between the two images at 10 specific locations. The first 50 levels, increasing in difficulty, were selected as samples for this dataset. The challenge lies in the LLM’s ability to first analyze the images and then accurately localize the correct object positions. GPT-4V may sometimes return I’m sorry, but I cannot assist with this request.. 

Metric. We guide GPT-4V to answer the pixel or matrix positions of the difference by dividing the prompt into question text + standardized answer requirement + prompt trick. We use OpenCV’s Hough Circle Transform to locate the correct image position of the difference. When the distance between the correct position and the position given by GPT-4V is less than 50 pixels, the difference positioning is considered successful. 

## C.4 Hallucination

Hallucination in vision-language contexts represents a misalignment between visual inputs and textual outputs in terms of object existence, attributes, and relationships. To evaluate the effectiveness of our method to mitigate hallucination in vision-language tasks, we select several challenging datasets including POPE Adversarial subset (Li et al., 2023b), HallusionBench (Hard) (Guan et al., 2023) and Mementos (Wang et al., 2024). The details of these benchmarks are as follows. 

## POPE (Li et al., 2023b)

Dataset Introduction. The POPE (Li et al., 2023b) benchmark is designed for measuring object hallucination in images. The questions are built based on the object annotations, challenging LMMs to determine object existence in images. 

Metric. All questions are formatted as general interrogative sentences, which can be answered with either "yes" or "no". Therefore, we guide the model to output the final answer within double square brackets [[]], and then use accuracy as the metric to evaluate the model’s performance. 

## HallusionBench (Li et al., 2023b)

Dataset Introduction. The HallusionBench (Li et al., 2023b) is designed for evaluating language hallucination and visual illusion in LMMs. The images and questions are meticulously crafted by human experts, posing great challenges to current LMMs. Due to the limited GPT-4V quota, we evaluate our method on the Hard set. 

<table><tr><td>Dataset</td><td>Standard</td><td>CoT</td><td>SCAFFOLD</td><td>SCAFFOLD + CoT</td></tr><tr><td>Wino.</td><td>17.0</td><td>33.0</td><td>33.0</td><td>41.0</td></tr><tr><td>V*</td><td>27.2</td><td>30.8</td><td>44.6</td><td>47.9</td></tr></table>


Table 7: Results of SCAFFOLD combined with Chain of-Thought (Wei et al., 2022) on V* Bench (Wu and Xie, 2023) and Winoground (Thrush et al., 2022).


Metric. All questions are formatted as general interrogative sentences, which can be answered with either "yes" or "no". Therefore, we simplify the evaluation process <sup>6</sup> and guide the model to output the final answer within double square brackets [[]], and then use accuracy as the metric to evaluate the model’s performance. 

## Mementos (Wang et al., 2024)

Dataset Introduction. The Mementos (Wang et al., 2024) is designed for evaluating reasoning capabilities across image sequences. Featuring 4,761 diverse image sequences with varying lengths, the dataset adopts a GPT-4_assisted metric to evaluate the correctness of objects and behaviors in generated descriptions, reflecting both reasoning capabilities and hallucination levels. 

Metric. The task is to generate a description of the image sequences. We adopt the evaluation script in Mementos official implementation <sup>7</sup>, which uses GPT-4 (Achiam et al., 2023) as the judge model to extract the objects and behaviors in the description, then calculates F1 score in terms of objects and behaviors respectively. Finally, we use the average F1 score to measure the performance. 

## D Complementary Cases

This section complements more cases on our evaluated benchmarks. 

## E Additional Ablation Studies

## E.1 Integration of SCAFFOLD with Chain-of-Thought

Our prompting method, characterized by its simplicity, can seamlessly integrate with zero-shot CoT by appending Let’s think step by step. to user instructions. To test its effectiveness, we conduct experiments on Winoground (Thrush et al., 2022) and V* Bench (Wu and Xie, 2023). Results from Table 7 demonstrate that combining our method with CoT enhances LMM performance beyond what 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/fa1f492d-3249-430e-bc39-42ccac2509e2/79cdae7a38d7fecf342635af1cb3d8b1a22fb26ccccadb7d60bf4b2e42f5426c.jpg)



Choose the correct caption for the image. (A) a frog is above rocks which are under the water (B) afrog is under rocks which are above the water.


GPT-4V + CoT: The frog is visible above the rocks, and the rocks are partially submerged in water, indicating that the frog is not under the rocks but rather resting on top of them or between them above the water level. A. 

GPT-4V + SCAFFOLD: The frog is located near coordinates (3,3) and (4,3), and it is clearly visible underneath the rocks, which are near coordinates (1,3), (2,3), (1,4), and (2,4). The water is visible around the frog, particularly near coordinates (5,4) and (6,4). Based on the spatial relationships, the rocks are above the water, and the frog is under the rocks, which are above the water. The correct caption for the image is: B. 

Figure 7: An example from the Winoground dataset. 

either method achieves independently. These findings underscore our method’s substantial compatibility and potential for performance improvement when combined with other methods. 

## E.2 Effect of Matrix Color

In terms of matrix color, we design different coloring strategies and compare their performances. As illustrated in Fig. 10, uniform coloring strategies adopt the same color for various scenes, occasionally blending into the surroundings. Complementary colors introduce large amounts of colors and may mislead model attention. Consequently, for simplicity and visibility, we choose the most contrasting color from black and white at each dot location. To assess the efficacy of our approach, we compared it against baseline coloring strategies, including uniform black, uniform white, and complementary coloring. As shown in Table 8, our binary coloring strategy slightly surpasses the alternatives in performance. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/fa1f492d-3249-430e-bc39-42ccac2509e2/b89ed629aebddf4bd46375caec1f2c240b953d996e014def1c7f97d0bb5f8171.jpg)



Figure 8: An example from the V* Bench dataset.


<table><tr><td>Coloring Strategy</td><td>WNG</td><td>VSR</td><td>POPE</td><td>Overall</td></tr><tr><td>None (Baseline)</td><td>70.0</td><td>64.0</td><td>73.0</td><td>69.0</td></tr><tr><td>White</td><td>78.0</td><td>72.0</td><td>78.0</td><td>76.0</td></tr><tr><td>Black</td><td>79.0</td><td>73.0</td><td>82.0</td><td>78.0</td></tr><tr><td>Complementary</td><td>77.0</td><td>71.0</td><td>81.0</td><td>76.3</td></tr><tr><td>Binary (Ours)</td><td>81.0</td><td>73.0</td><td>84.0</td><td>79.3</td></tr></table>

## E.3 Effect of Dot Placement Perturbations


Table 8: Results of different coloring strategies in ablation experiments, where WNG denotes the Winoground (Thrush et al., 2022) dataset and POPE denotes the POPE (Li et al., 2023b) Adversarial subset.


To assess the resilience of SCAFFOLD, we introduce Gaussian noise to the dots, slightly adjusting their positions without significantly changing their relative placements, as illustrated in Fig. 11. We model the original dot positions as (X, Y), with $l _ { h }$ and $l _ { w }$ representing the distances between neighboring dots along the x and y axes, respectively. The perturbed coordinate $( X _ { \mathrm { n e w } } , Y _ { \mathrm { n e w } } )$ reads: 

$$
\left[ \begin{array}{c} \mathbf {X} _ {\text {new}} \\ \mathbf {Y} _ {\text {new}} \end{array} \right] = \left[ \begin{array}{c} \mathbf {X} \\ \mathbf {Y} \end{array} \right] + \left[ \begin{array}{c} \mathcal {N} \left(0, \left(\frac {1}{4} \cdot l _ {h}\right) ^ {2}\right) \\ \mathcal {N} \left(0, \left(\frac {1}{4} \cdot l _ {w}\right) ^ {2}\right) \end{array} \right]\tag{2}
$$

According to the findings depicted in Fig. 11, the overall performance of perturbed coordinates are marginally lower than the standard implementation but still show impressive improvement on the baseline, indicating their robustness. We also surprisingly notice that in the VSR subset, the perturbed dots is slightly better. The results may come from two reasons. Firstly, the perturbation is minor, therefore it would not severely change the positions of the dots, making it still effective. Secondly, it is possible that some perturbed dots are closer to certain objects, and may slightly increase grounding capabilities. Furthermore, the experiments imply that it may be potentially beneficial to tailor the placement of the dots instead of uniform distribution in certain application fields. 


Figure 9: An example from the POPE (adversarial) dataset.


## E.4 Effect of Coordinate Formats

Coordinates, as textual references for dots, are vital for aligning visual inputs with textual outputs. To assess the effectiveness of our implementation, we experiment with various coordinate formats, including alphabetic, one-dimensional numerical, and pixel absolute coordinates. The examples of these formats are exhibited in Fig. 12. 

The results, detailed in Table 9, reveal that our approach surpasses alternative coordinate designs in performance. Additionally, all the coordinates designs perform better than the baseline without coordinates, indicating the flexibility of coordinates design and the adaptability to different scenarios. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/fa1f492d-3249-430e-bc39-42ccac2509e2/a794456514062584b4b07088d090928c30e4fe05c5eeaf15bda14022cb8ed86b.jpg)



(a) Black


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/fa1f492d-3249-430e-bc39-42ccac2509e2/31645495f5df2cd517eaec0722465592c3d95cd0ff6a3df429b4c372f03f83e0.jpg)



(b) White


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/fa1f492d-3249-430e-bc39-42ccac2509e2/d13e4d16f829705c708d584f679be6e2f3d170276beb8592af501fef7e3c5ecb.jpg)



(c) Complementary


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/fa1f492d-3249-430e-bc39-42ccac2509e2/ea54c0172249900ae0a00c46ab5b2cf11f286cdb9768f4695b1c87369d4c8647.jpg)



(d) Binary (Ours)



Figure 10: Examples of different color configurations of dot matrices and coordinates.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/fa1f492d-3249-430e-bc39-42ccac2509e2/6357af5f62ca851ad128e0d05f50696019f65a67c5d88c6116692d5885289ea2.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/fa1f492d-3249-430e-bc39-42ccac2509e2/45b4f04afb0fb163fbfa2af2fbe3c0e7a18336ec32fb565a991913d881b85aaf.jpg)



Semantic-SAM model for all the tasks evaluated and set the segmentation level to 3. Additionally, SAM model (Kirillov et al., 2023) generates too many labels that hinder visual understanding of LMMs.



Figure 11: The results of perturbed and standard coordinates (left) and a perturbed coordinatesoverlaid example (right), where WNG denotes the Winoground (Thrush et al., 2022) dataset and POPE denotes the POPE (Li et al., 2023b) Adversarial subset.


<table><tr><td>Coordinates</td><td>WNG</td><td>VSR</td><td>POPE</td><td>Overall</td></tr><tr><td>None (Baseline)</td><td>70.0</td><td>64.0</td><td>73.0</td><td>69.0</td></tr><tr><td>Alphabet</td><td>80.0</td><td>72.0</td><td>79.0</td><td>77.0</td></tr><tr><td>Pixel</td><td>78.0</td><td>75.0</td><td>77.0</td><td>76.7</td></tr><tr><td>One-Dimensional</td><td>72.0</td><td>71.0</td><td>81.0</td><td>74.7</td></tr><tr><td>Cartesian (Ours)</td><td>81.0</td><td>73.0</td><td>84.0</td><td>79.3</td></tr></table>


Table 9: Results of different coordinates designs in ablation experiments, where WNG denotes the Winoground (Thrush et al., 2022) dataset and POPE denotes the POPE (Li et al., 2023b) Adversarial subset.


## F Set-of-Mark Experiment Setup

To apply Set-of-Mark prompting (Yang et al., 2023a) to the selected vision-language benchmarks, we adopt the official implementation <sup>8</sup>. Since SoM incorporate various object segmentation models with different visual granularities, we need to select the proper setting for our selected tasks. Based on our primary tests, we find that Semantic-SAM model (Li et al., 2023a) can produce segmentation maps of proper granularity. Therefore, we use the 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/fa1f492d-3249-430e-bc39-42ccac2509e2/929bc4db1daa369abe4d74d44b3dbc1f2772aafb2e5bf6da1f0fb6fffbd9869e.jpg)



(a) Alphabet


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/fa1f492d-3249-430e-bc39-42ccac2509e2/c21caf9c16fa8cec6c4344b7fd618383d93c3b77c85b8fc5c517fb2e2370c207.jpg)



(b) Pixel


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/fa1f492d-3249-430e-bc39-42ccac2509e2/b4c9c3412def10ad88979bce9ffcd1db517327cb2c07785d08b77134f1368455.jpg)



(c) One-dimensional


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/fa1f492d-3249-430e-bc39-42ccac2509e2/0ddb480e7e92b3e476cc92faf67bdb759e7b2aec643d6317b81b836b8bf9b257.jpg)



(d) Cartesian (Ours)



Figure 12: Examples of different coordinate formats.
