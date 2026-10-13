# Advancing Fine-Grained Visual Understanding with Multi-Scale Alignment in Multi-Modal Models

Wei Wang<sup>*</sup> <sup>1,2</sup>, Zhaowei Li<sup>*</sup> <sup>2,3</sup>, Qi Xu<sup>2</sup>, Linfeng Li<sup>2</sup>, Yiqing Cai<sup>2</sup>, Botian Jiang<sup>2,3</sup>, Hang Song<sup>2</sup>, Xingcan Hu<sup>1</sup>, Pengyu Wang<sup>3</sup>, Li Xiao<sup>†1</sup> 

<sup>1</sup>MoE Key Laboratory of Brain-Inspired Intelligence Perception and Cognition University of Science and Technology of China, <sup>2</sup>ByteDance Inc, <sup>3</sup>Fudan University wangweiii@mail.ustc.edu.cn, lizhaowei126@gmail.com 

## Abstract

Multi-modal large language models (MLLMs) have achieved remarkable success in finegrained visual understanding across a range of tasks. However, they often encounter significant challenges due to inadequate alignment for fine-grained knowledge, which restricts their ability to accurately capture local details and attain a comprehensive global perception. While recent advancements have focused on aligning object expressions with grounding information, they typically lack explicit integration of object images, which contain affluent information be yond mere texts or coordinates. To bridge this gap, we introduce a novel fine-grained visual knowledge alignment method that effectively aligns and integrates multi-scale knowledge of objects, including texts, coordinates, and im ages. This innovative method is underpinned by our multi-scale fine-grained enhancement data synthesis pipeline, which provides over 300K essential training data to enhance alignment and improve overall performance. Furthermore, we present TinyGroundingGPT, a series of compact models optimized for highlevel alignments. With a scale of approximately 3B parameters, TinyGroundingGPT achieves outstanding results in grounding tasks while delivering performance comparable to larger MLLMs in complex visual scenarios. The data and code will be released in https://github. com/wwangweii/TinyGroundingGPT.git. 

## 1 Introduction

Recent advancements in multi-modal large language models (MLLMs) have showcased remarkable capabilities in multi-modal understanding, reasoning, and interaction, garnering unprecedented attention (Touvron et al., 2023; Bai et al., 2023; Li et al., 2025b,a; Zheng et al., 2025b,a; Jian et al., 2025a, 2024). MLLM research in finegrained visual understanding has advanced significantly, particularly through early contributions from Shikra (Chen et al., 2023a) and Kosmos-2 (Peng et al., 2023) in textually formatting positional vocabularies or object coordinates. Subsequent studies aimed at improving model performance primarily focused on common strategies, including parameter enlargement (Chen et al., 2023a; Peng et al., 2023; Li et al., 2024c; Bai et al., 2023) and dataset enrichment (Chen et al., 2023b; Bai et al., 2023; Wang et al., 2023; Chen et al., 2024b). Additionally, there is a growing interest in developing efficient, smaller fine-grained MLLMs (Li et al., 2024a; Hu et al., 2024; Yao et al., 2024; Zhu et al., 2023; Zhou et al., 2024) for real-world applications. Regardless of methods used, the core of finegrained models lies in achieving better alignment between object texts and visual features, encompassing both coordinate and semantic information. 


Multi-scale object representations


<table><tr><td rowspan="2"><img src="https://cdn-mineru.openxlab.org.cn/result/2026-09-17/cf9cee5f-3a6e-45d0-b53b-d5da013efb93/2b645394b1076ab5f8aed0a6fda53d1c254bc2158c3a6496e4d440b8526111de.jpg"/></td><td colspan="3">Coordinate“[x1, y1, x2, y2]”</td><td colspan="2">Text“A small cat with grey and white fur”</td><td colspan="2">Image“”</td></tr><tr><td colspan="7">Information density</td></tr><tr><td>Models</td><td>C-T</td><td>C-I</td><td>T-C</td><td>T-I</td><td>I-T</td><td>I-C</td><td></td></tr><tr><td>SAM</td><td>X</td><td>X</td><td>✓</td><td>X</td><td>X</td><td>✓</td><td></td></tr><tr><td>LLaVA</td><td>X</td><td>X</td><td>X</td><td>X</td><td>✓</td><td>X</td><td></td></tr><tr><td>Shikra, GroundingGPT</td><td>✓</td><td>X</td><td>✓</td><td>X</td><td>✓</td><td>X</td><td></td></tr><tr><td>Qwen2-VL, InternVL2</td><td>✓</td><td>X</td><td>✓</td><td>✓</td><td>✓</td><td>X</td><td></td></tr><tr><td>Ours</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td></td></tr></table>


Figure 1: The comparison of alignment for multi-scale object representations. The C, T, I denote object coordinates, texts and images respectively. The “X-Y” denote MLLMs handle input “X” and output “Y”.


While effective, these methods face a significant challenge, i.e., the lack of fine-grained alignments. Visual objects typically encompass multi-scale representations with varying levels of information, including coordinates, texts, and images, as illustrated in Fig. 1. In this context, coordinates provide low-level object grounding information, texts offer primary descriptions that may not capture every detail, and images convey high-level information that extends beyond words. Most fine-grained models (Chen et al., 2023a; You et al., 2023; Li et al., 2024c) primarily focus on alignments between object texts and coordinates (i.e., T-C and C-T), often neglecting direct interactions with object images. Although recent models like Qwen2-VL (Wang et al., 2024a) and InternVL2 (Chen et al., 2024b) can process multiple image inputs and understand relationships between the main image and object images (T-I), they still struggle to establish explicit alignments among object texts, coordinates, and images. This limitation can lead to hallucinations and insufficient grounding capabilities (Chen et al., 2024a). 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/cf9cee5f-3a6e-45d0-b53b-d5da013efb93/da5c50f34af97b183ca1b21daaa1a556f3dd4869cdd3f08680f886aaf0ba7ff2.jpg)



Figure 2: Illustration of the proposed multi-modal fine-grained visual knowledge alignment method. It adopts a three-stage training strategy that progresses from easy to hard and the multi-scale fine-grained enhancement data synthesis pipeline constructs over 300K fine-grained alignment data.


To achieve high-level alignments and integrate multi-granularity knowledge, as illustrated in Fig. 2(a), we introduce a fine-grained visual knowledge alignment method that effectively aligns object texts, coordinates, and images across multiple scales. Our method adopts a three-stage training strategy that progresses from easy to hard: 1) Object and Relation Perception Pretraining: To develop a foundational understanding of object texts and images, we implement a progressive training approach for MLLMs based on a pretrained LLM. 2) Multi-scale Fine-grained Local Knowledge Alignment: To attain fine-grained visual understanding and share multi-scale object knowledge, we conduct data-driven high-level alignments among object text descriptions, bounding box coordinates, and image features. 3) Detailed Global Knowledge Alignment: To enhance the model’s global understanding by integrating fine-grained knowledge, we guide the MLLMs to bridge different objects with multi-scale representations. To support this method, we propose a multi-scale fine-grained enhancement data synthesis pipeline (see Fig. 2(b)) that constructs alignment data from both local and global perspectives. Leveraging this framework, we propose TinyGroundingGPT, which requires less storage for deployment while outperforming larger models across multiple benchmarks, particularly in grounding tasks. Our contributions can be summarized as follows: 

• We introduce a fine-grained visual knowledge alignment method that enables the model to progressively enhance its fine-grained visual understanding through both global and local multi-scale object alignments. 

• We develop a multi-scale fine-grained enhancement data synthesis pipeline that leverages open-source datasets and advanced models to generate over 300K essential training data for fine-grained alignment. 

• We introduce TinyGroundingGPT, a series of compact models with 1.5B and 3B parameters, which excel in multi-modal understanding and grounding capabilities, achieving performance comparable to larger 7B MLLMs. 

## 2 Related Works

Multi-modal Large Language Models Recent progress in large language models (LLMs) such as ChatGPT and LLaMA (Touvron et al., 2023) has spurred the development of multi-modal LLMs. Notable models like GPT-4V (OpenAI, 2023) have demonstrated strong multi-modal capabilities in visual tasks. Early open-source models, including BLIP-2 (Li et al., 2023a), MiniGPT-4 (Zhu et al., 2023), and LLaVA (Liu et al., 2024), leverage pre-trained LLMs and perform well in visual question answering. Subsequent models, such as Qwen-VL (Bai et al., 2023), InternVL (Chen et al., 2024b), and MiniCPM-V (Yao et al., 2024), further enhance capabilities through dynamic resolution, expanded training data, and reinforcement learning, achieving notable results in OCR and grounding while improving response credibility. 

However, the high parameter counts of MLLMs lead to significant training and deployment costs, limiting their widespread use. To address this, lightweight LLMs such as Mini-Gemini (Li et al., 2024a), MobileVLM (Chu et al., 2024), and MiniCPM-V (Yao et al., 2024) have been developed. These models, combined with optimized structures and training strategies, achieve performance comparable to larger models. Additionally, studies (Hsieh et al., 2023; Wang et al., 2024b; Shu et al., 2024) have explored distilling capabilities from larger models to enable smaller models to acquire complex reasoning abilities. 

Fine-grained Multi-modal Models Recent works have focused on MLLMs for fine-grained understanding, with applications in tasks like grounding and OCR. Region-level understanding MLLMs (Yuan et al., 2024; Guo et al., 2024; Lu et al., 2023; Jian et al., 2025b) achieved local alignment between specific region features and texts. Methods such as Shikra (Chen et al., 2023a) and Kosmos-2 (Peng et al., 2023) enhanced visual grounding by constructing datasets with coordinate information, often converting visual tasks into instruction-following formats using templates. Other approaches integrated additional visual components, like GLaMM (Rasheed et al., 2024) and LLaVA-Grounding (Zhang et al., 2023b), or extracted regional features as supplementary inputs, as seen in Ferret (You et al., 2023), NExT-Chat (Zhang et al., 2023a), and GPT4RoI (Zhang et al., 2023c). GroundingGPT (Li et al., 2024c) extended support for multi-modal grounding tasks. Models like VisionLLMv2 (Wu et al., 2024) and UnifiedMLLM (Li et al., 2024b) expanded capabilities for various visual tasks, including image editing and segmentation. For fine-grained tasks, models such as LLaVA-UHD (Xu et al., 2024) and Qwen2-VL (Wang et al., 2024a) explored dynamic high-resolution techniques, improving OCR results. However, these models often lack systematic alignment among object texts, coordinates, and images, limiting the integration of multi-scale representations. 

## 3 Method

In this paper, we first introduce a novel fine-grained visual knowledge alignment method that harnesses the potential of MLLMs by aligning object texts, coordinates, and images across multiple scales, as shown in Fig. 2(a). Our method consists of three training stages that progress from easy to hard: (a) Object and Relation Perception Pretraining, which enables the model to understand multimodal inputs, identifying objects in images and their interrelations. (b) Multi-scale Fine-grained Local Knowledge Alignment by which the model is guided to achieve multi-scale, fine-grained alignments, accommodating diverse inputs such as object texts, coordinates, and images. (c) Detailed Global Knowledge Alignment which focuses on model training for global alignment and understanding, further integrating fine-grained information and bridging different objects with multi-scale representations. To support this high-level alignment, we then propose a multi-scale fine-grained enhancement data synthesis pipeline, as illustrated in Fig 2(b), which generates multi-scale alignment datasets from both global and local perspectives. Building on this framework, we propose Tiny-GroundingGPT, which requires less storage for deployment while outperforming larger parameter models across multiple benchmarks, particularly in hallucination evaluation and grounding tasks. 

## 3.1 Fine-grained Visual Knwoledge Alignment

We elaborate the three training stages in our finegrained visual knowledge alignment method below. 

Object and Relation Perception Pretraining In this stage, we aim for the model to comprehend multi-modal inputs, recognizing the objects present in the image and the relationships among them, which forms the foundation for subsequent reasoning and grounding tasks. Throughout the training process, we initially keep the LLM and encoder frozen, training only the projector to connect the text and image semantic spaces. Subsequently, we train both the LLM and the projector to enhance the understanding of objects and their relationships. We utilize LLaVA-Pretrain-595k (Liu et al., 2024) and each sample is accompanied by a sampled instruction that requires the model to provide a concise description of the image. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/cf9cee5f-3a6e-45d0-b53b-d5da013efb93/1bb992c594d6c77942a012f458dcba17b1770c06a7b4f373037a4ec5e31607db.jpg)



Figure 3: The model architecture of our proposed TinyGroundingGPT. It utilizes multi-scale visual encoders and supports queries regarding different object representations. Object images are cropped and zoomed from the input image according to the input coordinates.


Multi-scale Fine-grained Local Knowledge Alignment After the initial training stage, where the model learns to recognize objects and their relationships, it still lacks the grounding capability to accurately locate these objects in images and to integrate different representations of a single object. In this stage, we therefore train the model to achieve fine-grained alignments among object texts, coordinates, and images, fully sharing their multi-scale knowledge for each representation. We utilize original visual grounding datasets such as RefCOCO (Kazemzadeh et al., 2014), RefCOCO+ (Kazemzadeh et al., 2014), RefCOCOg (Mao et al., 2016) and Visual Genomes (Krishna et al., 2017), along with a developed multi-scale fine-grained enhancement data synthesis pipeline (details provided in the following subsection) to construct a fine-grained grounding dataset. The instances in the training data can be categorized into three classes: 

• Object Texts and Coordinates Alignment: The model refers to corresponding coordinates for a given object text description or describes a region based on input coordinates. 

• Object Images and Coordinates Alignment: Given an augmented object image, the model identifies its location within the image. When provided with coordinates, the model selects the most relevant object images. 

• Object Texts and Images Alignment: The model selects the most relevant augmented object image based on the input question or answers inquiries about the relationships involving augmented object images. 

Throughout the training process, we train both the LLM and the projector. Afterwards, the model can effectively perform fine-grained image understanding by achieving high-level alignments among object texts, images, and coordinates, while sharing multi-scale knowledge across each representation. 

Detailed Global Knowledge Alignment Despite achieving a fine-grained understanding of multimodal data in the previous stage, the model lacks systematic training for global image comprehension and the ability to connect different objects with varied representations. Specifically, in the previous stage, only the representations of individual objects in each training sample were aligned. In this stage, our goal is to further align and integrate multiple objects within a single image input to enhance global knowledge learning. To achieve this, in addition to utilizing common image annotation datasets for instruction tuning, including LLaVA-v1.5-mix665k (Liu et al., 2024) and ShareGPT4V (Chen et al., 2023b), we construct a global grounding dataset with high-level fine-grained alignments based on Flickr30K Entities (Plummer et al., 2015): 1) Multi-round Grounding Conversation Data: This dataset guides the model to achieve a global understanding of the image through multi-round conversations, requiring it to combine fine-grained knowledge and thoroughly explore the relationships among different representations of various objects. 2) Grounding Description Data: This dataset prompts the model to provide a detailed description of the image to connect multi objects in one-round conversations, where the generated object texts are enhanced with coordinates to confirm their existence and effectively integrate grounding information. 

This method enables us to leverage the finegrained grounding alignment learned in the second stage to enhance the model’s global grounding alignment. Additionally, we train both the LLM and the projector in this stage. 

## 3.2 Multi-scale Fine-grained Enhancement Data Synthesis Pipeline

As shown in Fig. 2(b), we develop a multi-scale fine-grained enhancement data synthesis pipeline, and construct a multi-scale fine-grained grounding dataset (in Stage2) as well as a global grounding dataset (in Stage3). Specifically, given an image, we perform the following steps: 

Object Recognition We employ expert models or MLLMs for object detection in input images, generating a list of identified objects, referred to as $L _ { 1 }$ A prompt example for GPT-4V (OpenAI, 2023) is provided in Appendix Fig.7. 

Object Grounding Beyond object text and coordinate pairs in original datasets such as RefCOCO, we apply grounding models to obtain bounding box coordinates. In this paper, we employ GroundingDINO (Liu et al., 2023b) to locate objects in $L _ { 1 }$ and filter out those with low confidence, resulting in an object bounding box dictionary $S _ { 1 }$ 

Relationship Extraction To uncover the relationships between objects for subsequent QA generation, we instruct GPT-4V to identify potential connections among objects. As shown in $\mathsf { A p - }$ pendix Fig. 7, given the object list $L _ { 1 }$ , GPT-4V generates a list $L _ { 2 }$ containing triples in the format (object1, relation, object2). 

QA Generation Based on above $L _ { 1 } , S _ { 1 }$ and $L _ { 2 }$ , we use task-specific prompts for GPT-4V to generate different kinds of datasets (we provide case examples in Appendix Figs. 8 and 9): (1) 256K Additional Multi-scale Fine-grained Grounding Dataset: 

Compared to previous works (Li et al., 2024c; Chen et al., 2023a) that focused solely on the alignment between object texts and coordinates, we enhance the alignment format by constructing an additional multi-scale, fine-grained dataset. This dataset incorporates object images, texts, and coordinates, facilitating more fine-grained image understanding and multi-scale alignment. Details can be seen in $\mathsf { A p - }$ pendix Fig. 10. (2) 57K Global Grounding Dataset: To enhance the global alignment and bridge objects with various representations, we construct two kinds of datasets: 1) 28K Multi-round Grounding Conversation: This dataset includes multi-turn dialogue formats, focusing on point-to-point questions about local details. 2) 29K Grounding Description: This dataset features single-turn dialogue formats, emphasizing an understanding of overall image descriptions with fine-grained grounding information. See details in Appendix Figs. 11 and 12. 

Filter We filter out QAs that contain object images with areas that are either too large or too small, as well as those with high Intersection over Union among object images in the options. Additionally, we exclude QAs related to objects with low confidence or those with an excessive number of bounding boxes. This helps avoid low-resolution noise or image reference ambiguity. Finally, we exclude low-quality QAs with the assistance of GPT-4V. 

## 3.3 TinyGroundingGPT

Using our proposed alignment method and synthesis data, we train TinyGroundingGPT to demonstrate the effectiveness of our method. Fig. 3 illustrates the overall architecture of TinyGroundingGPT. Images in various formats are processed through multi-scale vision encoders to extract features. Specifically, we use the pre-trained visual encoders ViT-L/14 (Radford et al., 2021) and DINOv2-L/14 (Oquab et al., 2023) to extract image features, concatenating them to combine the global perception of CLIP and the local fine-grained understanding of DINOv2 (Jiang et al., 2023). These features are then mapped to the LLM embedding space using an MLP. In TinyGroundingGPT, the input supports both global images and object images, each represented by different special tokens: < image > and < object >. These object images are cropped and zoomed from the global image based on the corresponding coordinates. We also support the input and output of object bounding box coordinates < loc >, represented in the text format [x1, y1, x2, y2] with values in [0.000, 1.000]. 

<table><tr><td rowspan="2">Type</td><td rowspan="2">Model</td><td rowspan="2">LLM Size</td><td colspan="3">RefCOCO</td><td colspan="3">RefCOCO+</td><td colspan="2">RefCOCOg</td><td rowspan="2">Avg</td></tr><tr><td>val</td><td>testA</td><td>testB</td><td>val</td><td>testA</td><td>testB</td><td>val</td><td>test</td></tr><tr><td rowspan="3">Specialist</td><td>UNITER (Chen et al., 2020)</td><td>-</td><td>81.41</td><td>87.04</td><td>74.17</td><td>75.90</td><td>81.45</td><td>66.70</td><td>74.02</td><td>68.67</td><td>76.17</td></tr><tr><td>MDETR (Kamath et al., 2021)</td><td>-</td><td>86.75</td><td>89.58</td><td>81.41</td><td>79.52</td><td>84.09</td><td>70.62</td><td>81.64</td><td>80.89</td><td>81.81</td></tr><tr><td>UniTAB (Yang et al., 2022)</td><td>-</td><td>86.32</td><td>88.84</td><td>80.61</td><td>78.70</td><td>83.22</td><td>69.48</td><td>79.96</td><td>79.97</td><td>80.89</td></tr><tr><td rowspan="7">Generalist</td><td>KOSMOS-2 (Peng et al., 2023)</td><td>1.6B</td><td>52.32</td><td>57.42</td><td>47.26</td><td>45.48</td><td>50.73</td><td>42.24</td><td>60.57</td><td>61.65</td><td>52.21</td></tr><tr><td>Shikra (Chen et al., 2023a)</td><td>7B</td><td>87.01</td><td>90.61</td><td>80.24</td><td>81.60</td><td>87.36</td><td>72.12</td><td>82.27</td><td>82.19</td><td>82.93</td></tr><tr><td>NExT-Chat* (Zhang et al., 2023a)</td><td>7B</td><td>85.50</td><td>90.00</td><td>77.90</td><td>77.20</td><td>84.50</td><td>68.00</td><td>80.10</td><td>79.80</td><td>80.38</td></tr><tr><td>Ferret* (You et al., 2023)</td><td>7B</td><td>87.49</td><td>91.35</td><td>82.45</td><td>80.78</td><td>87.38</td><td>73.14</td><td>83.93</td><td>84.76</td><td>83.91</td></tr><tr><td>GroundingGPT (Li et al., 2024c)</td><td>7B</td><td>88.02</td><td>91.55</td><td>82.47</td><td>81.61</td><td>87.18</td><td>73.18</td><td>81.67</td><td>81.99</td><td>83.46</td></tr><tr><td>InternVL2+ (Chen et al., 2024b)</td><td>2B</td><td>82.3</td><td>88.2</td><td>75.9</td><td>73.5</td><td>82.8</td><td>63.3</td><td>77.6</td><td>78.3</td><td>77.74</td></tr><tr><td>Qwen2-VL+ (Wang et al., 2024a)</td><td>2B</td><td>87.6</td><td>90.6</td><td>82.3</td><td>79.0</td><td>84.9</td><td>71.0</td><td>81.2</td><td>80.3</td><td>82.11</td></tr><tr><td rowspan="2">Generalist</td><td rowspan="2">TinyGroundingGPT</td><td>3B</td><td>89.16</td><td>92.24</td><td>85.38</td><td>81.70</td><td>87.16</td><td>75.09</td><td>83.27</td><td>84.08</td><td>84.76</td></tr><tr><td>1.5B</td><td>86.76</td><td>90.42</td><td>81.81</td><td>78.86</td><td>84.65</td><td>70.24</td><td>79.88</td><td>80.04</td><td>81.58</td></tr></table>


Table 1: Performance comparison on the referring expression comprehension(REC) task. "*" indicates that the model employs additional image region perception modules and "+" indicates that the model uses dynamic high resolution. The best results are highlighted in bold, while the second-best results are underlined.


## 4 Experiments

## 4.1 Experimental Setup

We employ Qwen2.5-3B and Qwen2.5-1.5B (Yang et al., 2024) as the language models for our TinyGroundingGPT. For more details on hyperparameter settings, training processes and datasets, please refer to the Appendix A.1 and A.2. 

## 4.2 Image Grounding Evaluation

To evaluate the image grounding capability of Tiny-GroundingGPT, we conducted experiments on the Reference Expression Understanding (REC) task, which involves locating the bounding box for a given text reference. We utilized three datasets: RefCOCO, RefCOCO+, and RefCOCOg. We compared TinyGroundingGPT against various baseline models, including end-to-end multi-modal models such as UNITER (Chen et al., 2020), MDETR (Kamath et al., 2021), and UniTAB (Yang et al., 2022), as well as LLM-based models like KOSMOS-2, Shikra, NExTChat, Ferret, and GroundingGPT. Additionally, smaller models such as InternVL2 and Qwen2-VL were included. We used a unified prompt formatted as “Output the coordinate of < exp >”, where “< exp >” represents the reference expression. As shown in Table 1, TinyGroundingGPT demonstrates strong performance across all datasets, even with smaller LLM sizes (3B and 1.5B), matching or exceeding the performance of specialized fine-tuned models and larger MLLMs with additional image perception modules. Notably, the 3B model achieved state-of-the-art results on several benchmarks, attaining the highest average accuracy. Furthermore, TinyGroundingGPT-1.5B showed comparable grounding results, outperforming Next-Chat-7B on nearly all test sets. 

## 4.3 Image Understanding Evaluation

We evaluated TinyGroundingGPT on seven benchmarks, providing a comprehensive assessment of its performance across various metrics. As shown in Table 2, TinyGroundingGPT-3B achieves results comparable to models such as MiniCPM-V-2 and InternVL-2, which utilize dynamic high resolution or enriched training data. Compared to models with similar fine-tuning data, including LLaVA-1.5, GroundingGPT, TinyLLaVA, and LLaVA-Phi, TinyGroundingGPT-3B demonstrates superior image understanding capabilities on the VQA<sup>v2</sup>, GQA, SQA, and POPE benchmarks, achieving increases of 2.6% on MMB and 1.2% on GQA over GroundingGPT-7B. Notably, TinyGroundingGPT-1.5B outperforms LLaVA-Phi, despite its larger parameter count, on most benchmarks. We further evaluated the MLLMs for object hallucination on the POPE benchmark, with more details provided in Appendix A.3. Overall, TinyGroundingGPT, optimized by our multi-scale visual knowledge alignment method, achieved impressive results across multiple evaluation sets. 

## 4.4 Ablation Study

Ablation Study on Additional Multi-scale Fine-grained Grounding Dataset. In Stage 2, compared to traditional methods that rely solely on alignment datasets for object texts and coordinates, we utilized our constructed multi-scale fine-grained grounding datasets for TinyGroundingGPT, enabling multi-scale alignment among object texts, images, and coordinates. The ablation study in Table 3 shows that our proposed multi-scale fine-grained alignment outperforms traditional referring data that only aligns object texts with coordinates. For both the 3B and 1.5B TinyGroundingGPT models, our method enhanced performance on the RefCOCO, RefCOCO+, and RefCOCOg benchmarks. For instance, on the RefCOCO+ benchmark, our method achieved an increase of 1.67% for the 3B model and 0.87% for the 1.5B model, demonstrating the effectiveness of our proposed fine-grained alignments and datasets. 

<table><tr><td>Models</td><td>LLM Size</td><td><eq>VQA^{v2}</eq></td><td>GQA</td><td><eq>SQA^I</eq></td><td>POPE</td><td><eq>MME^P</eq></td><td>MMB</td><td><eq>LLaVA^W</eq></td></tr><tr><td>BLIP-2 (Li et al., 2023a)</td><td>13B</td><td>41.0</td><td>41</td><td>61</td><td>85.3</td><td>1293.8</td><td>-</td><td>38.1</td></tr><tr><td>InstructBLIP (Dai et al., 2023)</td><td>7B</td><td>-</td><td>49.2</td><td>60.5</td><td>-</td><td>-</td><td>36</td><td>60.9</td></tr><tr><td>InstructBLIP (Dai et al., 2023)</td><td>13B</td><td>-</td><td>49.5</td><td>63.1</td><td>78.9</td><td>1212.8</td><td>-</td><td>58.2</td></tr><tr><td>Shikra (Chen et al., 2023a)</td><td>13B</td><td>77.4</td><td>-</td><td>-</td><td>-</td><td>-</td><td>58.8</td><td>-</td></tr><tr><td>LLaVA-1.5 (Liu et al., 2023a)</td><td>7B</td><td>78.5</td><td>62.0</td><td>66.8</td><td>85.9</td><td>1510.7</td><td>64.3</td><td>63.4</td></tr><tr><td>GroundingGPT (Li et al., 2024c)</td><td>7B</td><td>78.7</td><td>62.1</td><td>-</td><td>87.4</td><td>1454.2</td><td>63.8</td><td>70.9</td></tr><tr><td>Qwen-VL-Chat (Bai et al., 2023)</td><td>7B</td><td>78.2</td><td>-</td><td>68.2</td><td>-</td><td>1487.5</td><td>60.6</td><td>-</td></tr><tr><td>MiniCPM-V-2+ (Yao et al., 2024)</td><td>2.8B</td><td>-</td><td>-</td><td>-</td><td>87.8</td><td>-</td><td>69.6</td><td>69.2</td></tr><tr><td>InternVL-2+ (Chen et al., 2024b)</td><td>2B</td><td>-</td><td>61.0</td><td>-</td><td>88.3</td><td>1439.6</td><td>-</td><td>62.5</td></tr><tr><td>LLaVA-Phi (Zhu et al., 2024)</td><td>2.7B</td><td>71.4</td><td>-</td><td>68.4</td><td>86.7</td><td>1335.1</td><td>59.8</td><td>-</td></tr><tr><td>TinyLLaVA (Zhou et al., 2024)</td><td>2.7B</td><td>77.7</td><td>61.0</td><td>70.1</td><td>86.3</td><td>1437.3</td><td>68.3</td><td>67.1</td></tr><tr><td rowspan="2">TinyGroundingGPT</td><td>3B</td><td>79.3</td><td>63.3</td><td>70.3</td><td>87.9</td><td>1423.2</td><td>66.4</td><td>67.5</td></tr><tr><td>1.5B</td><td>77.9</td><td>62.2</td><td>63.1</td><td>87.6</td><td>1392.4</td><td>64.2</td><td>65.3</td></tr></table>


Table 2: Comparison of MLLMs on image understanding benchmarks. Benchmark names are abbreviated due to space limits. $\mathsf { V Q A } ^ { \mathrm { v } 2 }$ (Goyal et al., 2017); GQA (Hudson and Manning, 2019); SQA<sup>I</sup>:ScienceQA-IMG (Lu et al., 2022); POPE (Li et al., 2023b); MME (Fu et al., 2023); MMB:MMBench (Liu et al., 2025); LLaVA<sup>W</sup>: LLaVA-Bench (In-the-Wild) (Liu et al., 2024). "+" indicates that the model uses dynamic high-resolution.


<table><tr><td>Size</td><td>Alignment</td><td>RefCOCO</td><td>RefCOCO+</td><td>RefCOCOg</td></tr><tr><td rowspan="2">3B</td><td>T, C</td><td>87.35</td><td>78.89</td><td>83.25</td></tr><tr><td>T, C, I</td><td>88.50</td><td>80.56</td><td>83.69</td></tr><tr><td rowspan="2">1.5B</td><td>T, C</td><td>85.93</td><td>77.05</td><td>79.54</td></tr><tr><td>T, C, I</td><td>86.33</td><td>77.92</td><td>79.96</td></tr></table>


Table 3: Ablation study on our Additional Multi-scale Fine-grained Grounding Dataset in Stage 2. Here, T, C, and I denote Text, Coordinate, and Image, respectively. We report the average accuracy for each benchmark.


Ablation Study on Global Grounding Datasets. In Stage 3, we utilized the constructed Global Grounding Datasets for TinyGroundingGPT to bridge different objects with varied representations and enhance global image comprehension. The results presented in Table 4 showcased the effectiveness of this strategy. Notably, a reduction in hallucinations can be observed on the POPE benchmark. Overall, significant improvements in visual understanding benchmarks underscored the value of detailed global knowledge learning and the Global 


Grounding Datasets, which enhanced global object alignment by connecting different objects represented by texts, coordinates, and images.


<table><tr><td>Size</td><td>Global Align</td><td>GQA</td><td><eq>VQA^{v2}</eq></td><td>SQA</td><td>POPE</td><td>MMB</td></tr><tr><td rowspan="2">3B</td><td>×</td><td>61.7</td><td>77.4</td><td>65.6</td><td>86.6</td><td>63.1</td></tr><tr><td>√</td><td>63.3</td><td>79.3</td><td>70.3</td><td>87.9</td><td>66.4</td></tr><tr><td rowspan="2">1.5B</td><td>×</td><td>60.3</td><td>77.3</td><td>62.1</td><td>86.4</td><td>63.0</td></tr><tr><td>√</td><td>62.2</td><td>77.9</td><td>63.1</td><td>87.6</td><td>64.2</td></tr></table>


Table 4: Ablation study on our Global Grounding Datasets in Stage 3. If the model is trained without global alignment, it indicates that we do not use these datasets to further align different objects represented by texts, coordinates, and images.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/cf9cee5f-3a6e-45d0-b53b-d5da013efb93/61b1f125baca2bf0ecd1b4e2b4a97519e17076dcd57d88d2d4d4d4a4765dacf0.jpg)



Figure 4: A case for the outputs of our TinyGrounding-GPT when the input is either enhanced with coordinates or not. Probability values indicate the likelihood of generating corresponding tokens.


Ablation Study on Models. (1) Larger parameters: We applied our method to TinyGroundingGPT with the larger language model Qwen2.5-7B (Appendix A.4). (2) Vision encoders: We explored the effects of our multi-scale vision encoders (ViT and DINOv2) (Appendix A.5). (3) Qwen2 as pretrained LLM: We compared Qwen2.5 with Qwen2, demonstrating Qwen2.5’s effectiveness (Appendix A.6). 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/cf9cee5f-3a6e-45d0-b53b-d5da013efb93/1ffd8b3a92543efd43adadd7ee45116d9a16c2d8305988efd98c8ede1dde24cd.jpg)



Figure 5: Visualization of the attention map for image patches with different object representation outputs (texts, coordinates, and images, underlined). The red bounding box denotes the target region. The attention at the four corners serves as anchors for grounding, while attention at specific objects highlights their importance.


Overall, the results highlight the effectiveness of our proposed method and TinyGroundingGPT. 

## 5 Discussion

## 5.1 Effectiveness of Fine-grained Knowledge

Our fine-grained visual knowledge alignment method not only improves grounding ability but also enhances comprehensive image understanding. Examples of image descriptions generated by TinyGroundingGPT are provided in Appendix Fig. 13, demonstrating the model’s ability to avoid incorrect or nonexistent object descriptions. We evaluated the annotation quality by selecting 50 images from RefCOCO-test and using GPT-4V to score descriptions from different models. As detailed in Appendix A.7, TinyGroundingGPT-3B outperforms GroundingGPT-7B and Qwen2-VL-2B in overall quality and richness. Additionally, incorporating fine-grained knowledge into input questions for TinyGroundingGPT results in more accurate and persuasive responses. As shown in Fig. 4, adding coordinates to object texts in queries enhances response confidence compared to directly asking about objects in an image. This underscores the potential of fine-grained MLLMs. 

## 5.2 Interpretability for High-level Alignments

Grounding MLLMs fundamentally model the maximum likelihood output based on visual inputs and text prompts. By conditioning on the referring prompt, the model identifies which parts of the image significantly influence the output. To demonstrate the effectiveness of our multi-scale fine-grained grounding capability, we visualize the attention map in the last layer of our TinyGroundingGPT. As shown in Fig. 5, the attention maps of our TinyGroundingGPT reveal distinct location attributions, unlike the baseline GroundingGPT-7B. 

For object coordinates, high attention scores are concentrated at the four corners of the image, serving as anchors for bounding box coordinates, as well as at the locations of the intended objects mentioned in the prompt. When prompted to describe a specific region, the model directs increased attention to the corresponding object patches. For the output of an object image, the attention values between image patches and the target object highlight relevant regions and reinforce grounding anchors. This indicates that TinyGroundingGPT effectively learns both aligned features and grounding information for object images. In summary, our findings underscore the effectiveness of the proposed fine-grained visual knowledge alignment method, achieving high-level alignment among different object representations. This provides insights for further explaining MLLMs, particularly in grounding tasks. More visualizations can be found in Appendix Fig. 14. 

## 6 Conclusion

In this paper, we introduce a novel fine-grained visual knowledge alignment method for MLLMs to address the limitations of fine-grained alignments in previous works. Our method progresses from easy to hard, emphasizing multi-scale fine-grained alignments among object texts, coordinates, and images from both local and global perspectives. This empowers models to effectively learn fine-grained knowledge and facilitates reasoning and grounding tasks. Additionally, we develop a multi-scale finegrained enhancement data synthesis pipeline that leverages open-source datasets and advanced models to generate over 300K essential training samples. Building on this foundation, we train Tiny-

GroundingGPT, a series of smaller models (1.5B and 3B parameters) optimized through high-level alignments, capable of handling various visual and grounding tasks, often surpassing larger models. Experimental results demonstrate the effectiveness of our proposed method and the generated datasets. Our work contributes to the advancement of practical applications for MLLMs. 

## Limitations

Our work has developed a fine-grained visual knowledge alignment method for MLLMs. Based on this, we constructed the necessary datasets and trained our proposed TinyGroundingGPT. There are several aspects that can be further improved: (1) Additional techniques can be applied to TinyGroundingGPT to further enhance its performance. For example, the dynamic high-resolution in works (Xu et al., 2024; Chen et al., 2024b) has been proved to improve image understanding capabilities. (2) Additional datasets, such as OCR datasets described in (Wang et al., 2024a), can be utilized for supervised fine-tuning of TinyGroundingGPT in Stage3 to further enhance its multimodality capabilities. (3) Our current multi-scale alignments focus only on objects within a single image. This approach can be further extended to include similar objects or objects captured from different angles across multiple images, thereby enhancing the robustness and generality of alignment. 

## Acknowledgments

This work was supported by the National Natural Science Foundation of China under Grant 62202442. 

## References



Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. 2023. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966. 





Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. 2023a. Shikra: Unleashing multimodal llm’s referential dialogue magic. arXiv preprint arXiv:2306.15195. 





Lin Chen, Jisong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. 2023b. Sharegpt4v: Improving large multimodal models with better captions. arXiv preprint arXiv:2311.12793. 





Xiang Chen, Chenxi Wang, Yida Xue, Ningyu Zhang, Xiaoyan Yang, Qiang Li, Yue Shen, Jinjie Gu, and Huajun Chen. 2024a. Unified hallucination detection for multimodal large language models. arXiv preprint arXiv:2402.03190. 





Yen-Chun Chen, Linjie Li, Licheng Yu, Ahmed El Kholy, Faisal Ahmed, Zhe Gan, Yu Cheng, and Jingjing Liu. 2020. Uniter: Universal image-text representation learning. In European conference on computer vision, pages 104–120. Springer. 





Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. 2024b. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. arXiv preprint arXiv:2404.16821. 





Xiangxiang Chu, Limeng Qiao, Xinyu Zhang, Shuang Xu, Fei Wei, Yang Yang, Xiaofei Sun, Yiming Hu, Xinyang Lin, Bo Zhang, et al. 2024. Mobilevlm v2: Faster and stronger baseline for vision language model. arXiv preprint arXiv:2402.03766. 





Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. 2023. Instructblip: Towards general-purpose vision-language models with instruction tuning. arXiv preprint arXiv:2305.06500. 





Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. 2023. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394. 





Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. 2017. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 6904–6913. 





Qiushan Guo, Shalini De Mello, Hongxu Yin, Wonmin Byeon, Ka Chun Cheung, Yizhou Yu, Ping Luo, and Sifei Liu. 2024. Regiongpt: Towards region understanding vision language model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13796–13806. 





Cheng-Yu Hsieh, Chun-Liang Li, Chih-Kuan Yeh, Hootan Nakhost, Yasuhisa Fujii, Alexander Ratner, Ranjay Krishna, Chen-Yu Lee, and Tomas Pfister. 2023. Distilling step-by-step! outperforming larger language models with less training data and smaller model sizes. arXiv preprint arXiv:2305.02301. 





Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Weilin Zhao, et al. 2024. Minicpm: Unveiling the potential of small language models with scalable training strategies. arXiv preprint arXiv:2404.06395. 





Drew A Hudson and Christopher D Manning. 2019. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 6700–6709. 





Pu Jian, Junhong Wu, Wei Sun, Chen Wang, Shuo Ren, and Jiajun Zhang. 2025a. Look again, think slowly: Enhancing visual reflection in vision-language models. Preprint, arXiv:2509.12132. 





Pu Jian, Donglei Yu, Wen Yang, Shuo Ren, and Jiajun Zhang. 2025b. Teaching vision-language models to ask: Resolving ambiguity in visual questions. In Proceedings of the 63rd Annual Meeting of the As sociationfor Computational Linguistics (Volume 1: Long Papers), pages 3619–3638. 





Pu Jian, Donglei Yu, and Jiajun Zhang. 2024. Large language models know what is key visual entity: An llm-assisted multimodal retrieval for vqa. In Proceed ings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 10939–10956. 





Dongsheng Jiang, Yuchen Liu, Songlin Liu, Jin’e Zhao, Hao Zhang, Zhen Gao, Xiaopeng Zhang, Jin Li, and Hongkai Xiong. 2023. From clip to dino: Visual encoders shout in multi-modal large language models. arXiv preprint arXiv:2310.08825. 





Aishwarya Kamath, Mannat Singh, Yann LeCun, Gabriel Synnaeve, Ishan Misra, and Nicolas Carion. 2021. Mdetr-modulated detection for end-toend multi-modal understanding. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1780–1790. 





Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. 2014. Referitgame: Referring to objects in photographs of natural scenes. In Proceedings ofthe 2014 conference on empirical methods in natural language processing (EMNLP), pages 787– 798. 





Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. 2017. Visual genome: Connecting language and vision using crowdsourced dense image annotations. International journal ofcomputer vision, 123:32–73. 





Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. 2023a. Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. In International conference on machine learning, pages 19730–19742. PMLR. 





Junxian Li, Beining Xu, and Di Zhang. 2025a. Iag: Input-aware backdoor attack on vlms for visual grounding. arXiv preprint arXiv:2508.09456. 





Junxian Li, Di Zhang, Xunzhi Wang, Zeying Hao, Jingdi Lei, Qian Tan, Cai Zhou, Wei Liu, Yaotian Yang, Xinrui Xiong, et al. 2025b. Chemvlm: Exploring the power of multimodal large language models in chemistry area. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 415–423. 





Yanwei Li, Yuechen Zhang, Chengyao Wang, Zhisheng Zhong, Yixin Chen, Ruihang Chu, Shaoteng Liu, and Jiaya Jia. 2024a. Mini-gemini: Mining the potential of multi-modality vision language models. arXiv preprint arXiv:2403.18814. 





Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. 2023b. Evaluating object hallucination in large vision-language models. arXiv preprint arXiv:2305.10355. 





Zhaowei Li, Wei Wang, YiQing Cai, Xu Qi, Pengyu Wang, Dong Zhang, Hang Song, Botian Jiang, Zhida Huang, and Tao Wang. 2024b. Unifiedmllm: Enabling unified representation for multi-modal multitasks with large language model. arXiv preprint arXiv:2408.02503. 





Zhaowei Li, Qi Xu, Dong Zhang, Hang Song, Yiqing Cai, Qi Qi, Ran Zhou, Junting Pan, Zefeng Li, Vu Tu, et al. 2024c. Groundinggpt: Language enhanced multi-modal grounding model. In Proceedings of the 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 6657–6678. 





Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. 2023a. Improved baselines with visual instruction tuning. 





Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2024. Visual instruction tuning. Advances in neural information processing systems, 36. 





Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. 2023b. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499. 





Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. 2025. Mmbench: Is your multi-modal model an all-around player? In European Conference on Computer Vision, pages 216–233. Springer. 





Junyu Lu, Dixiang Zhang, Songxin Zhang, Zejian Xie, Zhuoyang Song, Cong Lin, Jiaxing Zhang, Bingyi Jing, and Pingjian Zhang. 2023. Lyrics: Boosting fine-grained language-vision alignment and comprehension via semantic-aware visual objects. arXiv preprint arXiv:2312.05278. 





Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. 2022. Learn to explain: Multimodal reasoning via thought chains for science question answering. In The 36th Conference on Neural Information Processing Systems (NeurIPS). 





Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan L Yuille, and Kevin Murphy. 2016. Generation and comprehension of unambiguous object descriptions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 11–20. 





OpenAI. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774. 





Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. 2023. Dinov2: Learning robust vi sual features without supervision. arXiv preprint arXiv:2304.07193. 





Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. 2023. Kosmos-2: Grounding multimodal large language models to the world. arXiv preprint arXiv:2306.14824. 





Bryan A Plummer, Liwei Wang, Chris M Cervantes, Juan C Caicedo, Julia Hockenmaier, and Svetlana Lazebnik. 2015. Flickr30k entities: Collecting region-to-phrase correspondences for richer imageto-sentence models. In Proceedings of the IEEE international conference on computer vision, pages 2641–2649. 





Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In International confer ence on machine learning, pages 8748–8763. PMLR. 





Hanoona Rasheed, Muhammad Maaz, Sahal Shaji, Abdelrahman Shaker, Salman Khan, Hisham Cholakkal, Rao M Anwer, Eric Xing, Ming-Hsuan Yang, and Fahad S Khan. 2024. Glamm: Pixel grounding large multimodal model. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13009–13018. 





Fangxun Shu, Yue Liao, Le Zhuo, Chenning Xu, Guanghao Zhang, Haonan Shi, Long Chen, Tao Zhong, Wanggui He, Siming Fu, et al. 2024. Llava-mod: Making llava tiny via moe knowledge distillation. arXiv preprint arXiv:2408.15881. 





Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971. 





Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. 2024a. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191. 





Wei Wang, Zhaowei Li, Qi Xu, Yiqing Cai, Hang Song, Qi Qi, Ran Zhou, Zhida Huang, Tao Wang, and Li Xiao. 2024b. Qcrd: Quality-guided contrastive rationale distillation for large language models. arXiv preprint arXiv:2405.13014. 





Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei 





Zhao, Xixuan Song, et al. 2023. Cogvlm: Visual expert for pretrained language models. arXiv preprint arXiv:2311.03079. 





Jiannan Wu, Muyan Zhong, Sen Xing, Zeqiang Lai, Zhaoyang Liu, Wenhai Wang, Zhe Chen, Xizhou Zhu, Lewei Lu, Tong Lu, et al. 2024. Visionllm v2: An end-to-end generalist multimodal large language model for hundreds of vision-language tasks. arXiv preprint arXiv:2406.08394. 





Ruyi Xu, Yuan Yao, Zonghao Guo, Junbo Cui, Zanlin Ni, Chunjiang Ge, Tat-Seng Chua, Zhiyuan Liu, Maosong Sun, and Gao Huang. 2024. Llava-uhd: an lmm perceiving any aspect ratio and high-resolution images. arXiv preprint arXiv:2403.11703. 





An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. 2024. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115. 





Zhengyuan Yang, Zhe Gan, Jianfeng Wang, Xiaowei Hu, Faisal Ahmed, Zicheng Liu, Yumao Lu, and Lijuan Wang. 2022. Unitab: Unifying text and box outputs for grounded vision-language modeling. In European Conference on Computer Vision, pages 521–539. Springer. 





Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. 2024. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800. 





Haoxuan You, Haotian Zhang, Zhe Gan, Xianzhi Du, Bowen Zhang, Zirui Wang, Liangliang Cao, Shih-Fu Chang, and Yinfei Yang. 2023. Ferret: Refer and ground anything anywhere at any granularity. arXiv preprint arXiv:2310.07704. 





Yuqian Yuan, Wentong Li, Jian Liu, Dongqi Tang, Xinjie Luo, Chi Qin, Lei Zhang, and Jianke Zhu. 2024. Osprey: Pixel understanding with visual instruction tuning. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 28202–28211. 





Ao Zhang, Liming Zhao, Chen-Wei Xie, Yun Zheng, Wei Ji, and Tat-Seng Chua. 2023a. Next-chat: An lmm for chat, detection and segmentation. arXiv preprint arXiv:2311.04498. 





Hao Zhang, Hongyang Li, Feng Li, Tianhe Ren, Xueyan Zou, Shilong Liu, Shijia Huang, Jianfeng Gao, Lei Zhang, Chunyuan Li, et al. 2023b. Llava-grounding: Grounded visual chat with large multimodal models. arXiv preprint arXiv:2312.02949. 





Shilong Zhang, Peize Sun, Shoufa Chen, Min Xiao, Wenqi Shao, Wenwei Zhang, Yu Liu, Kai Chen, and Ping Luo. 2023c. Gpt4roi: Instruction tuning large language model on region-of-interest. arXiv preprint arXiv:2307.03601. 





Xiaofan Zheng, Minnan Luo, and Xinghao Wang. 2025a. Unveiling fake news with adversarial arguments generated by multimodal large language models. In Proceedings ofthe 31st International Conference on Computational Linguistics, pages 7862–7869, Abu Dhabi, UAE. Association for Computational Linguistics. 





Xiaofan Zheng, Zinan Zeng, Heng Wang, Yuyang Bai, Yuhan Liu, and Minnan Luo. 2025b. From predictions to analyses: Rationale-augmented fake news detection with large vision-language models. In Pro ceedings ofthe ACM on Web Conference 2025, pages 5364–5375. 





Baichuan Zhou, Ying Hu, Xi Weng, Junlong Jia, Jie Luo, Xien Liu, Ji Wu, and Lei Huang. 2024. Tinyllava: A framework of small-scale large multimodal models. arXiv preprint arXiv:2402.14289. 





Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. 2023. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592. 





Yichen Zhu, Minjie Zhu, Ning Liu, Zhiyuan Xu, and Yaxin Peng. 2024. Llava-phi: Efficient multi-modal assistant with small language model. In Proceed ings of the 1st International Workshop on Efficient Multimedia Computing under Limited, pages 18–22. 



## A Appendix

## A.1 Implementation Details

We present additional details about our experimental configuration to facilitate the reproduction of our model. The hyperparameters for all stages are summarized in Table 5. We adopted a progressive training strategy in Stage 1 because the loss after pretraining the MLP for TinyGroundingGPT was still relatively high (about 3.0). Further finetuning TinyGroundingGPT with both the MLP and pretrained LLMs reduced the loss to around 1.5 for improving multi-modality ability. Moreover, the object coordinates in training were normalized after padding. 

## A.2 Dataset Details

We provide additional details about the datasets we utilized, as summarized in Table 6. We also include additional examples of the generated datasets in Fig. 8 for Stage 2 and in Fig. 9 for Stage 3. Specifically, during the training process, the object images are cropped and zoomed from the original image and then fed into the vision encoders to obtain object image features. These features are subsequently used to replace the placeholder denoted as <img> in the QA pairs. 

<table><tr><td rowspan="2">Size</td><td colspan="2">Stage 1</td><td rowspan="2">Stage 2</td><td rowspan="2">Stage 3</td></tr><tr><td>Pretrain</td><td>Finetune</td></tr><tr><td>Batch size</td><td>32</td><td>32</td><td>32</td><td>16</td></tr><tr><td>Learning rate</td><td>1e-3</td><td>2e-5</td><td>2e-5</td><td>2e-5</td></tr><tr><td>Epochs</td><td>1</td><td>1</td><td>1</td><td>2</td></tr><tr><td>Learning schedule</td><td></td><td colspan="2">Cosine decay</td><td></td></tr><tr><td>Warm-up ratio</td><td>0.03</td><td>0.03</td><td>0.03</td><td>0.03</td></tr><tr><td>Weight decay</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>BF16</td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>TF32</td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>DeepSpeed stage</td><td></td><td colspan="2">ZeRO2</td><td></td></tr><tr><td>GPUs</td><td></td><td colspan="2">8xA100</td><td></td></tr></table>


Table 5: The hyperparameters for model training.


As described in Section 3.2, we developed a multi-scale fine-grained enhancement data synthesis pipeline, which includes the construction of a multi-scale fine-grained grounding dataset (in Stage 2) and a global grounding dataset (in Stage 3). In Fig. 7, we present the prompt messages used for object recognition and relation extraction to prepare additional data material. Fig. 10 illustrates the detailed processing steps involved in constructing the multi-scale fine-grained grounding dataset. Furthermore, Figs. 11 and 12 outline the processing steps for constructing the global grounding dataset. Additionally, Table 7 shows an example prompt used to evaluate the generated QA pairs. Based on these evaluations, we filtered out low-quality QAs (or descriptions), specifically those with incorrect answers or low-quality scores (<3). 

<table><tr><td>Stage</td><td colspan="2">Dataset</td><td>Samples</td></tr><tr><td>Stage1</td><td colspan="2">LLaVA-Pretrain-595k</td><td>595K</td></tr><tr><td rowspan="3">Stage2</td><td rowspan="3">Alignment data</td><td>Text-coordinate pairs</td><td>4.1M</td></tr><tr><td>Image-coordinate pairs</td><td>210K</td></tr><tr><td>Text-image pairs</td><td>46K</td></tr><tr><td rowspan="3">Stage3</td><td rowspan="3">SFT data</td><td>LLaVA-v1.5, ShareGPT4V</td><td>665K</td></tr><tr><td>Grounding-conv</td><td>28K</td></tr><tr><td>Grounding-description</td><td>29K</td></tr></table>


Table 6: The dataset details used for model training.


## A.3 Object Hallucination Evaluation

We evaluated MLLMs for object hallucination, as shown in Table 8. Higher accuracy and F1-score metrics, along with a lower ’Yes’ metric, indicate better performance. Our TinyGroundingGPT yielded outstanding results across all three sampling subsets. Notably, TinyGroundingGPT-3B outperformed larger models like InstructBLIP-13B in the challenging Adversarial subset, achieving an increase of 14.67% in accuracy and a 8.90% increase in F1 score, despite a decrease of 27.77% in the ’Yes’ metric. Compared to GroundingGPT-7B, our 3B model excelled in the Popular and Adversarial subsets for both accuracy and F1 score. Similarly, TinyGroundingGPT-1.5B achieved higher accuracy and F1 score than some larger models like Shikra while maintaining a lower ’Yes’ score. This superior performance can be attributed to its fine-grained knowledge alignment from both global and local perspectives during training. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/cf9cee5f-3a6e-45d0-b53b-d5da013efb93/65bd7acbcfc8c82fbe31e8a4347e805a68cd5c5b890253a63487ce875eb2586d.jpg)



Table 7: The prompt for GPT-4V to evaluate generated QAs.


## A.4 Grounding Ability for Larger Model

We further apply our fine-grained visual knowledge alignment method to TinyGroundingGPT, using Qwen2.5-7B as the larger-parameter language model, to evaluate its image grounding capability. The results in Table 9 highlight the method’s effectiveness, with increases of 0.68% on RefCOCO+- testA and 0.81% on RefCOCO+-testB. 

## A.5 Ablation Study for Vision Encoders

Image features from ViT-L/14 (Radford et al., 2021) (second-to-last layer) capture more object semantics, while those from DINOv2-L/14 (Oquab et al., 2023) (last layer) capture more local finegrained details, as shown in Fig. 6. The multi-scale vision encoders in our proposed TinyGrounding-GPT align well with the fine-grained alignment of our method. As shown in Table 10, this approach improves performance on benchmarks highly related to fine-grained understanding, such as POPE and VQA<sup>v2</sup>. 

## A.6 Ablation Study for Pretrained LLM

We conduct additional experiments to explore the effect of the pretrained LLM on TinyGrounding-GPT. As shown in Table 11, TinyGroundingGPT with Qwen2.5-1.5B outperforms that with Qwen2- 1.5B in image understanding, highlighting the effectiveness of Qwen2.5 for our proposed TinyGroundingGPT. 

## A.7 Assessment for Image Annotation

As illustrated in Section 5.1, we provided examples of image descriptions generated by TinyGroundingGPT in Fig. 13. Moreover, we selected 50 images from RefCOCO-test and utilized GPT-4V to evaluate image descriptions produced by various methods. We assessed the image descriptions using scores ranging from 1 to 5 across three perspectives: "Quality," which reflects overall quality; "Richness," which measures the diversity of object descriptions; and "Accuracy," which pertains to precision. The prompt used for GPT-4V and the scoring details are presented in Table 12. As the results summarized in Table 13, TinyGroundingGPT achieved better overall quality and richness compared to GroundingGPT-7B and Qwen2-VL-2B. 

<table><tr><td rowspan="2">Model</td><td rowspan="2">Alignment</td><td colspan="3">RefCOCO+</td></tr><tr><td>val</td><td>testA</td><td>testB</td></tr><tr><td rowspan="2">TinyGroundingGPT-7B</td><td>T, C</td><td>83.98</td><td>88.08</td><td>77.90</td></tr><tr><td>T, C, I</td><td>84.56</td><td>88.76</td><td>78.71</td></tr></table>


Table 9: Performance comparison on the referring expression comprehension(REC) task for whether conducting our proposed Multi-scale Fine-grained Local Knowledge Alignment.


<table><tr><td rowspan="2">Models</td><td rowspan="2">LLM Size</td><td colspan="3">Random</td><td colspan="3">Popular</td><td colspan="3">Adversarial</td></tr><tr><td>Accuracy F1-Score</td><td>Yes</td><td>Accuracy F1-Score</td><td>Yes</td><td>Accuracy F1-Score</td><td>Yes</td><td>Accuracy F1-Score</td><td>Yes</td><td></td></tr><tr><td>LLaVA</td><td>7B</td><td>72.16</td><td>78.22</td><td>76.29</td><td>61.37</td><td>71.52</td><td>85.63</td><td>58.67</td><td>70.12</td><td>88.33</td></tr><tr><td>mPLUG-Owl</td><td>7B</td><td>53.97</td><td>68.39</td><td>95.63</td><td>50.90</td><td>66.94</td><td>98.57</td><td>50.67</td><td>66.82</td><td>98.67</td></tr><tr><td>MiniGPT-4</td><td>13B</td><td>79.67</td><td>80.17</td><td>52.53</td><td>69.73</td><td>73.02</td><td>62.20</td><td>65.17</td><td>70.42</td><td>67.77</td></tr><tr><td>InstructBLIP</td><td>13B</td><td>88.57</td><td>89.27</td><td>56.57</td><td>82.77</td><td>84.66</td><td>62.37</td><td>72.10</td><td>77.32</td><td>73.03</td></tr><tr><td>Shikra</td><td>7B</td><td>86.90</td><td>86.19</td><td>43.26</td><td>83.97</td><td>83.16</td><td>45.23</td><td>83.10</td><td>82.49</td><td>46.50</td></tr><tr><td>GroundingGPT</td><td>7B</td><td>89.79</td><td>89.22</td><td>43.13</td><td>88.23</td><td>87.38</td><td>43.23</td><td>86.17</td><td>85.50</td><td>45.43</td></tr><tr><td rowspan="2">TinyGroundingGPT</td><td>3B</td><td>89.93</td><td>89.47</td><td>43.08</td><td>88.56</td><td>87.90</td><td>43.43</td><td>86.77</td><td>86.22</td><td>45.26</td></tr><tr><td>1.5B</td><td>89.59</td><td>88.98</td><td>42.92</td><td>88.67</td><td>87.90</td><td>42.87</td><td>86.74</td><td>86.04</td><td>44.77</td></tr></table>


Table 8: Results on the POPE benchmark for object hallucination evaluation. "Yes" represents the probability of positive answers to the given question.


<table><tr><td>Vision Encoder</td><td>VQA<eq>^{v2}</eq></td><td>GQA</td><td>SQA<eq>^{I}</eq></td><td>POPE</td><td>MME<eq>^{P}</eq></td><td>MMB</td></tr><tr><td>ViT</td><td>76.5</td><td>61.4</td><td>64.8</td><td>85.8</td><td>1417.8</td><td>63.0</td></tr><tr><td>ViT + DINOv2</td><td>77.9</td><td>62.2</td><td>63.1</td><td>87.6</td><td>1392.4</td><td>64.2</td></tr></table>


Table 10: Ablation Study on our TinyGroundingGPT 1.5B for the multi-scale vision encoders.


## A.8 More Visualizations

As illustrated in Section 5.2, we visualized the lastlayer attention maps of both the GroundingGPT-7B baseline and our TinyGroundingGPT-1.5B. The attention map in grounding MLLMs not only enhances interpretability but also illustrates the alignment between the model’s output and the input image. The process for obtaining the heatmap of attention involves several steps: (1) we select the attention scores between image patches and object representations (i.e., texts, coordinates, and images); (2) we sum the attention scores across the dimensions of both the attention heads and object representations; (3) We map the normalized attention scores onto the input image patches. 

Additional visualizations are displayed in Fig. 14. As shown, TinyGroundingGPT reveals more distinct location attributions, indicating that it effectively learned multi-scale fine-grained knowledge and achieved high-level alignments among object texts, coordinates, and images. This provides insights for further explaining MLLMs, particularly in grounding tasks. We also provide a demo for utilizing TinyGroundingGPT in Fig. 15. 

<table><tr><td>Pretrained LLM</td><td>VQA<eq>^{v2}</eq></td><td>GQA</td><td>SQA<eq>^{I}</eq></td><td>POPE</td><td>MME<eq>^{P}</eq></td><td>MMB</td></tr><tr><td>Qwen2-1.5B</td><td>76.3</td><td>61.2</td><td>56.8</td><td>85.7</td><td>1386.9</td><td>58.9</td></tr><tr><td>Qwen2.5-1.5B</td><td>77.9</td><td>62.2</td><td>63.1</td><td>87.6</td><td>1392.4</td><td>64.2</td></tr></table>


Table 11: Ablation Study on our TinyGroundingGPT-1.5B for the pretrained LLM.


<table><tr><td>Model</td><td>Quality</td><td>Richness</td><td>Accuracy</td></tr><tr><td>GPT-4V</td><td>4.24</td><td>4.10</td><td>4.88</td></tr><tr><td>GroundingGPT-7B</td><td>3.68</td><td>3.20</td><td>3.38</td></tr><tr><td>Qwen2-VL-2B</td><td>3.90</td><td>3.64</td><td>4.18</td></tr><tr><td>TinyGroundingGPT-3B</td><td>4.04</td><td>3.90</td><td>3.66</td></tr></table>


Table 13: The assessment for image annotation by GPT-4V includes "Quality" for overall quality, "Richness" for the diversity of object descriptions, and "Accuracy" for precision. Scores are based on the average ratings (1-5) from 50 samples.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/cf9cee5f-3a6e-45d0-b53b-d5da013efb93/106bab9084b385b1d8ba881c6299c9ba500ea8a934e3926eed1d734ff26a081e.jpg)



Table 12: The prompt for GPT-4V to assess descriptions from the perspectives of Quality, Richness, and Accuracy.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/cf9cee5f-3a6e-45d0-b53b-d5da013efb93/de34b5e2040154c32991f2b348b71549fe1e7d7834c9a115eeaa8e1a8b558e4f.jpg)



Figure 6: The visualizations of image feature cosine similarity for the selected patch (highlighted with yellow box). Lighter colors indicate higher feature similarity.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/cf9cee5f-3a6e-45d0-b53b-d5da013efb93/d23454a2a3fc2d153a0e94b4cfcb8f2dee5c89f4ea1d379e42c0af137d810844.jpg)



Figure 7: The prompt message for object recognition and relation extraction.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/cf9cee5f-3a6e-45d0-b53b-d5da013efb93/438663481183627ce0580d38abc3ae0d7ca4c8ade537340b2b985113fbdc88b5.jpg)



Figure 8: Our generated various kinds of data used in Stage2 for achieving high-level alignments among texts, coordinates, and images, where the < img > denotes the corresponding augmented object image.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/cf9cee5f-3a6e-45d0-b53b-d5da013efb93/5d665b3eaf1fe2c606593e2a1717f09e6dc9fbdf4b7d7e735577a69be495b7f5.jpg)



Figure 9: Our generated various kinds of data used in Stage3 for achieving global object alignment, where the < img > denotes the corresponding augmented object image.


<table><tr><td>Prompt Message</td></tr><tr><td>1. You are a QA pair generator assistant. I will provide you with an image: {image}, a list of objects (including the objects present in the image): {Object Recognition}, and a list of triples describing the relationships between the objects : {Relation Extractions}. You may include additional objects and relationships that you are confident about. Please generate 5 optimal QA pairs that accurately reflect the image. Each question must include one object in the list highlighted with “&lt;&gt;”, and the options should represent the relationships. Output the QA pairs and corresponding answers in the format of a JSON list. Follow the output format as a JSON list, such as [QA1, QA2, ...]. Do not include any additional text.</td></tr><tr><td>2. You are a QA pair generator assistant. I will provide you with an image: {image}, a list of objects (including the objects present in the image): {Object Recognition}, and a list of triples describing the relationships between the objects : {Relation Extractions}. You may include additional objects and relationships that you are confident about. Please generate 5 optimal QA pairs that accurately reflect the image. For each question, the options must consist of only two objects in the list, highlighted with “&lt;&gt;”. Output the QA pairs and corresponding answers in the format of a JSON list. Follow the output format as a JSON list, such as [QA1, QA2, ...]. Do not include any additional text.</td></tr><tr><td>User Inputs</td></tr><tr><td>Object Recognition: A list of objects in the image.Relation Extractions: A triple list of relationship among objects.Object Grounding: A dictionary for bounding box coordinates of each subject.</td></tr><tr><td>Assistant</td></tr><tr><td>Example1Q1: Can you identityis worn by whom? A. the left woman B. the right manA1: The answer is B. the right man.Example2Q2: What is the right man wearing blue jacket reading? A.B.A2: The answer is A. newspaper.</td></tr><tr><td>Postprocess</td></tr><tr><td>1. Generate text and coordinate alignment QA pairs: (a) Given the object coordinates, formulate questions such as, “Describe the object located within the region [x1, y1, x2, y2].” Provide the corresponding object description as the answer. (b) Given the object description, create questions like, “In this image, where islocated?” Provide the corresponding object coordinates as the answer.2. Generate image and coordinate alignment QA pairs by utilizing object coordinates, such as: “Q: What is the location of? A:” and “Q: Which is the item in region? a. . A: The answer is a.” Extract the corresponding object images based on the object coordinates and replacewith the augmented object images.3. Generate image and text alignment QA pairs: (a) Identify all objects highlighted with “&lt;&gt;” in the questions or options. (b) Use the dictionary that maps each object to its bounding box coordinates in the image to extract the corresponding object images. (c) Replace the highlighted object in the text with the augmented object image.4. Ensure that the object images in each QA pair share a small Intersection over Union.</td></tr></table>


Figure 10: The prompt message and user’s input example used for generating our Fine-grained Grounding Dataset in Stage2.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/cf9cee5f-3a6e-45d0-b53b-d5da013efb93/718bbb27a012fdf5f25fda741d7d9cf28b7a8493376c33589ca258cdcd1d5a17.jpg)



Figure 11: The prompt message and user’s input example used for generating our Multi-round Grounding Conversa tion Data in Stage3.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/cf9cee5f-3a6e-45d0-b53b-d5da013efb93/a575c08bddc546067461b7267e7392a8cf23047baafe327b841a839cb9c9c621.jpg)



Figure 12: The prompt message and user’s input example used for generating our Grounding Description Data in Stage3.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/cf9cee5f-3a6e-45d0-b53b-d5da013efb93/307c61fa846902a40bc6c9eae6279c2f94f9cde2b83d2e9ad839e1443c77a688.jpg)



Figure 13: A comparison of generated image descriptions between TinyGroundingGPT trained with our method and without it.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/cf9cee5f-3a6e-45d0-b53b-d5da013efb93/68988c9118731b1151fa197a396934db8b27cd5d6ef5b2d4f0f75d5ca6a0d7ab.jpg)



Figure 14: The visualization of the attention map for image patches with different object representation outputs (texts, coordinates, and images, which are underlined), where the red bounding box denotes the target region


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/cf9cee5f-3a6e-45d0-b53b-d5da013efb93/2a05fe8c6c21b6fedb91fadcc3477d99d5d493b14bf8d85830f11cce819be02c.jpg)



Figure 15: A demo for the use of our TinyGroundingGPT.
