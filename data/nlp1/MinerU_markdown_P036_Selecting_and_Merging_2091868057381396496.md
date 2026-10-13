# Selecting and Merging: Towards Adaptable and Scalable Named Entity Recognition with Large Language Models

Zhuojun Ding<sup>1</sup>, Wei Wei*<sup>1</sup> and Chenghao Fan<sup>1</sup> <sup>1</sup> School of Computer Science & Technology, Huazhong University of Science and Technology {dingzj, weiw}@hust.edu.cn, facicofan@gmail.com 

## Abstract

Supervised fine-tuning (SFT) is widely used to align large language models (LLMs) with in formation extraction (IE) tasks, such as named entity recognition (NER). However, annotating such fine-grained labels and training domainspecific models is costly. Existing works typically train a unified model across multiple domains, but such approaches lack adaptation and scalability since not all training data benefits target domains and scaling trained models remains challenging. We propose the SaM framework, which dynamically Selects and Merges expert models at inference time. Specifically, for a target domain, we select domain-specific experts pre-trained on existing domains based on (i) domain similarity to the target domain and (ii) performance on sampled instances, respectively. The experts are then merged to create task-specific models optimized for the target domain. By dynamically merging experts beneficial to target domains, we improve generalization across various domains without extra training. Additionally, experts can be added or removed conveniently, leading to great scalability. Extensive experiments on multiple benchmarks demonstrate our framework’s effective ness, which outperforms the unified model by an average of 10%. We further provide insights into potential improvements, practical experi ence, and extensions of our framework. 

## 1 Introduction

Large language models (LLMs) demonstrate remarkable performance across a wide range of tasks (Achiam et al., 2023; Yang et al., 2024; Guo et al., 2025a), but still struggle with information extraction (IE) tasks (Xu et al., 2024; Ding et al., 2024b; Fan et al., 2024b), such as Named Entity Recognition (NER). The inherent gap between task formulations and LLM training objectives is a critical factor underlying this limitation. To mitigate this, supervised fine-tuning (SFT) has become a widely used strategy, demonstrating significant improvements (Wang et al., 2023a; Zhou et al., 2024b; Fan et al., 2025). 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/906a0161-4f11-48ac-9b69-307c345782ff/ff85da82d1dd871deeeeefe9469df827cb62097294ec130b4faa2467c34ac88d.jpg)



Figure 1: (a) Existing methods train a general unified model across multiple domains, while we dynamically select and merge expert models at inference time. (b) A trained system struggles to accommodate changes in training data, while we flexibly add or remove expert models, ensuring great scalability.


However, annotating data and training domainspecific models each time is costly, particularly for fine-grained IE tasks. Most existing approaches collect large-scale training data from multiple domains to train a unified model (Wang et al., 2023a; Sainz et al., 2024; Yang et al., 2025). Although such models exhibit cross-domain generalization capabilities, they frequently exhibit suboptimal performance in both in-domain and out-of-domain test scenarios. This limitation arises primarily because (1) not all training samples universally enhance performance on a given target domain (Liu et al., 2024; Zhou et al., 2024a), and (2) inherent conflicts may emerge across heterogeneous domains during joint training, leading to compromised optimization efficacy (Sainz et al., 2024; Yang et al., 2025; Fan et al., 2024a). Additionally, even when data from the target domain is available, effectively integrating it into a trained model without compromising performance remains challenging. 

To address these issues, we adopt a model merging strategy (Ilharco et al., 2023) to dynamically select domain-specific models for different target domains and fuse their parameters to obtain taskspecific models. Specifically, we first train multiple expert models in different domains with available data. Then, we design the SaM framework to derive task-specific models by Selecting and Merging experts from two perspectives: (i) Domain similarity. We assess the domain similarity between the target domain and each expert model, and select the most relevant experts for parameter fusion to create a task-specific model tailored to the target domain. (ii) Sampling evaluation. We randomly sample k data instances from the target domain and integrate the predictions of all experts as pseudo-labels (no ground truth labels required). Based on these labels, we assess the performance of each expert on the sampled data and select the best-performing experts to merge into another task-specific model. The two perspectives synergistically complement each other: domain similarity provides a high-level, coarse-grained assessment that establishes theoretical priors, while sampling evaluation delivers a fine-grained, empirical quantification of expert performance to yield actionable practical guidance. By integrating the outputs of both task-specific models, we achieve more comprehensive results. 

Compared to previous methods (Figure 1), we allow for task-specific customization across diverse target domains, providing improved generalization ability without extra training. It also provides great scalability, as experts can be easily added or removed based on practical needs. Notably, our approach is orthogonal to previous studies. By leveraging their practical insights, we can train more effective expert models, thereby enhancing the performance of our framework. In terms of resource requirements, our approach does not incur additional training costs. By leveraging parameter-efficient fine-tuning methods (Hu et al., 2022), we only introduce minimal storage overhead. Additionally, by employing either strategy individually or further integrating the target models derived from both strategies, we can achieve comparable performance without incurring additional inference costs. 

In summary, our contributions are as follows: (1) We introduce a model-merging paradigm for LLM-based Named Entity Recognition, enhancing adaptability and scalability. (2) We propose a model selection strategy based on domain similarity and sampling evaluation, which effectively selects expert models beneficial to the target domain for merging. (3) Experimental results demonstrate the effectiveness of our framework, which outperforms the unified model by an average of 10% and by up to 20% in certain domains. Further experiments analyze potential improvements, practical experience, and framework generalizability, providing deeper practical insights. 

## 2 Related Works

LLMs for Information Extraction Current LLMs-based IE mainly fall into two paradigms. 

One paradigm uses larger models. Training them needs significant computational resources, and finetuning them specifically for IE tasks may be not cost-effective. However, these models excel in instruction-following and reasoning. Therefore, such methods focus on optimizing task instructions, reasoning strategies, or in-context learning (ICL) demonstrations. Li et al. (2023) show that codestyle prompts enhance IE tasks. Pang et al. (2023) and Tong et al. (2025) prompt LLMs with more comprehensive information to improve task understanding. Xie et al. (2023) and Wan et al. (2023) introduce reasoning techniques such as Chain-of-Thought (CoT) to guide the model in step-by-step task completion. Xie et al. (2024) employ selfconsistency to generate reliable ICL examples. 

Another paradigm uses smaller models. While these models have weaker instruction-following capabilities, they require much fewer training resources. Such methods enhance LLMs through supervised fine-tuning (Wang et al., 2023a). Many studies design optimization strategies on the data side. Yang et al. (2025) resolves conflicts and redundancy in training data. Zhou et al. (2024b) distills more diverse data from ChatGPT. Li et al. (2024) formats training data in code style. Sainz et al. (2024) enriches instructions with detailed task descriptions. Ding et al. (2024a) emphasizes negative samples. In addition to instruction tuning, Qi et al. (2024) further employs alignment training (Rafailov et al., 2024), and Guo et al. (2025b) incorporates contrastive learning objectives. Instead of training one universal model, we train several domain experts and design a merging method to improve adaptability and scalability. 

Additionally, the backbone model is also critical. Recently, code-based LLMs have gained popularity, as they may better suit IE tasks than natural language-based LLMs (Li et al., 2023). 

Model Merging Model merging integrates multiple task-specific models at the parameter level to create a unified model, which could handle multiple tasks simultaneously and exhibit better out-ofdomain generalization. Unlike multi-task learning, model merging reuses existing models, reducing computational and data demands since only model parameters are needed. Beyond simple parameter averaging, Matena and Raffel (2022) assigns different importance to model parameters. Ilharco et al. (2023) applies arithmetic operations for finer control over model behavior. Jin et al. (2023) enforces output consistency between the merged model and its constituent models. Yu et al. (2024) and Yadav et al. (2023) mitigate inter-model interference by addressing parameter redundancy or sign inconsistency, and weight sparsity, respectively. Lu et al. (2024) decomposes model parameters into shared and task-specific components. In this paper, we introduce model merging to improve adaptability and scalability across different target domains. 

## 3 Methodology

In this section, we first provide an overview of the training process of domain expert models. Then we explain our SaM framework for selecting and merging experts, as shown in Figure 2. 

## 3.1 Training Domain Experts

Data Collection. We first collect more than 20 commonly used NER datasets and classify them into six domains based on their sources: News, Social media, Biomedical, STEM (Science, Technology, Engineering, and Mathematics), Legal, and Traffic. We remove 90% of the NA data that contains no entities. Through sampling or redundancy, we limit the total samples per domain to between 10,000 and 50,000. The number of sampled instances from each dataset was proportional to the number of entity types it contained. Detailed information is provided in Appendix A. 

Training Data Construction. Refering to practice of prior studies (Wang et al., 2023a; Qi et al., 2024), we format the raw data into task instructions, inputs, and outputs for training. 

The task instructions consist of: (1) Data source description: A brief overview of the dataset source. (2) Entity type description: Concise definition of entity types. (3) In-context learning demonstrations: 1 5 randomly selected input-output pairs from the training data. (4) Label drop: Excluding the requirement for recognizing certain entity types. (5) Label masking: Replacing entity labels with abstract placeholders such as “Type1”. Empirically, we apply (1), (2), and (3) to 70% of the data; (4) to 30%; and (5) to 5%. These modifications are applied independently, except (5), which should co-occur with (2). For output parts, we adopt three formats: JSON (e.g., “{entity span: entity type}”), enumeration (e.g., “Type: span1, span2, ...”), and natural language descriptions. 

These strategies help enhance model robustness to some extent. However, we claim this is not an optimal configuration, as our focus is not on training a best-performing model. 

Model Training. Following prior work, we train the model using instruction tuning. Given a dataset ${ \cal D } _ { \cal A } = \{ ( I , X , Y ) \}$ from domain A, where I is the task instruction, X is the input sequence, and $Y = \{ y _ { i } \} _ { i = 1 } ^ { L }$ is the output sequence (i.e., entity predictions), the training loss of the domain-specific expert model $\mathcal { M } _ { A }$ is defined as: 

$$
\mathcal {L} _ {\theta_ {A}} = - \sum_ {D _ {A}} \sum_ {t = 1} ^ {L} \log P _ {\theta_ {A}} (y _ {t} \mid I, X, y _ {<   t})\tag{1}
$$

where $\theta _ { A }$ denotes the parameters of $\mathcal { M } _ { A }$ 

## 3.2 Selecting and Merging Experts

When handling a specific target domain, we select a subset of expert models and fuse their parameters to obtain task-specific models. As shown in Figure 2, this process is conducted from two perspectives. 

Selecting with Domain Similarity. Given a domain with raw data $D _ { A } = \{ x _ { i } \}$ , we obtain the corresponding data embeddings $H _ { A } = \left\{ h _ { i } \right\}$ through a text encoder. The domain embedding is then defined as the centroid of these data embeddings: 

$$
h _ {A} = \frac {1}{| H _ {A} |} \sum_ {h _ {i} \in H _ {A}} h _ {i}\tag{2}
$$

We compute domain embeddings $\{ h _ { e _ { i } } \}$ for domains of all expert models and $h _ { t }$ for the target domain. Then we compute the similarity between the expert domains and the target domain with cosine distances. Finally, the top-m similar expert models are selected for model merging. 

The domain embedding inherently captures the data distribution in the embedding space. Thus, in theory, the selected expert models exhibit a certain degree of similarity and are expected to perform well in the target domain due to the resemblance in these data distributions. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/906a0161-4f11-48ac-9b69-307c345782ff/b42581dda253113b7a666c62bdc0e5534c5149f3c394185cccb35746b61ef238.jpg)



Figure 2: Framework overview. Given a target domain, we select expert models from two perspectives: (a) Domain Similarity, which selects experts from domains most similar to the target domain. We compute the centroid of all data embeddings as the domain embedding and measure similarities by cosine distance. (b) Sampling Evaluation, which selects experts with better performances on sampled instances from the target domain. To reduce reliance on ground-truth labels, we ensemble predictions from all experts as (pseudo) labels. We merge models within each expert subset to obtain two task-specific models. The final result integrates the outputs of the two task models.


Selecting with Sampling Evaluation. The selection of experts based on domain similarity is theoretically sound. However, in practice, we observe that the model with the highest domain similarity to the target domain does not always yield the best performance. Thus, we propose another selection strategy driven by model performance. 

Specifically, we first randomly sample k data instances from the target domain. Then each expert model generates predictions for them. To reduce dependence on ground-truth labels, we aggregate predictions via majority voting to construct pseudolabels, which are subsequently used to assess expert performance. The top m experts with the highest performance are selected for model merging. 

Unlike domain similarity-based selection, this approach prioritizes practical effectiveness. As a result, the selected experts generally perform better individually in the target domain. However, we aim to obtain a superior task model through model merging, where individual performance is not the sole determining factor. 

Merging Experts. Given a base model $\mathcal { M } _ { b a s e }$ and the supervised fine-tuned model $\mathcal { M } _ { s f t }$ , we denote their parameters as $\theta _ { b a s e }$ and $\theta _ { s f t }$ , respectively. The delta parameter $\delta _ { s f t } = \theta _ { s f t } - \theta _ { b a s e }$ serves as a parametric representation of the model’s learned capabilities and is also referred to as the task vector. Given multiple task-specific models $\{ \mathcal { M } _ { s f t _ { i } } \}$ we can merge them into a unified model <sub>merge</sub> with diverse capabilities (Matena and Raffel, 2022; Ilharco et al., 2023): 

$$
\theta_ {m e r g e} = \theta_ {b a s e} + \mathrm{Merge} (\delta_ {s f t _ {1}}, \delta_ {s f t _ {2}}, \dots)\tag{3}
$$

where Merge( ) denotes the model merging technique, such as simple averaging and task arithmetic (Ilharco et al., 2023). We employ the Ties-Merging (Yadav et al., 2023) method, which addresses parameter redundancy and sign inconsistency to mitigate inter-model interference when merging multiple models. We selected two sets of expert models based on Domain Similarity (DS) and Sampling Evaluation (SE), respectively. These experts are subsequently merged to obtain two taskspecific models, $\mathcal { M } _ { D S }$ and $\mathcal { M } _ { S E }$ 

## 3.3 Inference

For a target domain, we obtain two task-specific models, $\mathcal { M } _ { D S }$ and ${ \mathcal { M } } _ { S E } ,$ , following the methodology described in Section 3.2. Each model independently generates predictions, producing two output sets, $Y _ { D S }$ and $Y _ { S E }$ . Taking the intersection of two sets of predictions typically enhances reliability. However, our two task-specific models are already tailored for the target domain and capture different and complementary perspectives. Therefore, we adopt their union as the final result. 

<table><tr><td rowspan="2" colspan="2"></td><td colspan="5">CrossNER</td><td colspan="2">MIT</td><td rowspan="2">Average</td></tr><tr><td>AI</td><td>Literature</td><td>Music</td><td>Politics</td><td>Science</td><td>Movie</td><td>Restaurant</td></tr><tr><td rowspan="6">Recent Studies</td><td>InstructUIE</td><td>48.40</td><td>48.80</td><td>54.40</td><td>49.90</td><td>49.40</td><td>63.00</td><td>20.99</td><td>47.84</td></tr><tr><td>UniNER</td><td>62.90</td><td>64.90</td><td>70.60</td><td>66.90</td><td>70.80</td><td>61.20</td><td>35.20</td><td>61.79</td></tr><tr><td>GoLLIE</td><td>59.10</td><td>62.70</td><td>67.80</td><td>57.20</td><td>55.50</td><td>63.00</td><td>43.40</td><td>58.39</td></tr><tr><td>KnowCoder</td><td>60.30</td><td>61.10</td><td>70.00</td><td>72.20</td><td>59.10</td><td>50.00</td><td>48.20</td><td>60.13</td></tr><tr><td>GLiNER</td><td>57.20</td><td>64.40</td><td>69.60</td><td>72.60</td><td>62.60</td><td>57.20</td><td>42.90</td><td>60.90</td></tr><tr><td>B2NER</td><td>59.00</td><td>63.70</td><td>68.60</td><td>67.80</td><td>72.00</td><td>67.60</td><td>53.30</td><td>64.57</td></tr><tr><td rowspan="2">Fully-trained</td><td>Llama</td><td>54.11</td><td>63.44</td><td>72.53</td><td>62.92</td><td>59.09</td><td>64.81</td><td>49.68</td><td>60.94</td></tr><tr><td>Qwen</td><td>51.38</td><td>53.46</td><td>61.12</td><td>54.99</td><td>59.39</td><td>66.66</td><td>52.69</td><td>57.10</td></tr><tr><td rowspan="2">SaM (Ours)</td><td>Llama</td><td><eq>\underline{60.98}_{12.7\%}</eq></td><td><eq>\underline{66.93}_{5.50\%}</eq></td><td><eq>\underline{73.53}_{1.4\%}</eq></td><td><eq>\underline{74.47}_{18.4\%}</eq></td><td><eq>\underline{62.60}_{5.9\%}</eq></td><td><eq>\underline{72.17}_{11.4\%}</eq></td><td><eq>\underline{52.99}_{6.7\%}</eq></td><td><eq>\underline{66.24}_{8.70\%}</eq></td></tr><tr><td>Qwen</td><td><eq>60.01_{15.8\%}</eq></td><td><eq>61.99_{16.0\%}</eq></td><td><eq>65.93_{7.9\%}</eq></td><td><eq>67.05_{21.9\%}</eq></td><td><eq>62.41_{5.1\%}</eq></td><td><eq>\underline{71.65}_{7.50\%}</eq></td><td><eq>\underline{52.90}_{0.4\%}</eq></td><td><eq>63.13_{10.6\%}</eq></td></tr></table>


Table 1: Experimental results. We compare our method with recent studies and our fully-trained model (e.g., a unified model trained on all data we used). The best results are highlighted in bold, while suboptimal results are underlined. The right subscript denotes the percentage improvement compared to the fully trained model.


## 4 Experiments

## 4.1 Setup

Benchmarks, Baselines, and Metrics We evaluate our framework on two widely used benchmarks CrossNER (Liu et al., 2021) and MIT (Ushio and Camacho-Collados, 2021), which contain datasets from seven domains (AI, Literature, Music, Politics, Science, Movie, and Restaurant) in total.Our experiments are under zero-shot settings (i.e., no labeled target domain data), follow prior work. The source data for training and the target data for evaluation have different distributions. 

We introduce two types of baselines for comparison. The first is the Fully-trained model, a single unified model trained on data from all domains using the same training configuration as us. This serves as the primary baseline to assess the effectiveness of our framework. The second includes recent studies that also train unified models but incorporate other advanced training optimizations, including InstructUIE (Wang et al., 2023a), UniNER (Zhou et al., 2024b), GoL LIE (Sainz et al., 2024), KnowCoder (Li et al., 2024), GLiNER (Zaratiana et al., 2024), and B2NER (Yang et al., 2025). Most of these models use LLMs as the foundation, except for GLiNER, which contains only 300 million parameters. 

Following prior studies (Wang et al., 2023a), we use the entity-level micro-F1 score as the evaluation metric, where both the entity boundary and entity type should be correctly predicted. 

Implementations We employ models from the Qwen and Llama series as base models for our ex periments. Specifically, we adopt the base version of Qwen2.5-7B and Llama3.1-8B as foundations and train expert models using LoRA (Hu et al., 2022). We employ the all-MiniLM- $L 6 \mathrm { - } \mathbf { v } 2 ^ { 2 }$ text encoder to produce text embeddings. We set m (the number of selected models for merging) to 3. We set k (the number of sampled data instances) to 10. More details are reported in Appendix A. 

## 4.2 Main Results

As shown in Table 1, our approach significantly outperforms the fully trained model across all target domains, achieving an average improvement of approximately 10%, with gains of up to 20% in specific domains. This demonstrates the effectiveness and superior domain adaptability of our approach. To ensure practical comparability, we also compare our results with recent studies. These methods employ various training optimization strategies. For example, B2NER mitigates redundant and conflicting information in the training data. These techniques are orthogonal to ours. Notably, comparing these methods with our fully-trained model suggests that refining our training configuration could enhance our expert models and further improve the performance of our approach. Our approach may incur slight computational and storage overhead, which is acceptable, as discussed in Appendix C. 

## 4.3 Ablation Studies

We conduct ablation studies to validate the effectiveness of our design, as shown in Table 2: (1) w/o Merging, which directly uses the best expert model. (2) w/o Domain Similarity, which selects experts solely based on Sampling Evaluation. (3) w/o Sampling Evaluation, which selects experts based on Domain Similarity. (4) w/o Selection, which merges all experts without selection. Results demonstrate that the merged model consistently outperforms the best individual expert, even when the selected models are not necessarily optimal. Overall, both expert selection strategies are effective and complementary, yielding the best results when combined. However, in some cases, using a single selection strategy or none at all yields better results, likely due to expert redundancy or insufficiency, as we fix the number of selected experts to $k = 3$ in our experiments. Further analysis of k are presented in Section 4.5 

<table><tr><td colspan="2"></td><td>AI</td><td>Literature</td><td>Music</td><td>Politics</td><td>Science</td><td>Movie</td><td>Restaurant</td><td>Average</td></tr><tr><td rowspan="5">Llama</td><td>w/o Merging</td><td>53.72</td><td>59.95</td><td>71.61</td><td>71.88</td><td>59.09</td><td>66.43</td><td>50.62</td><td>61.90</td></tr><tr><td>w/o Domain Similarity</td><td>58.43</td><td>59.87</td><td>72.89</td><td><eq>\underline{74.20}</eq></td><td>59.92</td><td>68.91</td><td>51.69</td><td>63.70</td></tr><tr><td>w/o Sampling Evaluation</td><td>56.42</td><td><eq>\underline{63.92}</eq></td><td>72.25</td><td>72.99</td><td>61.71</td><td><eq>\underline{71.08}</eq></td><td><eq>\underline{52.48}</eq></td><td><eq>\underline{64.41}</eq></td></tr><tr><td>w/o Selection</td><td><eq>\underline{60.14}</eq></td><td>63.74</td><td><eq>\underline{74.83}</eq></td><td>73.00</td><td><eq>\underline{63.68}</eq></td><td>67.30</td><td>47.64</td><td>64.33</td></tr><tr><td>SaM (Ours)</td><td>60.98</td><td>66.93</td><td><eq>\underline{73.53}</eq></td><td>74.47</td><td><eq>\underline{62.60}</eq></td><td>72.17</td><td>52.99</td><td>66.24</td></tr><tr><td rowspan="5">Qwen</td><td>w/o Merging</td><td>57.90</td><td>59.01</td><td>65.33</td><td>65.73</td><td>60.21</td><td>65.45</td><td>51.11</td><td>60.68</td></tr><tr><td>w/o Domain Similarity</td><td><eq>\underline{58.57}</eq></td><td><eq>\underline{61.61}</eq></td><td><eq>\underline{65.63}</eq></td><td><eq>\underline{68.51}</eq></td><td>59.67</td><td>70.66</td><td><eq>\underline{52.69}</eq></td><td>62.48</td></tr><tr><td>w/o Sampling Evaluation</td><td>58.23</td><td>60.39</td><td>64.11</td><td>64.57</td><td>61.00</td><td><eq>\underline{71.04}</eq></td><td>50.11</td><td>61.35</td></tr><tr><td>w/o Selection</td><td>57.47</td><td>60.93</td><td>64.66</td><td>60.28</td><td><eq>\underline{61.08}</eq></td><td>69.43</td><td>47.78</td><td>60.23</td></tr><tr><td>SaM (Ours)</td><td>60.01</td><td>61.99</td><td>65.93</td><td><eq>\underline{67.05}</eq></td><td>62.41</td><td>71.65</td><td>52.90</td><td>63.13</td></tr></table>


Table 2: Ablation studies. Removing certain components typically results in performance degradation, confirming their significance. The best results are in bold, and the suboptimal ones are underlined.


<table><tr><td></td><td>Model</td><td>Mode2</td><td>Mode3</td><td>SaM (Ours)</td></tr><tr><td>AI</td><td>60.36</td><td>58.19</td><td>61.31</td><td>60.01</td></tr><tr><td>Literature</td><td>56.86</td><td>62.68</td><td>58.90</td><td>61.99</td></tr><tr><td>Music</td><td>63.78</td><td>66.50</td><td>66.85</td><td>65.93</td></tr><tr><td>Politics</td><td>66.05</td><td>68.27</td><td>61.30</td><td>67.05</td></tr><tr><td>Science</td><td>58.37</td><td>61.73</td><td>63.12</td><td>62.41</td></tr><tr><td>Movie</td><td>70.66</td><td>70.66</td><td>70.50</td><td>71.65</td></tr><tr><td>Restaurant</td><td>52.82</td><td>52.85</td><td>52.23</td><td>52.90</td></tr><tr><td>Average</td><td>61.27</td><td>62.98</td><td>62.03</td><td>63.13</td></tr></table>


Table 3: Merging into a single task model (based on Qwen). “Modei” denotes the method used to further extract a final set from two expert sets for model merging.


## 4.4 Merging into a Single Task Model

Since our approach employs two task-specific models, the inference cost is doubled. To mitigate this, we derive a single set from the two selected expert sets, reducing the number of task models to one. We propose and evaluate three strategies, with results in Table 3: (1) Mode1 leverages the intersection of the two selected expert sets. (2) Mode2 normalizes the evaluation metrics across both selection strategies (e.g., the domain similarity scores and F1-scores on sampled data points) to a common scale and selects the top three experts. (3) Mode3 takes the union of the two expert sets while limiting the total number of selected experts to three. Experimental results show that Mode2 and Mode3 achieve comparable performance to us, making them effective alternatives without increasing inference costs. We refer to these as the economic versions of our framework $\left( \mathbf { S a M } _ { e c o } \right)$ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/906a0161-4f11-48ac-9b69-307c345782ff/f1c04005bad28ba863c4eaa913e59c8f896b116894290a3fb5dee39c373da2ec.jpg)



Figure 3: Performance changes with the number of experts. The horizontal axis is the number of experts for merging, and the vertical denotes the F1 scores.


## 4.5 Numbers of Experts for Merging

This section analyzes the impact of the number of merged models, k, with results shown in Figure 3. Here, k = 1 denotes the performance of the best individual expert, while $k = 6$ corresponds to merging all expert models. The optimal k varies across target domains, typically ranging from 2 to 4. We set k = 3 as it yields the best average performance. Notably, this corresponds to the merging algorithm as well. We adopt the Ties-Merging technique, where selecting 2 4 models for merging is a commonly used configuration. 

<table><tr><td>Source Domains</td><td>AI</td><td>Literature</td><td>Music</td><td>Politics</td><td>Science</td><td>Movie</td><td>Restaurant</td><td>Average</td></tr><tr><td>Three</td><td>56.42</td><td>63.01</td><td>71.69</td><td>72.24</td><td>61.17</td><td>69.91</td><td>50.62</td><td>63.58</td></tr><tr><td>Six</td><td>60.98</td><td>66.93</td><td>73.53</td><td>74.47</td><td>62.60</td><td>72.17</td><td>52.99</td><td>66.24</td></tr><tr><td>Nine</td><td>59.90</td><td>66.93</td><td>73.53</td><td>73.61</td><td>62.78</td><td>72.17</td><td>53.84</td><td>66.11</td></tr></table>


Table 4: Performance under different number of source domains. Using six source domains yields better results than three, but increasing to nine provides no further gains.


<table><tr><td></td><td>AI</td><td>Literature</td><td>Music</td><td>Politics</td><td>Science</td><td>Movie</td><td>Restaurant</td><td>Average</td></tr><tr><td>Fully-trained</td><td>70.40</td><td>71.81</td><td>75.44</td><td>79.41</td><td>50.71</td><td>58.78</td><td>69.33</td><td>67.98</td></tr><tr><td>Sam(Ours)</td><td>83.45</td><td>71.81</td><td>77.11</td><td>81.56</td><td>77.14</td><td>78.01</td><td>73.33</td><td>77.49</td></tr></table>


Table 5: Performance under extreme scenarios with only one test sample (average of five trials). Our method still functions properly and adapts better than the fully trained single-model approach.


## 4.6 Number of Source Domains

We set six source domains in our experiments. This section presents a preliminary analysis of how the number of source domains affects performance, as shown in Table 4. While setting more source domains increases the diversity of candidate models during selection, it does not always lead to better results. The performance relies on the similarity between source and target domains, including conceptual relevance, data distribution, and overlap in entity types. Therefore, it’s important to balance the number of source domains, their similarity to target domains, and the overall complexity. 

## 4.7 Limited Target Resources Scenarios

The model selection process leverages targetdomain raw texts for domain similarity calculation and sampling evaluation, typically a few hundred samples for the former and 10 for the latter. To test our method under more constrained settings, we simulate an extreme case with only one targetdomain instance. As shown in Table 5, our method remains effective. However, very small sample pools can undermine the stability of expert selection, especially for the strategy using domain similarity. In such cases, we recommend using data augmentation to expand the sample pool. 

## 4.8 Weighting Experts for Merging

We employ two metrics, domain similarity and sampling evaluation, to select expert models. These metrics reflect the importance of experts. Our framework does not consider the importance of experts but instead assigns equal weight to all expert models. To investigate the impact, we propose two simple weighting strategies, as shown in Table 6: (1) Mode1 empirically assigns weights (1.5, 1.0, 0.5) to the top three selected experts. (2) Mode2 uses the middle-ranked metric value as a normalization factor to scale the three experts’ metrics for weighting. Experimental results suggest that weighting has the potential to improve performance (Mode1), aligning with intuitive expectations. However, excessive reliance on heuristics may not always be justified. For example, while Mode2 theoretically provides a more precise weighting based on expert importance, it underperforms compared to Mode1 and Ours. 

<table><tr><td></td><td>Mode1</td><td>Mode2</td><td>SaM (Ours)</td></tr><tr><td>AI</td><td>60.07</td><td>59.33</td><td>60.01</td></tr><tr><td>Literature</td><td>62.00</td><td>62.03</td><td>61.99</td></tr><tr><td>Music</td><td>66.35</td><td>65.17</td><td>65.93</td></tr><tr><td>Politics</td><td>65.54</td><td>66.45</td><td>67.05</td></tr><tr><td>Science</td><td>62.27</td><td>62.15</td><td>62.41</td></tr><tr><td>Movie</td><td>71.20</td><td>71.48</td><td>71.65</td></tr><tr><td>Restaurant</td><td>53.84</td><td>52.84</td><td>52.90</td></tr><tr><td>Average</td><td>63.04</td><td>62.78</td><td>63.13</td></tr></table>


Table 6: Weighting experts for merging. “Modei” denotes the method used to weight model parameters.


## 4.9 Finer Adaptation for Target Domains

Beyond the weighted merging strategy in Section 4.8, another potential approach for improvements is adopting finer adaptation. Specifically, we apply clustering to divide the target domain into multiple splits, selecting and merging expert models separately for each. However, as shown in Figure 4, this finer adaptation does not enhance performance. Instead, it results in an overall decline. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/906a0161-4f11-48ac-9b69-307c345782ff/2092afe0870be4c69e8fe0d087f8f44c6e0fdc746e36f27f3502935d6996cded.jpg)



Figure 4: Performance changes with the number of data splits. The horizontal axis is the number of splits, and the vertical denotes the entity-level F1 scores.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/906a0161-4f11-48ac-9b69-307c345782ff/83b95a0067edb8730208c184a346312896afa736437f4d32deb693cc6bb56581.jpg)



Figure 5: Performance changes with the number of data splits. We aggregate data from all domains into a unified dataset and subsequently conduct the split analysis.


To investigate this further, we aggregate data from all seven domains into a single dataset and conduct experiments. As shown in Figure 5, clustering improves performance across various strategies, including Sampling Evaluation (SE), Domain Similarity (DS), and their combination. Notably, despite the dataset spanning seven domains, the best performance is achieved when clustering divides it into only two or three groups. This is intuitively reasonable, as these seven domains originate from two broader datasets (CrossNER and MIT). The above analysis indicates that while finer adaptation to the target domain may bring improvements, excessive refinement without constraints may be counterproductive. For example, treating each data point as a distinct domain might seem optimal in theory but leads to poor performance in practice. 

## 4.10 Analysis of Merging Technique

In addition to the Ties-Merging algorithm (ties) we employed, this section analyzes alternative merging strategies, including linear, dare (Yu et al., 2024), and their combinations. As shown in Table 7, dare and ties are effective. We do not present the performance of linear strategy, as this strategy leads to significant degradation. Specifically, models merged via the linear approach still produce some meaningful content but almost lose the ability to generate structured outputs, which is crucial for NER and other IE tasks. Consequently, their performance is bad, though minor improvements can be achieved through extensive post-processing on model outputs. However, combining dare with linear yields improved results. Both dare and ties address the issue of parameter redundancy for merging. These findings suggest that handling parameter redundancy is crucial for NER and similar structured output tasks. Additionally, the relatively weaker performance of the dare-ties combination may stem from excessive redundancy reduction, which could compromise useful capabilities of models. 

<table><tr><td></td><td>dare-linear</td><td>ties</td><td>dare-ties</td></tr><tr><td>AI</td><td>54.67</td><td>60.01</td><td>52.57</td></tr><tr><td>Literature</td><td>56.22</td><td>61.99</td><td>55.81</td></tr><tr><td>Music</td><td>66.31</td><td>65.93</td><td>66.22</td></tr><tr><td>Politics</td><td>69.11</td><td>67.05</td><td>66.42</td></tr><tr><td>Science</td><td>62.01</td><td>62.41</td><td>61.07</td></tr><tr><td>Movie</td><td>68.31</td><td>71.65</td><td>65.12</td></tr><tr><td>Restaurant</td><td>52.12</td><td>52.90</td><td>51.40</td></tr><tr><td>Average</td><td>61.25</td><td>63.13</td><td>59.80</td></tr></table>


Table 7: Comparison of different merging techniques. We compare linear, dare, ties, and some combinations.


<table><tr><td rowspan="2"></td><td colspan="3">Llama</td><td colspan="3">Qwen</td></tr><tr><td>T-SC</td><td>E-SC</td><td>Ours</td><td>T-SC</td><td>E-SC</td><td>Ours</td></tr><tr><td>AI</td><td>54.20</td><td>60.31</td><td>60.98</td><td>51.47</td><td>57.36</td><td>60.01</td></tr><tr><td>Literature</td><td>64.13</td><td>58.98</td><td>66.93</td><td>54.61</td><td>56.08</td><td>61.99</td></tr><tr><td>Music</td><td>70.01</td><td>71.06</td><td>73.53</td><td>61.58</td><td>68.01</td><td>65.93</td></tr><tr><td>Politics</td><td>63.58</td><td>66.27</td><td>74.47</td><td>55.76</td><td>65.99</td><td>67.05</td></tr><tr><td>Science</td><td>59.01</td><td>60.03</td><td>62.60</td><td>58.73</td><td>60.99</td><td>62.41</td></tr><tr><td>Movie</td><td>64.72</td><td>67.78</td><td>72.17</td><td>66.58</td><td>67.86</td><td>71.65</td></tr><tr><td>Restaurant</td><td>50.21</td><td>50.31</td><td>52.99</td><td>52.50</td><td>50.43</td><td>52.90</td></tr><tr><td>Average</td><td>60.84</td><td>62.11</td><td>66.24</td><td>57.32</td><td>60.96</td><td>63.13</td></tr></table>


Table 8: Comparison with self-consistency (SC) methods. T-SC employs a fully trained model to generate multiple outputs by adjusting the Temperature hyperparameter to the ensemble, while E-SC ensemble outputs from different Expert models.


## 4.11 Comparing with Self-Consistency

Considering that we trained multiple expert models, an intuitive approach is self-consistency (SC) (Wang et al., 2023b), which ensembles multiple outputs through voting. Results are shown in Table 8, where T-SC employs a full-trained model to generate multiple outputs by adjusting the Temperature hyperparameter to ensemble, while 

E-SC ensemble outputs from different Experts (though this slightly deviates from the strict definition of “self”-consistency, we refer to "SC" for simplicity). For our NER task, traditional SC (T-SC) shows limited improvement, while E-SC offers more significant gains due to greater output diversity. However, both SC strategies perform worse than our method and require higher inference costs. 

## 4.12 Framework Generalization

Non-strict Domain In our experiments, each expert corresponds to a domain with real-world significance, such as news or law. To explore a more flexible scenario, we further test on non-strict domains. Specifically, we cluster each domain dataset into five subsets, and then sample one subset from each domain to create five new datasets, which are used to train five new expert models. The results in Table 9 indicate that our framework remains effective. Notably, the experts in this setting are no longer tied to specific domains but function as general-domain models while retaining diverse capabilities. This suggests that the key requirement is a set of experts with complementary strengths, which can enhance overall performance through mutual reinforcement. 

Multilingual Scenarios We extend our framework to multilingual scenarios and conduct preliminary experiments on six languages from the WikiANN dataset (Pan et al., 2017), including German (de), English (en), Spanish (es), Dutch (nl), Russian (ru), and Chinese (zh). We train a model for each language. When evaluating a target language, the model trained on that language is not used. The results in Table 10 provide several initial evidence that our framework has the potential to extend to multilingual scenarios. 

## 5 Future Work

Unified IE and Other Tasks We conduct experimental analyses with NER as a case study. Some prior studies train a unified model for multiple IE tasks, including NER, relation extraction (RE), event extraction (EE), etc. Our framework can be naturally extended to a broader IE setting by incorporating additional IE data to train IE experts. Additionally, our method extends beyond these tasks to a wide range of applications. 

Detailed and Complete Design Our extended experiments investigated several optimization strategies. Further improvements could be realized by: 

<table><tr><td rowspan="2"></td><td colspan="5">Expert Model</td><td rowspan="2">Ours</td></tr><tr><td>E1</td><td>E2</td><td>E3</td><td>E4</td><td>E5</td></tr><tr><td>AI</td><td>56.98</td><td>57.98</td><td>56.26</td><td>48.51</td><td>55.46</td><td>60.88</td></tr><tr><td>Literature</td><td>65.38</td><td>65.45</td><td>55.30</td><td>56.27</td><td>62.20</td><td>67.03</td></tr><tr><td>Music</td><td>62.38</td><td>66.00</td><td>63.63</td><td>63.88</td><td>65.17</td><td>67.42</td></tr><tr><td>Politics</td><td>60.94</td><td>66.49</td><td>63.52</td><td>57.54</td><td>58.50</td><td>66.09</td></tr><tr><td>Science</td><td>62.56</td><td>63.31</td><td>61.38</td><td>58.23</td><td>65.72</td><td>66.75</td></tr><tr><td>Movie</td><td>68.75</td><td>61.97</td><td>66.58</td><td>64.68</td><td>65.02</td><td>67.98</td></tr><tr><td>Restaurant</td><td>44.06</td><td>34.94</td><td>49.57</td><td>50.05</td><td>43.35</td><td>52.66</td></tr><tr><td>Average</td><td>60.15</td><td>59.45</td><td>59.46</td><td>57.02</td><td>59.35</td><td>64.12</td></tr></table>


Table 9: Analysis of experts for non-strict domains. We employ clustering to build five source domains and train expert models.


<table><tr><td rowspan="2"></td><td colspan="6">Expert Model</td><td rowspan="2">Ours</td></tr><tr><td>de</td><td>en</td><td>es</td><td>nl</td><td>ru</td><td>zh</td></tr><tr><td>de</td><td>-</td><td>80.69</td><td>80.75</td><td>81.00</td><td>81.74</td><td>74.99</td><td>82.42</td></tr><tr><td>en</td><td>72.29</td><td>-</td><td>77.10</td><td>77.51</td><td>73.08</td><td>69.18</td><td>76.84</td></tr><tr><td>es</td><td>86.09</td><td>87.28</td><td>-</td><td>91.30</td><td>88.46</td><td>78.85</td><td>89.67</td></tr><tr><td>nl</td><td>84.31</td><td>86.75</td><td>85.02</td><td>-</td><td>84.79</td><td>81.41</td><td>87.59</td></tr><tr><td>ru</td><td>75.91</td><td>75.68</td><td>74.44</td><td>76.93</td><td>-</td><td>61.29</td><td>78.16</td></tr><tr><td>zh</td><td>49.07</td><td>49.18</td><td>46.56</td><td>48.21</td><td>49.93</td><td>-</td><td>51.37</td></tr><tr><td>Avg</td><td>73.53</td><td>75.92</td><td>72.77</td><td>74.99</td><td>75.60</td><td>73.14</td><td>77.68</td></tr></table>


Table 10: Analysis of experts for different languages. When evaluating a target language, the model trained on that language is not used.


(1) Incorporating more task-specific designs. For instance, alongside domain-level similarity, we could also leverage entity-type similarity when selecting source models. (2) Dynamically determining the number of merged models, k. Model merging may be unnecessary for certain target domains. As shown in Appendix B, when the target and source domains coincide, the single corresponding model already delivers optimal results. Section 4.5 demonstrates that the best choice of k varies across different target domains. Additionally, our framework is agnostic to the model architecture and readily extends to other model types other than the Llama and Qwen LLM families. 

## 6 Conclusion

We propose the Select and Merging (SaM) framework for NER, which dynamically selects valuable domain expert models for the target domain and employs model merging to obtain the task-specific model. Compared to prior studies, we possess superior adaptability and scalability. Experimental results demonstrate the effectiveness of our framework. Extensive analysis further provides insights into potential improvements, practical experience, and broader extensions of our approach. 

## Limitations

We acknowledge the following limitations of our work: (1) Maintaining multiple expert models introduces some additional storage overhead, despite the use of LoRA. (2) For domain similarity calculation and clustering analysis, we simply employed a widely used encoder model from the Hugging-Face repository to obtain text embeddings. Further optimization is possible. (3) Our analysis is limited to Named Entity Recognition (NER). Further experiments are needed for other IE tasks. 

## Acknowledgements

This work was supported in part by the National Natural Science Foundation of China under Grant No.62276110, No.62172039. The authors would also like to thank anonymous reviewers for their comments on improving the quality of this paper. 

## References



Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774. 





Ting Wai Terence Au, Vasileios Lampos, and Ingemar Cox. 2022. E-ner—an annotated named entity recognition corpus of legal text. In Proceedings of the Natural Legal Language Processing Workshop 2022, pages 246–255. 





Pei Chen, Haotian Xu, Cheng Zhang, and Ruihong Huang. 2022. Crossroads, buildings and neighborhoods: A dataset for fine-grained location recognition. In Proceedings of the 2022 conference of the North American chapter ofthe associationfor com putational linguistics: human language technologies, pages 3329–3339. 





Nigel Collier, Tomoko Ohta, Yoshimasa Tsuruoka, Yuka Tateisi, and Jin-Dong Kim. 2004. Introduction to the bio-entity recognition task at jnlpba. In Proceedings ofthe International Joint Workshop on Natural Language Processing in Biomedicine and its Applications (NLPBA/BioNLP), pages 73–78. 





Leon Derczynski, Kalina Bontcheva, and Ian Roberts. 2016. Broad twitter corpus: A diverse named entity recognition resource. In Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: Technical Papers, pages 1169– 1179. 





Leon Derczynski, Eric Nichols, Marieke Van Erp, and Nut Limsopatham. 2017. Results of the wnut2017 shared task on novel and emerging entity recognition. In Proceedings ofthe 3rd Workshop on Noisy Usergenerated Text, pages 140–147. 





Yuyang Ding, Juntao Li, Pinzheng Wang, Zecheng Tang, Yan Bowen, and Min Zhang. 2024a. Rethinking negative instances for generative named entity recognition. In Findings ofthe Associationfor Computational Lin guistics: ACL 2024, pages 3461–3475. 





Zhuojun Ding, Wei Wei, Xiaoye Qu, and Dangyang Chen. 2024b. Improving pseudo labels with globallocal denoising framework for cross-lingual named entity recognition. In Proceedings ofthe Thirty-Third International Joint Conference on Artificial Intelligence, pages 6252–6260. 





Rezarta Islamaj Dogan, Robert Leaman, and Zhiyong˘ Lu. 2014. Ncbi disease corpus: a resource for disease name recognition and concept normalization. Journal ofbiomedical informatics, 47:1–10. 





Chenghao Fan, Zhenyi Lu, Sichen Liu, Xiaoye Qu, Wei Wei, Chengfeng Gu, and Yu Cheng. 2025. Make lora great again: Boosting lora with adaptive singular values and mixture-of-experts optimization alignment. 





Chenghao Fan, Zhenyi Lu, Wei Wei, Jie Tian, Xiaoye Qu, Dangyang Chen, and Yu Cheng. 2024a. On giant’s shoulders: Effortless weak to strong by dynamic logits fusion. In The Thirty-eighth Annual Conference on Neural Information Processing Systems. 





Chenghao Fan, Wei Wei, Xiaoye Qu, Zhenyi Lu, Wenfeng Xie, Yu Cheng, and Dangyang Chen. 2024b. Enhancing low-resource relation representations through multi-view decoupling. Proceedings of the AAAI Conference on Artificial Intelligence, 38:17968–17976. 





Runwei Guan, Ka Lok Man, Feifan Chen, Shanliang Yao, Rongsheng Hu, Xiaohui Zhu, Jeremy Smith, Eng Gee Lim, and Yutao Yue. 2024. Findvehicle and vehiclefinder: a ner dataset for natural languagebased vehicle retrieval and a keyword-based crossmodal vehicle retrieval system. Multimedia Tools and Applications, 83(8):24841–24874. 





Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. 2025a. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948. 





Quanjiang Guo, Yihong Dong, Ling Tian, Zhao Kang, Yu Zhang, and Sijie Wang. 2025b. Baner: Boundaryaware llms for few-shot named entity recognition. In Proceedings ofthe 31st International Conference on Computational Linguistics, pages 10375–10389. 





Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. 2022. Lora: Low-rank adaptation of large language models. In International Conference on Learning Representations. 





Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Ludwig Schmidt, Hannaneh Hajishirzi, and Ali Farhadi. 2023. Editing models with task arithmetic. In The Eleventh International Conference on Learning Representations. 





Xisen Jin, Xiang Ren, Daniel Preotiuc-Pietro, and Pengxiang Cheng. 2023. Dataless knowledge fusion by merging weights of language models. In The Eleventh International Conference on Learning Representations. 





Martin Krallinger, Obdulia Rabal, Florian Leitner, Miguel Vazquez, David Salgado, Zhiyong Lu, Robert Leaman, Yanan Lu, Donghong Ji, Daniel M Lowe, et al. 2015. The chemdner corpus of chemicals and drugs and its annotation principles. Journal ofcheminformatics, 7:1–17. 





Aman Kumar and Binil Starly. 2022. “fabner”: information extraction from manufacturing process science domain literature using named entity recognition. Journal of Intelligent Manufacturing, 33(8):2393– 2407. 





Jiao Li, Yueping Sun, Robin J Johnson, Daniela Sciaky, Chih-Hsuan Wei, Robert Leaman, Allan Peter Davis, Carolyn J Mattingly, Thomas C Wiegers, and Zhiyong Lu. 2016. Biocreative v cdr task corpus: a resource for chemical disease relation extraction. Database, 2016. 





Peng Li, Tianxiang Sun, Qiong Tang, Hang Yan, Yuanbin Wu, Xuan-Jing Huang, and Xipeng Qiu. 2023. Codeie: Large code generation models are better fewshot information extractors. In Proceedings of the 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 15339–15353. 





Zixuan Li, Yutao Zeng, Yuxin Zuo, Weicheng Ren, Wenxuan Liu, Miao Su, Yucan Guo, Yantao Liu, Lixiang Lixiang, Zhilei Hu, Long Bai, Wei Li, Yidan Liu, Pan Yang, Xiaolong Jin, Jiafeng Guo, and Xueqi Cheng. 2024. KnowCoder: Coding structured knowledge into llms for universal information extraction. In Proceedings of the 62nd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 8758–8779. 





Wei Liu, Weihao Zeng, Keqing He, Yong Jiang, and Junxian He. 2024. What makes good data for alignment? a comprehensive study of automatic data selection in instruction tuning. In The Twelfth International Conference on Learning Representations. 





Zihan Liu, Yan Xu, Tiezheng Yu, Wenliang Dai, Ziwei Ji, Samuel Cahyawijaya, Andrea Madotto, and Pascale Fung. 2021. Crossner: Evaluating crossdomain named entity recognition. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pages 13452–13460. 





Zhenyi Lu, Chenghao Fan, Wei Wei, Xiaoye Qu, Dangyang Chen, and Yu Cheng. 2024. Twin-merging: Dynamic integration of modular expertise in model merging. In The Thirty-eighth Annual Conference on Neural Information Processing Systems. 





Michael S Matena and Colin Raffel. 2022. Merging models with fisher-weighted averaging. In Advances in Neural Information Processing Systems. 





Alexis Mitchell, Stephanie Strassel, Shudong Huang, and Ramez Zakhary. 2005. Ace 2004 multilingual training corpus. LDC corpora. 





Xiaoman Pan, Boliang Zhang, Jonathan May, Joel Nothman, Kevin Knight, and Heng Ji. 2017. Cross-lingual name tagging and linking for 282 languages. In Proceedings ofthe 55th annual meeting ofthe association for computational linguistics (volume 1: long papers), pages 1946–1958. 





Chaoxu Pang, Yixuan Cao, Qiang Ding, and Ping Luo. 2023. Guideline learning for in-context information extraction. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 15372–15389. 





Sameer Pradhan, Alessandro Moschitti, Nianwen Xue, Hwee Tou Ng, Anders Björkelund, Olga Uryupina, Yuchen Zhang, and Zhi Zhong. 2013. Towards robust linguistic analysis using ontonotes. In Proceedings ofthe Seventeenth Conference on Computational Natural Language Learning, pages 143–152. 





Sampo Pyysalo and Sophia Ananiadou. 2014. Anatomical entity mention recognition at literature scale. Bioinformatics, 30(6):868–875. 





Yunjia Qi, Hao Peng, Xiaozhi Wang, Bin Xu, Lei Hou, and Juanzi Li. 2024. Adelie: Aligning large language models on information extraction. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 7371–7387. 





Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. 2024. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36. 





Oscar Sainz, Iker García-Ferrero, Rodrigo Agerri, Oier Lopez de Lacalle, German Rigau, and Eneko Agirre. 2024. Gollie: Annotation guidelines improve zero-shot information-extraction. In The Twelfth International Conference on Learning Representations. 





Larry Smith, Lorraine K Tanabe, Rie Johnson nee Ando, Cheng-Ju Kuo, I-Fang Chung, Chun-Nan Hsu, Yu-Shi Lin, Roman Klinger, Christoph M Friedrich, Kuzman Ganchev, et al. 2008. Overview of biocreative ii gene mention recognition. Genome biology, 9:1–19. 





Zeliang Tong, Zhuojun Ding, and Wei Wei. 2025. Evoprompt: Evolving prompts for enhanced zero-shot named entity recognition with large language models. In Proceedings ofthe 31st International Conference on Computational Linguistics, pages 5136–5153. 





Asahi Ushio, Francesco Barbieri, Vitor Sousa, Leonardo Neves, and Jose Camacho-Collados. 2022. Named entity recognition in twitter: A dataset and analysis on short-term temporal shifts. In Proceedings ofthe 2nd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 12th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 309–319. 





Asahi Ushio and Jose Camacho-Collados. 2021. Tner: An all-round python library for transformerbased named entity recognition. In Proceedings of the 16th Conference ofthe European Chapter ofthe Association for Computational Linguistics: System Demonstrations, pages 53–62. 





Christopher Walker, Stephanie Strassel, Julie Medero, and Kazuaki Maeda. 2006. Ace 2005 multilingual training corpus. LDC corpora. 





Zhen Wan, Fei Cheng, Zhuoyuan Mao, Qianying Liu, Haiyue Song, Jiwei Li, and Sadao Kurohashi. 2023. Gpt-re: In-context learning for relation extraction using large language models. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 3534–3547. 





Xiao Wang, Weikang Zhou, Can Zu, Han Xia, Tianze Chen, Yuansen Zhang, Rui Zheng, Junjie Ye, Qi Zhang, Tao Gui, et al. 2023a. Instructuie: Multi task instruction tuning for unified information extraction. arXiv preprint arXiv:2304.08085. 





Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2023b. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations. 





Zihan Wang, Jingbo Shang, Liyuan Liu, Lihao Lu, Jiacheng Liu, and Jiawei Han. 2019. Crossweigh: Training named entity tagger from imperfect annotations. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natu ral Language Processing (EMNLP-IJCNLP), pages 5154–5163. 





Tingyu Xie, Qi Li, Jian Zhang, Yan Zhang, Zuozhu Liu, and Hongwei Wang. 2023. Empirical study of zero-shot ner with chatgpt. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 7935–7956. 





Tingyu Xie, Qi Li, Yan Zhang, Zuozhu Liu, and Hongwei Wang. 2024. Self-improving for zero-shot named entity recognition with large language models. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 2: Short Papers), pages 583–593. 





Derong Xu, Wei Chen, Wenjun Peng, Chao Zhang, Tong Xu, Xiangyu Zhao, Xian Wu, Yefeng Zheng, Yang Wang, and Enhong Chen. 2024. Large language models for generative information extraction: A survey. Frontiers ofComputer Science, 18(6):186357. 





Prateek Yadav, Derek Tam, Leshem Choshen, Colin Raffel, and Mohit Bansal. 2023. Ties-merging: Resolving interference when merging models. In Annual Conference on Neural Information Processing Systems. 





An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. 2024. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115. 





Yuming Yang, Wantong Zhao, Caishuang Huang, Junjie Ye, Xiao Wang, Huiyuan Zheng, Yang Nan, Yuran Wang, Xueying Xu, Kaixin Huang, Yunke Zhang, Tao Gui, Qi Zhang, and Xuanjing Huang. 2025. Beyond boundaries: Learning a universal entity taxonomy across datasets and languages for open named entity recognition. In Proceedings ofthe 31st International Conference on Computational Linguistics. 





Le Yu, Bowen Yu, Haiyang Yu, Fei Huang, and Yongbin Li. 2024. Language models are super mario: Absorbing abilities from homologous models as a free lunch. In Forty-first International Conference on Machine Learning. 





Urchade Zaratiana, Nadi Tomeh, Pierre Holat, and Thierry Charnois. 2024. Gliner: Generalist model for named entity recognition using bidirectional transformer. In Proceedings of the 2024 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 5364–5376. 





Chunting Zhou, Pengfei Liu, Puxin Xu, Srinivasan Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, et al. 2024a. Lima: Less is more for alignment. Advances in Neural Information Processing Systems, 36. 





Wenxuan Zhou, Sheng Zhang, Yu Gu, Muhao Chen, and Hoifung Poon. 2024b. Universalner: Targeted distillation from large language models for open named entity recognition. In The Twelfth International Conference on Learning Representations. 



## A Experimental Details

Training Data We collected 17 widely used NER datasets and categorized them into six domains based on their data sources. Table 11 presents detailed statistics for each dataset. As described in Section 3.1, we sample these datasets based on the proportion of their entity types while ensuring that the total data volume per domain remains between 10,000 and 50,000. The distribution of the training data across domains is illustrated in Figure 6, and the exact number of sampled instances per dataset is listed in the “#Sampled” column of Table 11. Figure 7 and 8 show examples of the formatted training data we constructed. 

Implementation Details We adopt the Qwen2.5- 7B <sup>3</sup> and Llama3.1-8B <sup>4</sup> as foundations and train expert models using LoRA (Hu et al., 2022). The LoRA rank is set to 32, with three training epochs, a batch size of 16, a learning rate of 2e 5, and a warmup ratio of 0.05. During LoRA training, all linear layers are activated. We set the temperature to 0 for LLMs when inference. All experiments are conducted on one NVIDIA 4090 GPU. 

## B Data Merging or Model Merging

We use “Data Merging” to denote the prior approaches of training a unified model by integrating data from multiple domains and “Model Merging” to denote synthesizing a new model by merging the parameters of expert models. This section presents a preliminary analysis that motivates the adoption of the model merging strategy. Specifically, we compare three types of models: (1) Experts, which are trained on single-domain data. (2) Data Merging, which is trained on a mixture of all domain data. (3) Model Merging, which is obtained by merging the parameters of all expert models. 

We conduct evaluations on both in-domain and out-of-domain settings, with the results presented in Tables 12 and 13. Our key observations are as follows: (1) Data Merging consistently yields suboptimal performance in both settings. (2) In in-domain tasks, Model Merging also performs suboptimally and is inferior to Data Merging. (3) In out-of-domain tasks, Model Merging generally achieves the best performance. 

The aforementioned observation is our primary motivation for introducing model merging. For in-domain tasks, directly utilizing the corresponding expert model is sufficient. For out-of-domain tasks, merging expert models improves generalization. However, experimental results indicate that in some cases, a single expert model performs better, especially for in-domain tasks where merging may be unnecessary. Combined with Section 4.5, the number of selected experts, k, could be dynamically determined, with k = 1 being a valid consideration. This is a promising direction for further enhancing the adaptability of our framework. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/906a0161-4f11-48ac-9b69-307c345782ff/329212ef60bb3d2816f4d64574a03e7710f6dd07661f044a50095395a0f33f6f.jpg)



Figure 6: Distribution of the training data.


## C Analysis of Cost

Parameter Count and Storage Cost We assume H denotes the model dimension, r the rank of LoRA adapters, L the number of layers, n the number of domain-specific experts, and V the vocabulary size. For computational simplicity, we adopt a simplified Transformer architecture (e.g., omitting grouped-query attention mechanisms) in our base models (Llama3.1 and Qwen2.5). Since our LoRA implementation applies to all linear layers, each domain-specific expert requires 18HrL additional storage parameters. Consequently, storing n experts incurs a total overhead of $n \times 1 8 H r L$ . The base model itself requires approximately $( 1 2 H ^ { 2 } +$ 13H)L + VH parameters. Given that nr H, the additional storage overhead remains negligible. During inference, we merge LoRA adapters into one or two task-specific models, achieving state-ofthe-art performance with only equivalent or doubled storage costs compared to the base model. 

<table><tr><td>Domain</td><td>Dataset</td><td>#Type</td><td>#Train</td><td>#Dev</td><td>#Test</td><td>#Sampled</td></tr><tr><td rowspan="6">Biomedical</td><td>AnatEM (Pyysalo and Ananiadou, 2014)</td><td>1</td><td>5,861</td><td>2,118</td><td>3,830</td><td>3,297</td></tr><tr><td>JNLPBA (Collier et al., 2004)</td><td>5</td><td>16,691</td><td>1,855</td><td>3,856</td><td>16,487</td></tr><tr><td>bc2gm (Smith et al., 2008)</td><td>1</td><td>12,500</td><td>2,500</td><td>5,000</td><td>3,297</td></tr><tr><td>bc4chemd (Krallinger et al., 2015)</td><td>1</td><td>30,682</td><td>30,639</td><td>26,364</td><td>3,297</td></tr><tr><td>bc5cdr (Li et al., 2016)</td><td>2</td><td>4,560</td><td>4,581</td><td>4,797</td><td>6,594</td></tr><tr><td>ncbi (Doğan et al., 2014)</td><td>1</td><td>5,432</td><td>923</td><td>940</td><td>3,297</td></tr><tr><td>Law</td><td>E-NER (Au et al., 2022)</td><td>7</td><td>9,313</td><td>1,164</td><td>1,165</td><td>10,000</td></tr><tr><td rowspan="4">News</td><td>ACE04 (Mitchell et al., 2005)</td><td>7</td><td>6,202</td><td>745</td><td>812</td><td>9,722</td></tr><tr><td>ACE05 (Walker et al., 2006)</td><td>7</td><td>7,299</td><td>971</td><td>1,060</td><td>9,722</td></tr><tr><td>conllpp (Wang et al., 2019)</td><td>4</td><td>14,041</td><td>3,250</td><td>3,452</td><td>5,555</td></tr><tr><td>OntoNotes (Pradhan et al., 2013)</td><td>18</td><td>59,924</td><td>8,528</td><td>8,262</td><td>25,000</td></tr><tr><td rowspan="4">Social media</td><td>WNUT2017 (Derczynski et al., 2017)</td><td>6</td><td>3,394</td><td>1,009</td><td>1,287</td><td>6,094</td></tr><tr><td>HarveyNER (Chen et al., 2022)</td><td>4</td><td>3,967</td><td>1,301</td><td>1,303</td><td>4,063</td></tr><tr><td>BroadTweetCorpus (Derczynski et al., 2016)</td><td>3</td><td>5,334</td><td>2,001</td><td>2,000</td><td>3,047</td></tr><tr><td>TweetNER7 (Ushio et al., 2022)</td><td>7</td><td>7,111</td><td>886</td><td>576</td><td>7,110</td></tr><tr><td>STEM</td><td>FabNER (Kumar and Starly, 2022)</td><td>11</td><td>9,435</td><td>2,182</td><td>2,064</td><td>10,000</td></tr><tr><td>Traffic</td><td>FindVehicle (Guan et al., 2024)</td><td>8</td><td>21,565</td><td>20,777</td><td>20,777</td><td>21,565</td></tr></table>


Table 11: Statistics of raw training data and the number of sampled instances for training.


<table><tr><td colspan="2"></td><td>Biomedical</td><td>Legal</td><td>News</td><td>Socia media</td><td>STEM</td><td>Traffic</td></tr><tr><td rowspan="6">Experts</td><td>Biomedical</td><td>82.57</td><td>40.02</td><td>36.61</td><td>48.09</td><td>27.27</td><td>21.05</td></tr><tr><td>Legal</td><td>40.40</td><td>84.79</td><td>42.36</td><td>46.90</td><td>17.92</td><td>32.48</td></tr><tr><td>News</td><td>53.83</td><td>48.40</td><td>85.94</td><td>48.28</td><td>23.53</td><td>41.84</td></tr><tr><td>Social media</td><td>53.11</td><td>41.31</td><td>42.06</td><td>66.57</td><td>24.17</td><td>22.96</td></tr><tr><td>STEM</td><td>28.22</td><td>16.26</td><td>22.81</td><td>23.53</td><td>76.99</td><td>24.96</td></tr><tr><td>Traffic</td><td>42.14</td><td>28.23</td><td>32.67</td><td>42.01</td><td>20.23</td><td>99.96</td></tr><tr><td colspan="2">Data Merging</td><td>80.53</td><td>81.87</td><td>84.97</td><td>60.27</td><td>77.40</td><td>98.91</td></tr><tr><td colspan="2">Model Merging</td><td>67.57</td><td>64.13</td><td>55.47</td><td>56.25</td><td>31.80</td><td>45.29</td></tr></table>


Table 12: In-domain performance of expert models, full data trained model (Data Merging), and model obtained from merging all expert models (Model Merging).


<table><tr><td colspan="2"></td><td>AI</td><td>Literature</td><td>Music</td><td>Politics</td><td>Science</td><td>Movie</td><td>Restaurant</td></tr><tr><td rowspan="6">Experts</td><td>Biomedical</td><td>56.90</td><td>59.01</td><td>63.98</td><td>65.73</td><td>60.21</td><td>57.96</td><td>39.41</td></tr><tr><td>Legal</td><td>39.85</td><td>56.88</td><td>64.33</td><td>62.72</td><td>55.92</td><td>63.26</td><td>47.31</td></tr><tr><td>News</td><td>41.77</td><td>41.03</td><td>54.74</td><td>40.36</td><td>49.66</td><td>62.13</td><td>39.28</td></tr><tr><td>Social media</td><td>53.84</td><td>55.83</td><td>61.14</td><td>60.50</td><td>57.24</td><td>59.29</td><td>42.63</td></tr><tr><td>STEM</td><td>41.01</td><td>41.35</td><td>43.35</td><td>45.41</td><td>37.38</td><td>46.61</td><td>26.78</td></tr><tr><td>Traffic</td><td>49.97</td><td>50.82</td><td>62.20</td><td>64.09</td><td>53.25</td><td>65.45</td><td>52.11</td></tr><tr><td colspan="2">Data Merging</td><td>51.38</td><td>53.46</td><td>61.12</td><td>54.99</td><td>59.39</td><td>66.66</td><td>52.69</td></tr><tr><td colspan="2">Model Merging</td><td>57.47</td><td>61.93</td><td>64.66</td><td>60.28</td><td>61.08</td><td>69.43</td><td>47.78</td></tr></table>


Table 13: Our-of-domain performance of expert models, full data trained model (Data Merging), and model obtained from merging all expert models (Model Merging).


Computation FLOPs Analysis Compared to multi-task full fine-tuning (FFT), our approach utilizes the same amount of training data to produce multiple LoRA models, resulting in identical computational costs during training. During inference, however, our method employs a single task-specific model—either derived from a single strategy or by integrating models from both strategies—which requires only one LoRA adapter. This achieves comparable inference costs to multi-task FFT while delivering superior performance. When leveraging models from both strategies simultaneously, our approach incurs twice the inference cost but further enhances performance, offering a flexible trade-off between efficiency and effectiveness. Table 14 compares our approach with several recent works that train a unified model across multiple domains. We compare (1) the amount of training data (with instance-level data size provided for reference), (2) the number of inference rounds required for model prediction, and (3) storage space requirements (with model size as the reference unit). We also report the average performance. 

<table><tr><td rowspan="2"></td><td colspan="5">Unified Models</td><td rowspan="2">SaM</td><td rowspan="2"><eq>SaM_{cco}</eq></td></tr><tr><td>InstructUIE</td><td>UniNER</td><td>GoLLIE</td><td>B2NER</td><td>Ours</td></tr><tr><td>Training Instances</td><td>215.9K</td><td>45.9K</td><td>165.2K</td><td>51.9K</td><td>148.1K</td><td>148.1K</td><td>148.1K</td></tr><tr><td>Inference Times</td><td>1×</td><td>1×</td><td>1×</td><td>1×</td><td>1×</td><td>2×</td><td>1×</td></tr><tr><td>Storage (Normalized)</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1+0.02n</td><td>1+0.02n</td></tr><tr><td>Performance (Average)</td><td>47.84</td><td>61.79</td><td>58.39</td><td>64.57</td><td>60.94</td><td>66.24</td><td>65.49</td></tr></table>


Table 14: Comparison of Resource Requirements. We compare unified models trained across multiple domains with our merging-based approach. $\mathbf { S a M } _ { e c o }$ (economic) refers to integrating two task-specific models into a single one (details in Section 4.4). For storage, we normalize the value by setting the model size to 1. Here, n represents the number of experts. We achieve superior results with minimal additional overhead, particularly with our $\mathbf { S a M } _ { e c o }$


<table><tr><td rowspan="2"></td><td colspan="4">Llama</td><td colspan="4">Qwen</td></tr><tr><td>Mode1</td><td>Mode2</td><td>Mode3</td><td>SaM(Ours)</td><td>Mode1</td><td>Mode2</td><td>Mode3</td><td>SaM(Ours)</td></tr><tr><td>AI</td><td>60.36</td><td>58.19</td><td>61.31</td><td>60.01</td><td>54.73</td><td>55.62</td><td>57.90</td><td>60.98</td></tr><tr><td>Literature</td><td>56.86</td><td>62.68</td><td>58.90</td><td>61.99</td><td>60.16</td><td>59.87</td><td>63.42</td><td>66.93</td></tr><tr><td>Music</td><td>63.78</td><td>66.50</td><td>66.85</td><td>65.93</td><td>68.98</td><td>72.89</td><td>72.42</td><td>73.53</td></tr><tr><td>Politics</td><td>66.05</td><td>68.27</td><td>61.30</td><td>67.05</td><td>71.16</td><td>75.42</td><td>73.61</td><td>74.47</td></tr><tr><td>Science</td><td>58.37</td><td>61.73</td><td>63.12</td><td>62.41</td><td>62.19</td><td>59.58</td><td>62.78</td><td>62.60</td></tr><tr><td>Movie</td><td>70.66</td><td>70.66</td><td>70.50</td><td>71.65</td><td>67.07</td><td>68.91</td><td>73.44</td><td>72.17</td></tr><tr><td>Restaurant</td><td>52.82</td><td>52.85</td><td>52.23</td><td>52.90</td><td>51.69</td><td>51.69</td><td>54.84</td><td>52.99</td></tr><tr><td>Average</td><td>61.27</td><td>62.98</td><td>62.03</td><td>63.13</td><td>62.28</td><td>63.43</td><td>65.49</td><td>66.24</td></tr></table>


Table 15: Merging into a single task model. Complete experimental results of Section 4.4.


## D Experimental Supplements

Section 4.4 extracts a single expert set for merging and presents the results based on Qwen. Here, we supplement the results of Llama in Table 15. 

Our framework is based on two perspectives: Domain Similarity (DS) and Sampling Evaluation (SE). The experimental section reports the overall performance of the framework. Here, we provide several additional experimental results regarding these two strategies, respectively. 

Section 4.5 explores the relationship between model performance and the number of merged experts. Here, we present results for each strategy, as shown in Figures 9 and 10. It can be seen that the two strategies exhibit similar overall trends, but with distinct differences. This further indicates that both strategies are important, highlighting the importance and complementarity of each. 

Section 4.9 discusses fine-grained adaptation. Due to the compact nature of the target domain, this does not bring improvements and may even reduce the performance. Here, we analyze the performance of each strategy individually. As shown in Figure 11, the Domain Similarity strategy has a minimal impact on the subdivision (with a performance difference of less than 0.2 points on average), supporting the hypothesis that the target domain is already too homogeneous to generate distinct splits. In contrast, Figure 12 shows significant changes when using Sampling Evaluation, with a general downward trend, which accounts for the overall performance degradation. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/906a0161-4f11-48ac-9b69-307c345782ff/495aa8aa4c3238e961cba5c09f7226a73d4a576173e786005faf7db86f6f360e.jpg)



Figure 7: Formatted training data example. The example consists of task instructions, inputs, and outputs for training. For the task instructions, the Data Source Description, Entity Type drop/mask, and ICL Demonstrations are optional, with details in section 3.1. We adopt three output formats: JSON, enumeration, and natural language descriptions. The output format of ICL Demonstrations and Answers should be consistent with the specified.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/906a0161-4f11-48ac-9b69-307c345782ff/c7540798f19657d870488fd9ab9e962b27eb764f455c18425997bfdb8c3d95a8.jpg)



Figure 8: Another example of our formatted training data. The instance here is the same as that of Figure 7, adopting the entity type drop and mask processing methods.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/906a0161-4f11-48ac-9b69-307c345782ff/492d88ee9de1f1404b00887a3f1fe9c01691e344c2ebb47fe48f596ae70230cd.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/906a0161-4f11-48ac-9b69-307c345782ff/609208172b9fff927f6711ca7e8898fa256ad9dcc8db7d87c4c25ae4d5bb13fd.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/906a0161-4f11-48ac-9b69-307c345782ff/e1d785d6bb493822c363d1e523824f2d4dd77fce593f05bc5a5a522e3f7e531e.jpg)



Figure 9: Performance changes with the number of expert models (denotes as k). The horizontal axis is the number of experts for merging, and the vertical denotes the entity-level F1 scores. Only using Domain Similarity for expert selection. It can be seen that the optimal k varies across target domains, typically ranging from 2 to 4.



Figure 10: Performance changes with the number of expert models. The horizontal axis is the number of experts for merging, and the vertical denotes the entity-level F1 scores. Only using Sampling Evaluation for expert selection. It exhibits similar overall trends to Figure 9, but with distinct differences, indicating that both selecting strategies are important and complementary.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/906a0161-4f11-48ac-9b69-307c345782ff/895a04202e63bca351b1a9d34d7f9c116c21b979724d89956fb11c2118b59eeb.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/906a0161-4f11-48ac-9b69-307c345782ff/a05646d7f3a9c4da10824ff07c318fac8181f75fd464aacaa624cdc30cf8c7ba.jpg)



Figure 11: Performance changes with the number of data splits. The horizontal axis is the number of splits, and the vertical denotes the entity-level F1 scores. Only using Domain Similarity for expert selection. It can be seen that the Domain Similarity strategy has a minimal impact, since data from the target domain may be too homogeneous.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/906a0161-4f11-48ac-9b69-307c345782ff/f0d04339de8b131dcae028778e53d5c753d27c18fdca0cb0adeb7e040b75bf37.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/906a0161-4f11-48ac-9b69-307c345782ff/83887d5549d97f24efe5852e9d213a9a175eb74a4de8fbb16adaff647b256100.jpg)



Figure 12: Performance changes with the number of data splits. The horizontal axis is the number of splits, and the vertical denotes the entity-level F1 scores. Only using Sampling Evaluation for expert selection. It shows significant changes when using Sampling Evaluation, with a general downward trend.
