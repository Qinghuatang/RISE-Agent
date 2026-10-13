# Language Fusion for Parameter-Efficient Cross-lingual Transfer

Philipp Borchert<sup>1,2</sup>, Ivan Vulic´<sup>3</sup>, Marie-Francine Moens<sup>4</sup> Jochen De Weerdt<sup>1</sup> 

<sup>1</sup>Research Centre for Information Systems Engineering, KU Leuven, Belgium 

<sup>2</sup>IESEG School of Management, France <sup>3</sup>Language Technology Lab, University of Cambridge, United Kingdom <sup>4</sup>Department of Computer Science, KU Leuven, Belgium 

## Abstract

Limited availability of multilingual text corpora for training language models often leads to poor performance on downstream tasks due to undertrained representation spaces for languages other than English. This ‘under-representation has motivated recent cross-lingual transfer methods to leverage the English representa tion space by e.g. mixing English and ‘non-English’ tokens at the input level or extending model parameters to accommodate new languages. However, these approaches often come at the cost of increased computational complexity. We propose Fusion for Language Representations (FLARE) in adapters, a novel method that enhances representation quality and downstream performance for languages other than English while maintaining parameter efficiency. FLARE integrates source and tar get language representations within low-rank (LoRA) adapters using lightweight linear trans formations, maintaining parameter efficiency while improving transfer performance. A series of experiments across representative crosslingual natural language understanding tasks, including natural language inference, questionanswering and sentiment analysis, demonstrate FLARE’s effectiveness. FLARE achieves perfor mance improvements of 4.9% for Llama 3.1 and 2.2% for Gemma 2 compared to standard LoRA fine-tuning on question-answering tasks, as measured by the exact match metric. 

## 1 Introduction

Representation degradation for ‘non-English’ languages poses a challenge in the context of pretrained multilingual language models (mPLMs)<sup>2</sup>. 

Large-scale English text corpora are widely available for self-supervised pretraining, resulting in superior representation quality and downstream task performance when compared to low(er)-resource languages (Lauscher et al., 2020; Yang et al., 2022). Despite the substantial improvements, the imbalance in pretraining resources still substantially reduces performance (Winata et al., 2022). 

Cross-lingual transfer (termed XLT henceforth) aims to narrow this performance gap by transferring task-specific knowledge acquired in highresource languages to lower-resource languages (Ruder et al., 2019). Given the dominance of English in pretraining corpora, machine translations (MT) are frequently utilized to avoid processing non-English data (Shi et al., 2010; Artetxe et al., 2020, 2023; Ansell et al., 2023). However, translation can result in information loss, including the loss of cultural nuances, which can negatively impact downstream task performance (Conia et al., 2024). Various XLT techniques address this issue by leveraging both source and target language representation spaces, such as language mixup (Yang et al., 2022) and concatenating multilingual input sequences for in-context XLT (Kim et al., 2024; Tanwar et al., 2023; Cueva et al., 2024). These approaches, while improving XLT, typically focus on representations in a specific mPLM layer or require extensive training and computational resources by extending the input length. 

Parameter-efficient fine-tuning (PEFT) methods are designed to acquire new knowledge and specialize general-purpose models for specific tasks or domains while minimizing the number of extra parameters required and keeping the large underlying mPLM frozen (Hu et al., 2022; Pfeiffer et al., 2023). In particular, bottleneck-style adapters, such as low-rank adapters (LoRA), extract relevant features from new data by compressing model representations with the assumption that task information can be captured in a lower-dimensional space (Houlsby et al., 2019; Hu et al., 2022). This directly aligns with the XLT objectives, providing resourceefficient language and task adaptation capabilities. In XLT, adapters are widely used for acquiring task and language knowledge (Pfeiffer et al., 2020). Yet, the extent of knowledge transfer across languages within adapters remains underexplored. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/96539746-f7a6-4f58-bc71-aca9b8949d34/fba72471a12df55f568eb5b6ecaffae344dae839b1f4dfbade99699a126547ca.jpg)



Figure 1: Fusion of source and target representations in LoRA adapters inserted within the query and value matrices. The representations are fused in the adapter bottlenecks and the outputs are added + to the query and value outputs before softmax activation.


In this work, we introduce Fusion for Language Representations (FLARE), a novel approach that merges latent representations from different languages within lower-dimensional adapter bottlenecks to enable parameter-efficient XLT. By merging representations from high-resource languages like English into target language representations through lightweight fusion functions, such as addition or multiplication, FLARE facilitates effective cross-lingual information transfer with minimal computational overhead. As illustrated in the schematic overview in Figure 1, FLARE performs token-wise fusion of source and target language representations within each transformer block, without adding additional parameters to LoRA and maintaining computational efficiency. FLARE is applied during task-specific fine-tuning, enabling models to extract knowledge from both the source and target languages and apply it effectively in the target language. 

Our experiments demonstrate FLARE’s versatility and effectiveness across tasks like natural language inference, sentiment classification, and question answering, using encoder-only, encoder-decoder, and decoder-only multilingual pre-trained language models (mPLMs). It is particularly beneficial for downstream tasks that involve text generation, such as question answering. For instance, FLARE improves the absolute exact match performance for Llama 3.1 and Gemma 2 on the TyDiQA dataset by 4.9% and 2.2%, respectively. Further experiments illustrate that computational efficiency can be further enhanced by using latent translations as source language inputs in FLARE, and demonstrate the flexibility of the method, which is orthogonal to the choice of mPLMs and MT systems. 

Contributions. 1) We introduce FLARE, a novel method that fuses language representations within adapter bottlenecks for parameter-efficient crosslingual transfer. 2) Our approach improves performance across diverse multilingual downstream tasks, particularly benefiting text generation tasks (e.g., applied to question answering). 3) We demonstrate the adaptability of our approach by incorporating machine translation encoder representations directly into the mPLM. 

## 2 Related Work

Cross-lingual Representation Transfer. Improving performance for underrepresented languages with mPLMs often involves aligning and combining latent representations from different languages (Oh et al., 2022). Several methods have been proposed to achieve this, including concatenating multilingual input sequences to leverage a shared representation space (Kim et al., 2024; Tanwar et al., 2023; Cueva et al., 2024). Another line of work focuses on projection-based methods, where target language representations are projected onto high-resource languages, such as English, to enhance feature extraction (Xu et al., 2023). Yang et al. (2022) introduced X-Mixup, which combines source and target representations in one specific mPLM layer using cross-attention during downstream task adaptation. Building on this idea, Cao et al. (2023) proposed using cross-attention with additional semantic and token-level alignment loss terms. In contrast, our FLARE method provides a more parameter-efficient approach by directly merging latent source and target language representations within adapter bottlenecks, thereby contributing to the stream of parameter-efficient XLT. 

Representation fusion has also been applied to integrate information across different modalities, such as vision and language (Fang et al., 2021; Ramnath et al., 2021). For instance, Qu et al. (2025) used feature routing in cross-modal visionlanguage tasks, guiding language model representations through LoRA bottlenecks using the last hidden state of a vision model. Our work differs in its scope and fusion methodology: FLARE extracts richer representations from the source and target languages by capturing layer-wise representations for each transformer block in the mPLMs. Moreover, by ensuring dimensional align ment, we perform token-wise representation fusion within adapter bottlenecks, thereby transferring finer-grained information across languages. 

PEFT in Multilingual Language Models and Cross-Lingual Transfer. PEFT aims to incorporate task or language-specific knowledge into mPLMs without updating all model weights (Pfeiffer et al., 2020). Most prominent techniques include sparse fine-tuning, which selectively updates model parameters (Ansell et al., 2022), and inserting adapter modules that reduce trainable parameters to a small fraction of total weights of the underlying mPLM (Houlsby et al., 2019). Furthermore, PEFT modules are composable, allowing for the combination of information from multiple modules (Wang et al., 2022; Lee et al., 2022b). Bottleneck adapters, such as LoRA (Hu et al., 2022) and its variants (Liu et al., 2024), are widely used for finetuning language models. These adapters project model representations into a lower-dimensional space, creating a bottleneck that regulates information flow (Houlsby et al., 2019). 

In XLT, mixtures of task and language adapters are used to merge language representation spaces effectively (Lee et al., 2022b). For example, AdaMergeX combines the weights of adapters trained on task data in English with those trained on target language data. (Zhao et al., 2024). In contrast, our fusion approach enables the model to learn from the interactions between source and target languages by combining inputs from multiple languages during the adapter fine-tuning process. This allows for a more dynamic transfer of taskspecific knowledge between languages. 

## 3 Methodology

## 3.1 Language Representation Fusion

Our methodology is based on the hypothesis that incorporating English with target language representations enhances cross-lingual knowledge transfer and distills task-relevant information into the target language. We assume (typically MT-created) parallel corpora $\mathcal { P } = \{ ( x ^ { S } , x ^ { \bar { T } } ) \}$ during task finetuning, where x are instances in the respective source and target language. Our methodology focuses on using machine-translated parallel data, where widely available English data is translated into the target language for training, and target language data is translated into English during inference. This approach reflects practical real-world scenarios, as human-annotated data is often scarce and costly. 

Yang et al. (2022) introduced cross-lingual manifold mixup (X-Mixup), aligning multilingual representations within a specific transformer layer using consistency loss terms and a cross-attention module. However, this method introduces additional model parameters and shows performance variability depending on the choice of the mixup layer. Another potentially effective method for aligning multilingual representations is to concatenate source and target language input sequences $x ^ { S , T } = [ x ^ { S } ; x ^ { T } ]$ where $x \in \mathbb { R } ^ { 2 m }$ , with m representing the sequence length of both source and target languages. This so-called input-levelfusion enables cross-lingual knowledge transfer across all layers of the mPLM, facilitating in-context learning, which typically does not require additional training (Cueva et al., 2024). However, this approach can become computationally expensive, especially for longer inputs, due to increased input sequence lengths and encounters scalability issues related to the context length limitations in mPLMs. 

To address these limitations, we propose FLARE, a method for representation-level language fusion within bottleneck adapters, as illustrated in Figure 2. Instead of extending the input, FLARE processes source and target language representations independently and fuses them only within the adapters, thus preserving computational efficiency. Source language representations $v _ { i } ^ { S }$ , extracted from the frozen mPLM without adapters, and target language representation $v _ { i } ^ { T }$ at transformer block i are down-projected using $W ^ { d o w n }$ and combined with fusion function ϕ (see Section 3.2) to create a fused representation $h \ = \ \phi \left( v _ { i + 1 } ^ { S } W ^ { d o w n } , v _ { i } ^ { T } W ^ { d o w n } \right)$ where $\boldsymbol { h } \in \mathbb { R } ^ { m \times r }$ with sequence length m and bottleneck dimensions r (Qu et al., 2025). We utilize the source representation $v _ { i + 1 } ^ { S }$ , which has been processed by the subsequent transformer block, to leverage task-specific information extracted from the source language. Following a standard LoRA procedure, this fused low-rank representation is then up-projected and added to the frozen attention outputs $v ^ { 0 }$ to form the target language output representation $v _ { i + 1 } ^ { T } = h W ^ { u p } \dot { + } v ^ { 0 }$ of the attention block. 

This enhances model performance during task adaptation in the target language by directing the model’s attention to task-relevant information. Thereby, the adapter bottleneck is used for crosslingual knowledge transfer, as well as task and language adaptation. A key advantage of FLARE is the reduction in computational complexity, thereby enhancing parameter efficiency for both task and language adaptation. By processing multilingual inputs separately and only fusing highly compressed representations within adapter bottlenecks, our method avoids the computational overhead associated with quadratic scaling in attention computations for model dimensions $d ,$ thus enhancing resource efficiency. Furthermore, the memory requirements are limited to the last hidden states obtained from the output of each transformer block. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/96539746-f7a6-4f58-bc71-aca9b8949d34/1b4a3c4bad22eeee079138eb1363b8f997d99850862ad49044c9f9d613b0bc2c.jpg)



Figure 2: During the forward pass with FLARE, source language representations $x ^ { S }$ are processed by transformer block i and before fusion with target language representations $x ^ { T }$ . Source representations are obtained by inferencing the mPLM without the fusion adapters.


Moreover, our fusion approach is agnostic regarding the source language representation. We exploit this flexibility in the FLARE MT variant, which explores the impact of reducing computational resources for processing the source language on cross-lingual transfer performance. Specifically, FLARE MT utilizes representations from a MT encoder as ‘latent translations’ that serve as source language representations. This avoids discretizing the translation as text through the MT decoder. FLARE MT further enhances resource efficiency compared to regular FLARE by bypassing the forward pass of the source language in the mPLM. We extract a single representation (latent translation) from the MT encoder by processing the target language input $v ^ { T } = \mathcal { M } \mathbf { \Psi } ( \mathbf { \dot { x } } ^ { T } )$ , where $v ^ { T } \in \bar { \mathbb { R } } ^ { m \times d _ { \mathcal { M } } }$ 

To ensure compatibility between the dimensionality of the MT encoder outputs and the mPLM, we utilize a linear projection layer $W ^ { p r o j }$ . This projection is jointly trained during the adaptation to the downstream task, ensuring resource efficiency. The up-projected representation $v ^ { T } W ^ { p r o j }$ is fused with the target language representation within the adapter bottlenecks of each mPLM layer, as displayed in Figure 3. 

## 3.2 Fusion Functions

To fuse cross-lingual representations in bottleneck adapters, we evaluate both linear and nonlinear transformations that do not require additional model parameters, alongside cross-attention. We extract token-wise representations from source and target language sequences, capturing rich contextual information at the token level. 

The down-projected representations in the adapter bottlenecks for source and target languages are denoted as $S = v ^ { S } W ^ { d o w n }$ and $T = v ^ { T } \bar { W } ^ { d o w n }$ where S and $T$ are representations of dimensions $\mathbb { R } ^ { m \times r }$ These representations are subsequently combined at the token level through the following fusion functions: 

1. element-wise addition (add): $S + T$ 

2. element-wise multiplication $( m u l ) \colon S \circ T$ 

3. cross-attention:<sup>3</sup> softmax $\left( \frac { W _ { a } ^ { Q } S \bigl ( W _ { a } ^ { K } T \bigr ) ^ { \prime } } { \sqrt { r } } \right) W _ { a } ^ { V } T$ 

$W _ { a } ^ { Q } , W _ { a } ^ { K }$ and $W _ { a } ^ { V }$ are the weight matrices of the query, key and value projections in the adapter $^ { a , }$ respectively, and ′ denotes the matrix transpose. We focus on lightweight linear transformations to maintain parameter and computational efficiency. 

Additionally, linear fusion functions are extended with non-linear transformations through rectified linear units $R e L U \left( S \right)$ and $R e L U \left( T \right)$ (Qu et al., 2025). This allows for selective information flow in token representations, which can be particularly beneficial for multilingual input sequences that may be misaligned at the token level. By introducing non-linear transformation functions, we can restrict the propagation of misaligned information, potentially leading to improved task performance. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/96539746-f7a6-4f58-bc71-aca9b8949d34/5a3d9fbff52b38505e634339f4dd1913b2788cd39bdb2a2169e76c0451cd61c8.jpg)



Figure 3: Illustration of the FLARE MT variant where projected encoder representations from an MT model are directly fused with target language representations within the fusion adapters in the mPLM. Encoder rep resentations from the MT model serve as latent translations, avoiding discretization in the decoder.


## 3.3 Training

To adapt the mPLM to downstream tasks in the target language, we insert LoRA fusion adapters into the query and value weight matrices of the mPLM that has been previously fine-tuned on English task data, referred to as the base model. These adapters implement fusion function ϕ that combines source and target language input representations into a single fused representation. Consistent with standard PEFT training, only the task head and LoRA parameters are trainable, while all other parameters remain frozen. 

During the forward pass, illustrated in Figure 2, we extract representations from both the source and target languages at each transformer block. Source language representations are obtained from the base model without fusion adapters. These layer-wise representations are stacked in matrix $\bar { V ^ { S } } \in \mathbb { R } ^ { l \times m \times \bar { d } }$ , where l represents the number of layers in the mPLM. Target language representations are obtained during the forward pass through the base model with fusion adapters. In our FLARE approach, the source and target language representations are compressed to lower dimensions r d using the adapter’s down-projection $W ^ { d o w n }$ 

The compressed representations are then combined through the fusion function, and decompressed in the up-projection. By sharing the down-projection layers for both source and target language representations before fusion, we hypothesize that the model’s reliance on the English representation space is reduced. 

## 4 Experimental Setup

## 4.1 Underlying Models and Baselines

mPLMs. First, we again note that the design of FLARE makes it transformer architecture-agnostic; it is directly applicable to encoder, decoder and encoder-decoder models. Our experiments are thus based on various mPLMs including the encoderonly XLM-R Large (550M) (Conneau et al., 2020), the encoder-decoder mT5-XL (3.7B) (Xue et al., 2021), the decoder-only Llama 3.1 (8B) (Grattafiori et al., 2024), and the decoder-only Gemma 2 (9B) (Gemma Team et al., 2024). 

Fine-Tuning Setup. We follow a modular XLT approach where the mPLM is fine-tuned on English task data and subsequently adapted using task data in the target language (Zhao et al., 2021). For decoder-only models like Llama and Gemma, we use a causal language modeling objective for finetuning, and the models generate predictions as text accordingly. We employ the QLoRA fine-tuning approach with 4-bit quantization and insert LoRA adapters in all linear layers for Llama and Gemma models (Dettmers et al., 2023). Importantly, we apply representation fusion in FLARE only in the attention modules, ensuring a consistent experimental setup across different transformer architectures. The LoRA configurations use r = 64 and α = 128, where r corresponds to the bottleneck dimensions and α is a scaling factor applied to the low-rank updates before merging them with the frozen attention outputs. Detailed hyperparameter configurations for each model can be found in Table 5 in the appendix. 

Baselines. We evaluate FLARE against several baselines, including zero-shot cross-lingual transfer, translate-test, as well as translate-train methods such as regular LoRA fine-tuning, X-Mixup, and input-level fusion. We focus on baseline methods that fine-tune on task-specific data in the target language, as these provide the most competitive performance. All translate-train models are trained with the same LoRA configurations. Unless otherwise specified, FLARE models are trained using the add+relu fusion function, with a detailed comparison of fusion functions presented in Table 2. Model checkpoints are selected based on validation data that was machine-translated from English to the respective target languages. 

X-Mixup aligns source and target language representations through cross-attention in one specific transformer layer and further aligns model outputs using consistency loss terms (Yang et al., 2022). In contrast, input-level fusion combines source and target language texts directly in the input prompt of the mPLM, doubling the sequence length (Kim et al., 2024; Cueva et al., 2024). More details on the baselines below: 

Zero-Shot XLT. The base model fine-tuned on English task data is directly evaluated on test data in the target languages without further training. 

Translate-Test. Test sets in each target language are translated into English using NLLB (NLLB Team et al., 2022). Subsequently, the base model is evaluated on these machine-translated test sets.<sup>4</sup> Translate-Train. The base model is fine-tuned on machine-translated task data in the respective target languages. The training data comprises instances translated from English to the target language using NLLB. For fusion methods and X-Mixup, we obtain the required ‘silver’ parallel data also through MT (using NLLB). The training set consists of parallel sets of English and MT-ed instances, whereas the validation and test sets consist of parallel target language instances and corresponding machine translations into English. We posit that the assumed absence of gold translations both during training and during inference is the most realistic evaluation of FLARE models. 

## 4.2 Evaluation Tasks and Datasets

XNLI consists of machine-translated sentence pairs that are translated from English to 15 languages (Conneau et al., 2018). The task involves determining whether a sentence entails, contradicts, or is neutral to a given premise. 

NusaX is a human-annotated sentiment classification dataset that spans 11 Indonesian languages, including low-resource languages (Winata et al., 2023). With 500 labeled instances for each language, the dataset evaluates few-shot adaptation. 

TyDiQA-GoldP is a human-annotated extractive QA dataset covering 8 languages (Clark et al., 2020). The task is to extract the answer spans from context passages. 

Additional information on evaluation languages and datasets used for source language fine-tuning are available in Table 10 in the appendix. 

## 4.3 Machine Translations

We utilize the NLLB 3.3B variant (NLLB Team et al., 2022) as the main MT model, employing greedy decoding to obtain translations (Artetxe et al., 2023). Additionally, FLARE MT utilizes the encoder of the NLLB 600M variant to generate latent translations. To maintain consistency in our experimental setup, we also translate languages that are not directly supported by NLLB. Specifically, Madurese (mad) and Ngaju (nij) are translated using the Indonesian language identifier, as these languages are not supported by NLLB<sup>5</sup> (Winata et al., 2023). For translating extractive QA datasets, we employ EasyProject (Chen et al., 2023), which involves enclosing answer spans within marker tokens prior to translation with NLLB. This method allows us to determine the position of the translated answer spans by locating these marker tokens in the translated text. Instances that fail to retain the marker tokens in the translated output are excluded from evaluation. 

## 5 Results and Discussion

Main Results. The results displayed in Table 1 confirm our hypothesis that task-specific knowledge can be efficiently transferred from English to other languages within adapter bottlenecks. Our proposed approach, FLARE, consistently surpasses all baselines across various tasks, demonstrating robust performance and validating the effectiveness of our method. It improves the average performance, averaged across metrics for all tasks, by 2.14% and 1.27% for Llama and Gemma, respectively, when compared to standard LoRA finetuning. The most substantial performance gains are observed on the TyDiQA dataset, particularly for text generation tasks with decoder-only models. FLARE substantially improves performance on this dataset, with the largest gains achieved on Indonesian, Russian, and Swahili. This suggests that latent representation fusion with FLARE works best for text generation when the target languages have a similar word order to the source language, in this case, subject-verb-object. However, we also observe substantial performance gains for the Llama model on Telugu, which has a different word order than English, indicating that FLARE can still achieve considerable improvements even when the word order differs. 

<table><tr><td>Model</td><td>XNLI</td><td>TyDiQA</td><td>NusaX</td><td>Avg.</td><td>Avg. rank</td></tr><tr><td colspan="6">Zero-Shot Cross-Lingual Transfer (models are trained on English data)</td></tr><tr><td>XLM-R Large</td><td>76.95 ± 0.3</td><td>36.31 ± 2.3</td><td>75.26 ± 1.0</td><td>62.84</td><td>2.33</td></tr><tr><td>mT5-XL</td><td>77.92 ± 1.2</td><td>45.90 ± 0.2</td><td>74.72 ± 1.6</td><td>66.18</td><td>1.67</td></tr><tr><td>Llama 3.1 8B</td><td>77.40 ± 0.2</td><td>2.36 ± 0.2</td><td>71.74 ± 2.8</td><td>50.50</td><td>3.33</td></tr><tr><td>Gemma 2 9B</td><td>80.47 ± 0.1</td><td>2.46 ± 0.2</td><td>71.61 ± 3.4</td><td>51.51</td><td>2.67</td></tr><tr><td colspan="6">Translate-Test (test data is translated to English)</td></tr><tr><td>XLM-R Large</td><td>77.13 ± 0.2</td><td>41.06 ± 1.6</td><td>74.85 ± 1.0</td><td>64.35</td><td>2.67</td></tr><tr><td>mT5-XL</td><td>79.03 ± 0.2</td><td>47.92 ± 0.2</td><td>75.77 ± 0.3</td><td>67.57</td><td>1.67</td></tr><tr><td>Llama 3.1 8B</td><td>79.08 ± 0.5</td><td>2.53 ± 0.4</td><td>72.67 ± 2.4</td><td>51.54</td><td>2.67</td></tr><tr><td>Gemma 2 9B</td><td>79.99 ± 0.9</td><td>2.28 ± 0.2</td><td>71.61 ± 3.4</td><td>51.29</td><td>3.00</td></tr><tr><td colspan="6">Translate-Train (models are trained on training data translated to the target language)</td></tr><tr><td>XLM-R Large w/ LoRA</td><td>80.49 ± 1.3</td><td>40.14 ± 0.4</td><td>77.00 ± 0.8</td><td>65.88</td><td>3.33</td></tr><tr><td>w/ X-Mixup (Yang et al., 2022)</td><td>79.47 ± 0.2</td><td>38.24 ± 3.2</td><td>76.37 ± 2.8</td><td>64.69</td><td>4.67</td></tr><tr><td>w/ input-level fusion</td><td>77.24 ± 0.8</td><td>40.45 ± 0.5</td><td>78.53 ± 0.3</td><td>65.41</td><td>3.00</td></tr><tr><td>w/ FLARE MT</td><td>81.60 ± 0.3</td><td>38.88 ± 1.3</td><td>77.18 ± 0.2</td><td>65.89</td><td>2.67</td></tr><tr><td>w/ FLARE</td><td>80.99 ± 0.9</td><td>40.93 ± 0.2</td><td>79.18 ± 1.4</td><td>67.03</td><td>1.33</td></tr><tr><td>mT5-XL w/ LoRA</td><td>79.79 ± 2.1</td><td>46.76 ± 0.7</td><td>80.41 ± 0.2</td><td>68.99</td><td>3.67</td></tr><tr><td>w/ X-Mixup (Yang et al., 2022)</td><td>79.63 ± 1.0</td><td>48.23 ± 0.5</td><td>78.61 ± 0.2</td><td>68.82</td><td>4.00</td></tr><tr><td>w/ input-level fusion</td><td>78.81 ± 0.2</td><td>47.58 ± 0.2</td><td>80.12 ± 0.2</td><td>68.84</td><td>4.33</td></tr><tr><td>w/ FLARE MT</td><td>80.80 ± 1.4</td><td>48.48 ± 0.2</td><td>81.37 ± 0.8</td><td>70.22</td><td>1.67</td></tr><tr><td>w/ FLARE</td><td>81.00 ± 1.2</td><td>49.34 ± 0.3</td><td>80.54 ± 0.2</td><td>70.29</td><td>1.33</td></tr><tr><td>Llama 3.1 8B w/ LoRA</td><td>80.74 ± 0.4</td><td>42.84 ± 0.7</td><td>74.76 ± 1.4</td><td>66.11</td><td>3.00</td></tr><tr><td>w/ X-Mixup (Yang et al., 2022)</td><td>80.22 ± 0.2</td><td>17.47 ± 1.6</td><td>75.91 ± 0.7</td><td>57.87</td><td>4.00</td></tr><tr><td>w/ input-level fusion</td><td>80.70 ± 0.5</td><td>46.09 ± 0.9</td><td>74.60 ± 1.6</td><td>67.13</td><td>3.33</td></tr><tr><td>w/ FLARE MT</td><td>80.83 ± 0.2</td><td>38.95 ± 0.2</td><td>74.52 ± 1.6</td><td>64.77</td><td>3.67</td></tr><tr><td>w/ FLARE</td><td>80.92 ± 0.2</td><td>47.74 ± 1.2</td><td>76.08 ± 1.1</td><td>68.25</td><td>1.00</td></tr><tr><td>Gemma 2 9B w/ LoRA</td><td>84.89 ± 0.4</td><td>49.93 ± 0.7</td><td>79.37 ± 1.2</td><td>71.40</td><td>2.67</td></tr><tr><td>w/ X-Mixup (Yang et al., 2022)</td><td>84.62 ± 0.5</td><td>35.45 ± 2.0</td><td>79.94 ± 1.2</td><td>66.67</td><td>3.67</td></tr><tr><td>w/ input-level fusion</td><td>80.53 ± 0.2</td><td>51.29 ± 0.3</td><td>77.98 ± 1.1</td><td>69.93</td><td>4.00</td></tr><tr><td>w/ FLARE MT</td><td>84.84 ± 0.3</td><td>49.63 ± 0.9</td><td>78.09 ± 0.9</td><td>70.85</td><td>3.67</td></tr><tr><td>w/ FLARE</td><td>85.01 ± 0.4</td><td>52.14 ± 0.7</td><td>80.86 ± 0.5</td><td>72.67</td><td>1.00</td></tr></table>


Table 1: Average performance (with standard deviation) on natural language understanding datasets, computed over 5 random seeds. Metrics used are: Accuracy for XNLI, Exact Match for TyDiQA, and Macro F1 for NusaX. The best-performing results for each XLT model are highlighted in bold.


training time on TyDiQA by more than 40% when compared to input-level fusion. 

The results on XNLI and NusaX do not exhibit a clear correlation between performance benefits for languages with subject-verb-object word order. Furthermore, the results on NusaX demonstrate that FLARE can consistently provide performance improvements for lower-resourced languages, even when only a few training data is available. This highlights the potential of FLARE to support language adaptation in low-resource settings, where data is scarce. Compared to all benchmarked models, FLARE provides consistent performance benefits, demonstrating its effectiveness in transferring knowledge from English to other languages, and its potential to improve the performance of downstream tasks in low-resource languages. Beyond performance benefits, FLARE reduces the average 

Impact of Translation Quality. Figure 4 presents averaged performance results for XLM-R Large with FLARE on TyDiQA and NusaX, comparing the use of different-sized machine translation (MT) models, specifically NLLB 3.3B and NLLB 600M. The results demonstrate that FLARE is robust to lower-quality machine translations. Although utilizing the larger NLLB 3.3B model yields performance improvements of 1.27% and 1.77% on NusaX and TyDiQA, respectively, the FLARE models trained on lower-quality machine translations still achieve competitive performance with standard LoRA fine-tuning based on higher-quality machine translations. This demonstrates how FLARE can further enhance resource efficiency by effectively leveraging smaller MT models, thereby reducing computational requirements without compromising performance relative to its benchmarks. 

On Latent MT Fusion. For encoder-only models like XLM-R Large and encoder-decoder models like mT5, latent MT fusion provides notable performance benefits compared to standard LoRA finetuning, X-Mixup, and input-level fusion, as shown in Table 2. However, for decoder-only models like Llama and Gemma, we do not observe substantial performance benefits from latent MT fusion. This suggests that reducing the computational resources for processing the source language representations in regular FLARE can negatively impact cross-lingual transfer performance, particularly for larger models. Nonetheless, it provides a resourceefficient alternative to regular FLARE for smaller mPLMs by avoiding the need for decoding in the MT and eliminating the forward pass for the source language representations. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/96539746-f7a6-4f58-bc71-aca9b8949d34/a7daa650e1d87b00f942521f3614a57d30fe8a7cdf7cc9522b990f0d8a4ba441.jpg)



Figure 4: Average performance differences on NusaX and TyDiQA for XLM-R Large using FLARE with MT models of different size.


<table><tr><td>Fusion Function</td><td>TyDiQA</td><td>NusaX</td></tr><tr><td colspan="3">Translate-Train (models are trained on data translated to the target language)</td></tr><tr><td>add</td><td>40.76</td><td>79.56</td></tr><tr><td>mul</td><td>40.44</td><td>78.81</td></tr><tr><td>add+relu</td><td>40.93</td><td>79.18</td></tr><tr><td>cross-attention</td><td>39.63</td><td>78.11</td></tr></table>


Table 2: Average performance of fusion functions using XLM-R Large with FLARE, evaluated on TyDiQA with Exact Match and on NusaX with Macro F1.


Impact of Fusion Function. Our study on the impact of fusion functions, presented in Table 2, shows that adding non-linearity to the fusion functions does not necessarily provide decisive performance benefits over simpler linear transformations. Notably, the functions add and add+relu demonstrate the best performance. Despite the additional parameters available in cross-attention, this technique does not yield superior downstream performance, consistent with the low performance of X-Mixup in Table 1. These findings suggest that the optimal fusion function is task-dependent and can be regarded as a hyperparameter that can be finetuned based on validation data. 

<table><tr><td>Model</td><td>r</td><td>TyDiQA</td><td>NusaX</td></tr><tr><td colspan="4">Translate-Train (models are trained on training data translated to the target language)</td></tr><tr><td rowspan="2">XLM-R Large w/ FLARE MTw/ FLARE</td><td rowspan="2">8</td><td>40.86</td><td>77.84</td></tr><tr><td>42.37</td><td>79.52</td></tr><tr><td rowspan="2">XLM-R Large w/ FLARE MTw/ FLARE</td><td rowspan="2">64</td><td>38.88</td><td>77.18</td></tr><tr><td>40.93</td><td>79.18</td></tr><tr><td rowspan="2">XLM-R Large w/ FLARE MTw/ FLARE</td><td rowspan="2">128</td><td>40.21</td><td>77.18</td></tr><tr><td>40.88</td><td>78.32</td></tr></table>


Table 3: Average performance for varying adapter bottleneck size r in LoRA; based on XLM-R Large, using FLARE. Evaluation metrics include Exact Match for Ty-DiQA and Macro F1 for NusaX.


Impact of Adapter Capacity. We investigate the impact of the adapter capacity on FLARE’s performance, with results presented in Table 3. The findings indicate that small bottleneck sizes (r = 8) yield optimal performance for XLM-R Large on the TyDiQA and NusaX datasets. This observation is consistent with the findings in the original LoRA paper (Hu et al., 2022), indicating that the introduction of our fusion adapter does not affect the intrinsic rank of the tasks. 

Layer-wise Language Activation. Figure 6 shows that the magnitudes of source and target language activations across the entire XLM-R Large are comparable. This indicates that FLARE does not overly rely on either source or target representations during fusion, but instead integrates both sources of information in a balanced manner. Further, Figure 5 displays the average activations for English and Acehnese in the first adapter bottleneck: this confirms that both source and target languages maintain similar activation magnitudes. Hence, subsequent Acehnese representations are infused with the English representations from this initial transfer, integrating balanced source and target language information. Detailed activations for individual instances are illustrated in Figure 7, which show positional activation differences and demonstrate the alignment of source and target languages for information transfer. 

## 6 Conclusion

In this paper, we introduced Fusion for Language Representations (FLARE), a parameter-efficient method for cross-lingual transfer (XLT) that enhances representation quality and downstream performance for languages other than English. Our experimental results demonstrate that FLARE consistently outperforms strong XLT baselines, including target language fine-tuning with LoRA adapters, 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/96539746-f7a6-4f58-bc71-aca9b8949d34/5715c2cad7d891f83246bd143fdb9b68d9d7643113a9c1f1ccfdc24e38c81365.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/96539746-f7a6-4f58-bc71-aca9b8949d34/785bb3c2f61237ebdb1d04db749d23dcabf6feb718147f83104c3e56a0c591b4.jpg)



Figure 5: Average activation values for English and Acehnese in the first bottleneck query layer in XLM-R Large for the NusaX test set; add+relu fusion.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/96539746-f7a6-4f58-bc71-aca9b8949d34/1d517af4e298ad5ba1fe9af0063aefd19c1f54861bc51ce90a60de94ffd5cd66.jpg)



Figure 6: Average activations in the adapters across all XLM-R Large layers for the NusaX test set.


X-Mixup, and input-level fusion, on various natural language understanding tasks. FLARE demonstrates robust performance, even for lower-quality machine translations. A key takeaway is that FLARE remains more parameter-efficient compared to benchmarked baseline approaches, while yielding superior performance. Furthermore, FLARE provides most substantial performance benefits for multilingual question answering with decoder-only language models such as those from Llama and Gemma model families. 

## 7 Acknowledgments

The resources and services used in this work were provided by the VSC (Flemish Supercomputer Center), funded by the Research Foundation - Flanders (FWO) and the Flemish Government. The work of Ivan Vulic has been supported by a personal´ Royal Society University Research Fellowship ‘Inclusive and Sustainable Language Technologyfor a Truly Multilingual World’ (no 221137). Marie-Francine Moens was supported by the AIDAVA project (Horizon Europe: EU HORIZON-HLTH-2021-TOOL-06-03). 

## 8 Limitations

Our work demonstrates that highly compressed English language representations can be effectively transferred to other languages within adapter bottlenecks. However, our experiments focus on bilingual transfer settings. Extending fusion adapters to integrate multiple target languages is non-trivial, as it requires adapters to extract language-agnostic information across multiple languages. 

The proposed FLARE method by design relies data availability for both source and target languages. Consequently, the application of FLARE is dependent upon the availability of machine translation models. Furthermore, our evaluation exclusively employs English as the high-resource source language for representation fusion. While English is predominantly used in mPLM pretraining corpora, exploring other high-resource languages that share linguistic similarities, with the target languages could potentially yield similar or improved cross-lingual transfer performance. 

The reliance on machine translation also pose challenges for extremely low-resource languages, particularly where translation quality suffers or where cultural nuances are difficult to preserve (Alemayehu et al., 2024; Conia et al., 2024). To examine the impact of translation quality on FLARE, we conducted a manual error analysis on 200 randomly selected TyDiQA examples using Llama 3.1 and Gemma 2. We found no cases where machine translation quality was responsible for model errors. Most errors were due to answers unrelated to the question, while a smaller subset consisted of correct answers that did not match the dataset’s ground truth. Additionally, the flexibility of FLARE’s fusion functions provides opportunities to incorporate cultural context more explicitly in future work. 

We opt for the original and established LoRA architecture as our go-to PEFT model, due to its wide adoption and popularity. However, we note that FLARE is not tied to the LoRA architecture, and might be combined with other, more recent and more sophisticated PEFT architectures (Liu et al., 2024; Kopiczko et al., 2024) in future research. 

Finally, our choice of base multilingual LMs has been motivated by the current state-of-the-art (SotA) in the field of multilingual NLP and XLT to low-resource languages for NLU tasks, combined with our computational budget and constraints. Therefore, the main models are SotA encoder-only (XLM-R) and encoder-decoder mPLMs (mT5), and decoder-only LLMs (Llama 3, Gemma 2). However, we note that the LLM technology and its adaptation to XLT for NLU in lower-resource languages has not been proven to be fully mature yet (Lin et al., 2024; Razumovskaia et al., 2024). 

## References



Hizkiel Mitiku Alemayehu, Hamada M Zahera, and Axel-Cyrille Ngonga Ngomo. 2024. Error analysis of multilingual language models in machine translation: A case study of English-Amharic translation. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 19758–19768, Miami, Florida, USA. Association for Computational Linguistics. 





Alan Ansell, Marinela Parovic, Ivan Vuli´ c, Anna Ko-´ rhonen, and Edoardo Ponti. 2023. Unifying crosslingual transfer across scenarios of resource scarcity. In Proceedings of the 2023 Conference on Empiri cal Methods in Natural Language Processing, pages 3980–3995, Singapore. Association for Computational Linguistics. 





Alan Ansell, Edoardo Ponti, Anna Korhonen, and Ivan Vulic. 2022.´ Composable sparse fine-tuning for crosslingual transfer. In Proceedings ofthe 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1778–1796, Dublin, Ireland. Association for Computational Linguistics. 





Mikel Artetxe, Vedanuj Goswami, Shruti Bhosale, Angela Fan, and Luke Zettlemoyer. 2023. Revisiting machine translation for cross-lingual classification. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 6489–6499, Singapore. Association for Computational Linguistics. 





Mikel Artetxe, Gorka Labaka, and Eneko Agirre. 2020. Translation artifacts in cross-lingual transfer learning. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 7674–7684, Online. Association for Computational Linguistics. 





Laurie Burchell, Alexandra Birch, Nikolay Bogoychev, and Kenneth Heafield. 2023. An open dataset and model for language identification. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 865–879, Toronto, Canada. Association for Computational Linguistics. 





Tingfeng Cao, Chengyu Wang, Chuanqi Tan, Jun Huang, and Jinhui Zhu. 2023. Sharing, teaching and aligning: Knowledgeable transfer learning for cross-lingual machine reading comprehension. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2023, pages 455–467, Singapore. Association for Computational Linguistics. 





Yang Chen, Chao Jiang, Alan Ritter, and Wei Xu. 2023. Frustratingly easy label projection for cross-lingual transfer. In Findings ofthe Associationfor Computational Linguistics: ACL 2023, pages 5775–5796, 





Toronto, Canada. Association for Computational Linguistics. 





Jonathan H. Clark, Eunsol Choi, Michael Collins, Dan Garrette, Tom Kwiatkowski, Vitaly Nikolaev, and Jennimaria Palomaki. 2020. TyDi QA: A benchmark for information-seeking question answering in typologically diverse languages. Transactions ofthe Associationfor Computational Linguistics, 8:454–470. 





Simone Conia, Daniel Lee, Min Li, Umar Farooq Minhas, Saloni Potdar, and Yunyao Li. 2024. Towards cross-cultural machine translation with retrievalaugmented generation from multilingual knowledge graphs. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 16343–16360, Miami, Florida, USA. Association for Computational Linguistics. 





Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 2020. Unsupervised cross-lingual representation learning at scale. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 8440– 8451, Online. Association for Computational Linguistics. 





Alexis Conneau, Ruty Rinott, Guillaume Lample, Adina Williams, Samuel Bowman, Holger Schwenk, and Veselin Stoyanov. 2018. XNLI: Evaluating crosslingual sentence representations. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2475–2485, Brussels, Belgium. Association for Computational Linguistics. 





Emilio Cueva, Adrian Lopez Monroy, Fernando Sánchez-Vega, and Thamar Solorio. 2024. Adaptive cross-lingual text classification through in-context one-shot demonstrations. In Proceedings ofthe 2024 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 8317–8335, Mexico City, Mexico. Association for Computational Linguistics. 





Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. 2023. QLoRA: Efficient finetuning of quantized LLMs. In Thirty-seventh Conference on Neural Information Processing Systems. 





Benedikt Ebing and Goran Glavaš. 2024. To translate or not to translate: A systematic investigation of translation-based cross-lingual transfer to lowresource languages. In Proceedings of the 2024 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 5325–5344, Mexico City, Mexico. Association for Computational Linguistics. 





Yuwei Fang, Shuohang Wang, Zhe Gan, Siqi Sun, and Jingjing Liu. 2021. Filter: An enhanced fusion 



method for cross-lingual language understanding. Proceedings of the AAAI Conference on Artificial Intelligence, 35(14):12776–12784. 

Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupati raju, Léonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ramé, Johan Ferret, Peter Liu, Pouya Tafti, Abe Friesen, Michelle Casbon, Sabela Ramos, Ravin Kumar, Charline Le Lan, Sammy Jerome, Anton Tsitsulin, Nino Vieillard, Piotr Stanczyk, Sertan Girgin, Nikola Momchev, Matt Hoffman, Shantanu Thakoor, Jean-Bastien Grill, Behnam Neyshabur, Olivier Bachem, Alanna Wal ton, Aliaksei Severyn, Alicia Parrish, Aliya Ah mad, Allen Hutchison, Alvin Abdagic, Amanda Carl, Amy Shen, Andy Brock, Andy Coenen, An thony Laforge, Antonia Paterson, Ben Bastian, Bila Piot, Bo Wu, Brandon Royal, Charlie Chen, Chintu Kumar, Chris Perry, Chris Welty, Christopher A Choquette-Choo, Danila Sinopalnikov, David Wein berger, Dimple Vijaykumar, Dominika Rogozinska,´ Dustin Herbison, Elisa Bandy, Emma Wang, Eric Noland, Erica Moreira, Evan Senter, Evgenii Elty shev, Francesco Visin, Gabriel Rasskin, Gary Wei, Glenn Cameron, Gus Martins, Hadi Hashemi, Hanna Klimczak-Plucinska, Harleen Batra, Harsh Dhand,´ Ivan Nardini, Jacinda Mein, Jack Zhou, James Svens son, Jeff Stanway, Jetha Chan, Jin Peng Zhou, Joana Carrasqueira, Joana Iljazi, Jocelyn Becker, Joe Fer nandez, Joost van Amersfoort, Josh Gordon, Josh Lipschultz, Josh Newlan, Ju yeong Ji, Kareem Mo hamed, Kartikeya Badola, Kat Black, Katie Mil lican, Keelin McDonell, Kelvin Nguyen, Kiranbir Sodhia, Kish Greene, Lars Lowe Sjoesund, Lau ren Usui, Laurent Sifre, Lena Heuermann, Leti cia Lago, Lilly McNealus, Livio Baldini Soares, Logan Kilpatrick, Lucas Dixon, Luciano Martins, Machel Reid, Manvinder Singh, Mark Iverson, Mar tin Görner, Mat Velloso, Mateo Wirth, Matt Davi dow, Matt Miller, Matthew Rahtz, Matthew Watson, Meg Risdal, Mehran Kazemi, Michael Moynihan, Ming Zhang, Minsuk Kahng, Minwoo Park, Mofi Rahman, Mohit Khatwani, Natalie Dao, Nenshad Bardoliwalla, Nesh Devanathan, Neta Dumai, Nilay Chauhan, Oscar Wahltinez, Pankil Botarda, Parker Barnes, Paul Barham, Paul Michel, Pengchong Jin, Petko Georgiev, Phil Culliton, Pradeep Kup pala, Ramona Comanescu, Ramona Merhej, Reena Jana, Reza Ardeshir Rokni, Rishabh Agarwal, Ryan Mullins, Samaneh Saadat, Sara Mc Carthy, Sarah Cogan, Sarah Perrin, Sébastien M. R. Arnold, Se bastian Krause, Shengyang Dai, Shruti Garg, Shruti Sheth, Sue Ronstrom, Susan Chan, Timothy Jordan, Ting Yu, Tom Eccles, Tom Hennigan, Tomas Kocisky, Tulsee Doshi, Vihan Jain, Vikas Yadav, Vilobh Meshram, Vishal Dharmadhikari, Warren Barkley, Wei Wei, Wenming Ye, Woohyun Han, Woosuk Kwon, Xiang Xu, Zhe Shen, Zhitao Gong, Zichuan Wei, Victor Cotruta, Phoebe Kirk, Anand Rao, Minh Giang, Ludovic Peran, Tris Warkentin Eli Collins, Joelle Barral, Zoubin Ghahramani, Raia Hadsell, D. Sculley, Jeanine Banks, Anca Dragan, Slav Petrov, Oriol Vinyals, Jeff Dean, Demis Hass abis, Koray Kavukcuoglu, Clement Farabet, Elena Buchatskaya, Sebastian Borgeaud, Noah Fiedel, Armand Joulin, Kathleen Kenealy, Robert Dadashi, and Alek Andreev. 2024. Gemma 2: Improving open language models at a practical size. Preprint, arXiv:2408.00118. 

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al Dahle, Aiesha Letman, Akhil Mathur, Alan Schel ten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mi tra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurelien Ro driguez, Austen Gregerson, Ava Spataru, Baptiste Roziere, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, Danny Wyatt, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Francisco Guzmán, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Govind Thattai, Graeme Nail, Gregoire Mi alon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel Kloumann, Ishan Misra, Ivan Evtimov, Jack Zhang, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Karthik Prasad, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, Khalid El-Arini, Krithika Iyer, Kshitiz Malik, Kuenley Chiu, Kunal Bhalla, Kusha Lakhotia, Lauren Rantala-Yeary, Laurens van der Maaten, Lawrence Chen, Liang Tan, Liz Jenkins, Louis Martin, Lovish Madaan, Lubo Malo, Lukas Blecher, Lukas Landzaat, Luke de Oliveira, Madeline Muzzi, Mahesh Pasupuleti, Mannat Singh, Manohar Paluri, Marcin Kardas, Maria Tsimpoukelli, Mathew Oldham, Mathieu Rita, Maya Pavlova, Melanie Kam badur, Mike Lewis, Min Si, Mitesh Kumar Singh Mona Hassan, Naman Goyal, Narjes Torabi, Niko lay Bashlykov, Nikolay Bogoychev, Niladri Chatterji, Ning Zhang, Olivier Duchenne, Onur Çelebi, Patrick Alrassy, Pengchuan Zhang, Pengwei Li, Petar Va sic, Peter Weng, Prajjwal Bhargava, Pratik Dubal, Praveen Krishnan, Punit Singh Koura, Puxin Xu, Qing He, Qingxiao Dong, Ragavan Srinivasan, Raj Ganapathy, Ramon Calderer, Ricardo Silveira Cabral, Robert Stojnic, Roberta Raileanu, Rohan Maheswari, Rohit Girdhar, Rohit Patel, Romain Sauvestre, Ron nie Polidoro, Roshan Sumbaly, Ross Taylor, Ruan Silva, Rui Hou, Rui Wang, Saghar Hosseini, Sa hana Chennabasappa, Sanjay Singh, Sean Bell, Seo hyun Sonia Kim, Sergey Edunov, Shaoliang Nie, Sha-

ran Narang, Sharath Raparthy, Sheng Shen, Shengye Wan, Shruti Bhosale, Shun Zhang, Simon Van denhende, Soumya Batra, Spencer Whitman, Sten Sootla, Stephane Collot, Suchin Gururangan, Syd ney Borodinsky, Tamar Herman, Tara Fowler, Tarek Sheasha, Thomas Georgiou, Thomas Scialom, Tobias Speckbacher, Todor Mihaylov, Tong Xiao, Ujjwal Karn, Vedanuj Goswami, Vibhor Gupta, Vignesh Ramanathan, Viktor Kerkez, Vincent Gonguet, Vir ginie Do, Vish Vogeti, Vítor Albiero, Vladan Petro vic, Weiwei Chu, Wenhan Xiong, Wenyin Fu, Whit ney Meers, Xavier Martinet, Xiaodong Wang, Xi aofang Wang, Xiaoqing Ellen Tan, Xide Xia, Xin feng Xie, Xuchao Jia, Xuewei Wang, Yaelle Gold schlag, Yashesh Gaur, Yasmine Babaei, Yi Wen, Yiwen Song, Yuchen Zhang, Yue Li, Yuning Mao, Zacharie Delpierre Coudert, Zheng Yan, Zhengxing Chen, Zoe Papakipos, Aaditya Singh, Aayushi Sri vastava, Abha Jain, Adam Kelsey, Adam Shajnfeld, Adithya Gangidi, Adolfo Victoria, Ahuva Goldstand, Ajay Menon, Ajay Sharma, Alex Boesenberg, Alexei Baevski, Allie Feinstein, Amanda Kallet, Amit San gani, Amos Teo, Anam Yunus, Andrei Lupu, An dres Alvarado, Andrew Caples, Andrew Gu, Andrew Ho, Andrew Poulton, Andrew Ryan, Ankit Ramchan dani, Annie Dong, Annie Franco, Anuj Goyal, Apara jita Saraf, Arkabandhu Chowdhury, Ashley Gabriel Ashwin Bharambe, Assaf Eisenman, Azadeh Yaz dan, Beau James, Ben Maurer, Benjamin Leonhardi, Bernie Huang, Beth Loyd, Beto De Paola, Bhargavi Paranjape, Bing Liu, Bo Wu, Boyu Ni, Braden Han cock, Bram Wasti, Brandon Spence, Brani Stojkovic, Brian Gamido, Britt Montalvo, Carl Parker, Carly Burton, Catalina Mejia, Ce Liu, Changhan Wang, Changkyu Kim, Chao Zhou, Chester Hu, Ching Hsiang Chu, Chris Cai, Chris Tindal, Christoph Fe ichtenhofer, Cynthia Gao, Damon Civin, Dana Beaty, Daniel Kreymer, Daniel Li, David Adkins, David Xu, Davide Testuggine, Delia David, Devi Parikh Diana Liskovich, Didem Foss, Dingkang Wang, Duc Le, Dustin Holland, Edward Dowling, Eissa Jamil, Elaine Montgomery, Eleonora Presani, Emily Hahn, Emily Wood, Eric-Tuan Le, Erik Brinkman, Este ban Arcaute, Evan Dunbar, Evan Smothers, Fei Sun, Felix Kreuk, Feng Tian, Filippos Kokkinos, Firat Ozgenel, Francesco Caggioni, Frank Kanayet, Frank Seide, Gabriela Medina Florez, Gabriella Schwarz, Gada Badeer, Georgia Swee, Gil Halpern, Grant Herman, Grigory Sizov, Guangyi, Zhang, Guna Lakshminarayanan, Hakan Inan, Hamid Shojanaz eri, Han Zou, Hannah Wang, Hanwen Zha, Haroun Habeeb, Harrison Rudolph, Helen Suk, Henry As pegren, Hunter Goldman, Hongyuan Zhan, Ibrahim Damlaj, Igor Molybog, Igor Tufanov, Ilias Leontiadis, Irina-Elena Veliche, Itai Gat, Jake Weissman, James Geboski, James Kohli, Janice Lam, Japhet Asher, Jean-Baptiste Gaya, Jeff Marcus, Jeff Tang, Jen nifer Chan, Jenny Zhen, Jeremy Reizenstein, Jerem Teboul, Jessica Zhong, Jian Jin, Jingyi Yang, Joe Cummings, Jon Carvill, Jon Shepard, Jonathan Mc Phie, Jonathan Torres, Josh Ginsburg, Junjie Wang, Kai Wu, Kam Hou U, Karan Saxena, Kartikay Khan delwal, Katayoun Zand, Kathy Matosich, Kaushik Veeraraghavan, Kelly Michelena, Keqian Li, Ki ran Jagadeesh, Kun Huang, Kunal Chawla, Kyle Huang, Lailin Chen, Lakshya Garg, Lavender A, Leandro Silva, Lee Bell, Lei Zhang, Liangpeng Guo, Licheng Yu, Liron Moshkovich, Luca Wehrst edt, Madian Khabsa, Manav Avalani, Manish Bhatt, Martynas Mankus, Matan Hasson, Matthew Lennie, Matthias Reso, Maxim Groshev, Maxim Naumov, Maya Lathi, Meghan Keneally, Miao Liu, Michael L. Seltzer, Michal Valko, Michelle Restrepo, Mihir Pa tel, Mik Vyatskov, Mikayel Samvelyan, Mike Clark Mike Macey, Mike Wang, Miquel Jubert Hermoso, Mo Metanat, Mohammad Rastegari, Munish Bansal, Nandhini Santhanam, Natascha Parks, Natasha White, Navyata Bawa, Nayan Singhal, Nick Egebo Nicolas Usunier, Nikhil Mehta, Nikolay Pavlovich Laptev, Ning Dong, Norman Cheng, Oleg Chernoguz, Olivia Hart, Omkar Salpekar, Ozlem Kalinli, Parkin Kent, Parth Parekh, Paul Saab, Pavan Balaji, Pe dro Rittner, Philip Bontrager, Pierre Roux, Piotr Dollar, Polina Zvyagina, Prashant Ratanchandani, Pritish Yuvraj, Qian Liang, Rachad Alao, Rache Rodriguez, Rafi Ayub, Raghotham Murthy, Raghu Nayani, Rahul Mitra, Rangaprabhu Parthasarathy, Raymond Li, Rebekkah Hogan, Robin Battey, Rocky Wang, Russ Howes, Ruty Rinott, Sachin Mehta, Sachin Siby, Sai Jayesh Bondu, Samyak Datta, Sara Chugh, Sara Hunt, Sargun Dhillon, Sasha Sidorov, Satadru Pan, Saurabh Mahajan, Saurabh Verma, Seiji Yamamoto, Sharadh Ramaswamy, Shaun Lind say, Shaun Lindsay, Sheng Feng, Shenghao Lin, Shengxin Cindy Zha, Shishir Patil, Shiva Shankar, Shuqiang Zhang, Shuqiang Zhang, Sinong Wang, Sneha Agarwal, Soji Sajuyigbe, Soumith Chintala, Stephanie Max, Stephen Chen, Steve Kehoe, Steve Satterfield, Sudarshan Govindaprasad, Sumit Gupta, Summer Deng, Sungmin Cho, Sunny Virk, Suraj Subramanian, Sy Choudhury, Sydney Goldman, Ta Remez, Tamar Glaser, Tamara Best, Thilo Koehler, Thomas Robinson, Tianhe Li, Tianjun Zhang, Tim Matthews, Timothy Chou, Tzook Shaked, Varun Vontimitta, Victoria Ajayi, Victoria Montanez, Vija Mohan, Vinay Satish Kumar, Vishal Mangla, Vlad Ionescu, Vlad Poenaru, Vlad Tiberiu Mihailescu, Vladimir Ivanov, Wei Li, Wenchen Wang, Wenwen Jiang, Wes Bouaziz, Will Constable, Xiaocheng Tang, Xiaojian Wu, Xiaolan Wang, Xilun Wu, Xinbo Gao, Yaniv Kleinman, Yanjun Chen, Ye Hu, Ye Jia, Ye Qi, Yenda Li, Yilin Zhang, Ying Zhang, Yossi Adi, Youngjin Nam, Yu, Wang, Yu Zhao, Yuchen Hao, Yundi Qian, Yunlu Li, Yuzi He, Zach Rait, Zachary DeVito, Zef Rosnbrick, Zhaoduo Wen, Zhenyu Yang, Zhiwei Zhao, and Zhiyu Ma. 2024. The llama 3 herd of models. Preprint, arXiv:2407.21783. 



Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. 2019. Parameter-efficient transfer learning for NLP. In Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pages 2790–2799. PMLR. 





Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan 





Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations. 





Sunkyoung Kim, Dayeon Ki, Yireun Kim, and Jinsik Lee. 2024. Cross-lingual QA: A key to unlocking in-context cross-lingual performance. In ICML 2024 Workshop on In-Context Learning. 





Dawid Jan Kopiczko, Tijmen Blankevoort, and Yuki M. Asano. 2024. VeRA: Vector-based random matrix adaptation. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. 





Anne Lauscher, Vinit Ravishankar, Ivan Vulic, and´ Goran Glavaš. 2020. From zero to hero: On the limitations of zero-shot language transfer with multilingual Transformers. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 4483–4499, Online. Association for Computational Linguistics. 





En-Shiun Annie Lee, Sarubi Thillainathan, Shravan Nayak, Surangika Ranathunga, David Ifeoluwa Adelani, Ruisi Su, and Arya D. McCarthy. 2022a. Pretrained multilingual sequence-to-sequence models: A hope for low-resource language translation? In Findings ofthe Associationfor Computational Lin guistics: ACL 2022, pages 58–67, Dublin, Ireland. Association for Computational Linguistics. 





Jaeseong Lee, Seung-won Hwang, and Taesup Kim. 2022b. FAD-X: Fusing adapters for cross-lingual transfer to low-resource languages. In Proceedings of the 2nd Conference ofthe Asia-Pacific Chapter ofthe Association for Computational Linguistics and the 12th International Joint Conference on Natural Language Processing (Volume 2: Short Papers), pages 57–64, Online only. Association for Computational Linguistics. 





Peiqin Lin, Shaoxiong Ji, Jörg Tiedemann, André F. T. Martins, and Hinrich Schütze. 2024. MaLA-500: Massive language adaptation of large language models. Preprint, arXiv:2401.13303. 





Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. Visual instruction tuning. In Advances in Neural Information Processing Systems, volume 36, pages 34892–34916. Curran Associates, Inc. 





Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, Kwang-Ting Cheng, and Min-Hung Chen. 2024. Dora: Weightdecomposed low-rank adaptation. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net. 





NLLB Team, Marta R. Costa-jussà, James Cross, Onur Çelebi, Maha Elbayad, Kenneth Heafield, Kevin Hef fernan, Elahe Kalbassi, Janice Lam, Daniel Licht, Jean Maillard, Anna Sun, Skyler Wang, Guillaume 





Wenzek, Al Youngblood, Bapi Akula, Loic Barrault, Gabriel Mejia Gonzalez, Prangthip Hansanti, John Hoffman, Semarley Jarrett, Kaushik Ram Sadagopan, Dirk Rowe, Shannon Spruit, Chau Tran, Pierre Andrews, Necip Fazil Ayan, Shruti Bhosale, Sergey Edunov, Angela Fan, Cynthia Gao, Vedanuj Goswami, Francisco Guzmán, Philipp Koehn, Alexandre Mourachko, Christophe Ropers, Safiyyah Saleem, Holger Schwenk, and Jeff Wang. 2022. No language left behind: Scaling human-centered machine translation. Preprint, arXiv:2207.04672. 





Jaehoon Oh, Jongwoo Ko, and Se-Young Yun. 2022. Synergy with translation artifacts for training and inference in multilingual tasks. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 6747–6754, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics. 





Jonas Pfeiffer, Sebastian Ruder, Ivan Vulic, and Edoardo M. Ponti. 2023. Modular deep learning. Transactions on Machine Learning Research, 2023. 





Jonas Pfeiffer, Ivan Vulic, Iryna Gurevych, and Se-´ bastian Ruder. 2020. MAD-X: An Adapter-Based Framework for Multi-Task Cross-Lingual Transfer. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 7654–7673, Online. Association for Computational Linguistics. 





Ayu Purwarianti and Ida Ayu Putu Ari Crisdayanti. 2019. Improving bi-lstm performance for indonesian sentiment analysis using paragraph vector. In 2019 International Conference ofAdvanced Informatics: Concepts, Theory and Applications (ICAICTA), pages 1–5. 





Tingyu Qu, Tinne Tuytelaars, and Marie-Francine Moens. 2025. Introducing routing functions to visionlanguage parameter-efficient fine-tuning with lowrank bottlenecks. In Computer Vision – ECCV 2024, pages 291–308, Cham. Springer Nature Switzerland. 





Kiran Ramnath, Leda Sari, Mark Hasegawa-Johnson, and Chang Yoo. 2021. Worldly wise (WoW) - cross-lingual knowledge fusion for fact-based visual spoken-question answering. In Proceedings of the 2021 Conference ofthe North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies, pages 1908–1919, Online. Association for Computational Linguistics. 





Evgeniia Razumovskaia, Ivan Vulic, and Anna Korho-´ nen. 2024. Analyzing and adapting large language models for few-shot multilingual NLU: are we there yet? Preprint, arXiv:2403.01929. 





Sebastian Ruder, Ivan Vulic, and Anders Søgaard. 2019.´ A survey of cross-lingual word embedding models. J. Artif. Int. Res., 65(1):569–630. 





Fabian David Schmidt, Philipp Borchert, Ivan Vulic, and´ Goran Glavaš. 2024. Self-distillation for model stacking unlocks cross-lingual NLU in 200+ languages. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 6724–6743, Miami, Florida, USA. Association for Computational Linguistics. 





Lei Shi, Rada Mihalcea, and Mingjun Tian. 2010. Cross language text classification by model translation and semi-supervised learning. In Proceedings ofthe 2010 Conference on Empirical Methods in Natural Language Processing, pages 1057–1067, Cambridge, MA. Association for Computational Linguistics. 





Tianyi Tang, Wenyang Luo, Haoyang Huang, Dongdong Zhang, Xiaolei Wang, Xin Zhao, Furu Wei, and Ji-Rong Wen. 2024. Language-specific neurons: The key to multilingual capabilities in large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5701–5715, Bangkok, Thailand. Association for Computational Linguistics. 





Eshaan Tanwar, Subhabrata Dutta, Manish Borthakur, and Tanmoy Chakraborty. 2023. Multilingual LLMs are better cross-lingual in-context learners with alignment. In Proceedings ofthe 61st Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 6292–6307, Toronto, Canada. Association for Computational Linguistics. 





Yaqing Wang, Sahaj Agarwal, Subhabrata Mukherjee, Xiaodong Liu, Jing Gao, Ahmed Hassan Awadallah, and Jianfeng Gao. 2022. AdaMix: Mixtureof-adaptations for parameter-efficient model tuning. In Proceedings of the 2022 Conference on Empiri cal Methods in Natural Language Processing, pages 5744–5760, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics. 





Chris Wendler, Veniamin Veselovsky, Giovanni Monea, and Robert West. 2024. Do llamas work in English? on the latent language of multilingual transformers. In Proceedings of the 62nd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 15366–15394, Bangkok, Thailand. Association for Computational Linguistics. 





Adina Williams, Nikita Nangia, and Samuel Bowman. 2018. A broad-coverage challenge corpus for sentence understanding through inference. In Proceedings ofthe 2018 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 1112–1122, New Orleans, Louisiana. Association for Computational Linguistics. 





Genta Winata, Shijie Wu, Mayank Kulkarni, Thamar Solorio, and Daniel Preotiuc-Pietro. 2022. Crosslingual few-shot learning on unseen languages. In Proceedings ofthe 2nd Conference ofthe Asia-Pacific Chapter of the Association for Computational Linguistics and the 12th International Joint Conference 





on Natural Language Processing (Volume 1: Long Papers), pages 777–791, Online only. Association for Computational Linguistics. 





Genta Indra Winata, Alham Fikri Aji, Samuel Cahyawijaya, Rahmad Mahendra, Fajri Koto, Ade Romadhony, Kemal Kurniawan, David Moeljadi, Radityo Eko Prasojo, Pascale Fung, Timothy Baldwin, Jey Han Lau, Rico Sennrich, and Sebastian Ruder. 2023. NusaX: Multilingual parallel sentiment dataset for 10 Indonesian local languages. In Proceedings ofthe 17th Conference ofthe European Chapter of the Associationfor Computational Linguistics, pages 815–834, Dubrovnik, Croatia. Association for Computational Linguistics. 





Shijie Wu and Mark Dredze. 2020. Are all languages created equal in multilingual BERT? In Proceedings ofthe 5th Workshop on Representation Learningfor NLP, pages 120–130, Online. Association for Computational Linguistics. 





Shaoyang Xu, Junzhuo Li, and Deyi Xiong. 2023. Language representation projection: Can we transfer factual knowledge across languages in multilingual language models? In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 3692–3702, Singapore. Association for Computational Linguistics. 





Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, and Colin Raffel. 2021. mT5: A massively multilingual pre-trained text-to-text transformer. In Proceedings ofthe 2021 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, pages 483–498, Online. Association for Computational Linguistics. 





Huiyun Yang, Huadong Chen, Hao Zhou, and Lei Li. 2022. Enhancing cross-lingual transfer by manifold mixup. In International Conference on Learning Representations. 





Mengjie Zhao, Yi Zhu, Ehsan Shareghi, Ivan Vulic, Roi´ Reichart, Anna Korhonen, and Hinrich Schütze. 2021. A closer look at few-shot crosslingual transfer: The choice of shots matters. In Proceedings ofthe 59th Annual Meeting ofthe Associationfor Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 5751–5767, Online. Association for Computational Linguistics. 





Yiran Zhao, Wenxuan Zhang, Huiming Wang, Kenji Kawaguchi, and Lidong Bing. 2024. AdaMergeX: Cross-lingual transfer with large language models via adaptive adapter merging. Preprint, arXiv:2402.18913. 



## A Detailed Evaluation Results

Figure 7 displays average activations within the first adapter bottlenecks in the XLM-R Large model using FLARE and the add+relu fusion function. This visualization highlights the positional alignment process between English and Acehnese token representations, with varying activation values across different sequence positions reflecting the dynamics of language representation fusion. 

Table 4 presents the performance of FLARE and input-level fusion when using gold translations for fusion, as opposed to machine translations generated by NLLB. The results demonstrate that inputlevel fusion performance is sensitive to the quality of English input provided. Notably, when gold translations are available, input-level fusion replicates English performance, indicating that it heavily relies on the quality of English inputs. In contrast, FLARE balances the fusion of source and target language information, as evident from the findings in Figure 6. While input-level fusion outperforms FLARE when gold translations are available, FLARE achieves substantially higher performance in the more realistic setting using machine-translated data. 

Table 6 shows the results for the XNLI dataset for each language in zero-shot XLT, translatetest, translate-train settings, including translatetrain with gold translations in the source language. The results confirm that FLARE consistently improves XTL performance in the translate-train setting across different languages without particular bias towards typological relatedness to English or frequency in pretraining corpora. 

Table 7 details the results for the TyDiQA dataset for each language in the zero-shot XLT, translatetest, and translate-train settings. The outcomes demonstrate that FLARE performance extends to tasks including positional information, such as extractive question-answering. 

Table 9 reports p-values from the Pitman permutation tests comparing FLARE and FLARE MT to baseline methods, based on average per-language performance from Tables 6, 7, and 8. The results indicate that FLARE and FLARE MT yield consistent and often statistically significant improvements over regular LoRA, X-Mixup, and input-level fusion, with particularly strong gains for the decoder-only Llama and Gemma models. 

Table 8 outlines the performance for the NusaX dataset for each language in zero-shot XLT, translate-test, translate-train, and translate-train settings with gold translations in the source language. Even with few training samples, our FLARE method demonstrates consistent performance improvements across the low-resource languages included in the NusaX dataset. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/96539746-f7a6-4f58-bc71-aca9b8949d34/1046a2acd530a0a9361a7a05ef25fb7fd434a7edbb7c8bf98f69ae76b2f8d149.jpg)



Figure 7: Activation values for individual instances included in the NusaX test set. English and Acehnese activation values are extracted from the first bottleneck query layer in XLMR-Large, which is trained with the add+relu fusion function.


## B Training Details

Our evaluation results are averaged across five random seeds. Initially, we fine-tune the language models on English task data using LoRA adapters set with r = 64 and $\alpha = 1 2 8$ , which are subsequently integrated into the model’s weights prior to task fine-tuning in the target languages. Hyperparameter configurations for each mPLM are provided in Table 5. 

The total computation time for the experimental results exceeds 5,000 GPU hours. All models are trained using half-precision. 

## C Implementation Details of FLARE MT

We introduce FLARE MT as variant of FLARE aimed at further enhancing resource efficiency. FLARE MT improves efficiency in two key ways. Firstly, it leverages latent translations generated by the machine translation (MT) encoder, thereby reducing the computational resources required to produce full text translations. Secondly, it eliminates the need for a forward pass through the source language representations in the mPLM, resulting in significant computational savings. As a results, a single source language representation, namely the latent translation, is fused with the target language representation in the fusion adapters for each transformer layer. To enable this fusion, a projection module is introduced to align the dimensions of the MT encoder with those of the mPLM. Although this module adds additional parameters, it is essential for ensuring compatibility between the two models. Notably, related work suggests that extending the single projection layer to a MLP and training it on additional self-supervised data can yield substantial performance benefits (Liu et al., 2023; Schmidt et al., 2024). This provides a promising direction for future research and potential improvements to the FLARE MT approach. 

## D Practical Implications

The practical implementation of bilingual crosslingual transfer methods, such as FLARE, requires an additional step of language identification to determine bilingual adapter for model inference. While this introduces a preprocessing stage, language identification systems are widely accessible and highly accurate. For example, NLLB achieves a 95% F1 score across 193 FLORES languages, including many low-resource languages (Burchell et al., 2023), ensuring that this step can be seamlessly integrated into real-world applications. 

## E Another Ablation: Representation Fusion during Training Only

To investigate the importance of utilizing source language representations during inference, we modified FLARE to restrict representation fusion to the training phase only. Specifically, we limited the fusion with source language representations to 50% of the training instances and excluded source language data during inference. This evaluates cross-lingual transfer capabilities based on instance-independent patterns learned from source language representations during training. Our findings reveal that fusion adapters struggle to learn patterns that are independent of specific instances from source language representations during training. As a result, when implemented in the XLM-R Large model on the NusaX test set, the performance of the train-only FLARE variant decreased by 30%. Crucially, this drop in performance underscores the importance of incorporating source language representations during inference to achieve effective cross-lingual adaptation. 

<table><tr><td>Model</td><td>XNLI</td><td>NusaX</td></tr><tr><td colspan="3">Translate-Train (fusion models are trained on data translated into the target language and evaluated using gold translations from the target language to the source language)</td></tr><tr><td>XLM-R Largew/ input-level fusion</td><td>87.19</td><td>90.93</td></tr><tr><td>w/ FLARE</td><td>88.15</td><td>84.66</td></tr><tr><td>mT5-XLw/ input-level fusion</td><td>89.67</td><td>90.57</td></tr><tr><td>w/ FLARE</td><td>86.57</td><td>80.72</td></tr></table>


Table 4: Average performance for the translate-train setting with gold English translations during inference across languages included in the XNLI, and NusaX datasets, representing optimal translation quality. Evaluation metrics include accuracy for XNLI and Macro F1 for NusaX.


<table><tr><td>Model</td><td>Hparam</td><td>XNLI</td><td>TyDiQA</td><td>NusaX</td></tr><tr><td rowspan="4">XLM-R Large</td><td>epochs</td><td>10</td><td>10</td><td>20</td></tr><tr><td>batch size</td><td>64</td><td>64</td><td>64</td></tr><tr><td>sequence length</td><td>128</td><td>512</td><td>128</td></tr><tr><td>learning rate</td><td>2e-5</td><td>2e-4</td><td>2e-4</td></tr><tr><td rowspan="4">mT5-XL</td><td>epochs</td><td>10</td><td>10</td><td>20</td></tr><tr><td>batch size</td><td>64</td><td>64</td><td>64</td></tr><tr><td>sequence length</td><td>128</td><td>512</td><td>128</td></tr><tr><td>learning rate</td><td>2e-5</td><td>2e-4</td><td>2e-4</td></tr><tr><td rowspan="4">Llama 3.1 8B</td><td>epochs</td><td>3</td><td>3</td><td>5</td></tr><tr><td>batch size</td><td>64</td><td>64</td><td>64</td></tr><tr><td>sequence length</td><td>128</td><td>512</td><td>128</td></tr><tr><td>learning rate</td><td>2e-5</td><td>2e-4</td><td>2e-4</td></tr><tr><td rowspan="4">Gemma 2 9B</td><td>epochs</td><td>3</td><td>3</td><td>5</td></tr><tr><td>batch size</td><td>64</td><td>64</td><td>64</td></tr><tr><td>sequence length</td><td>128</td><td>512</td><td>128</td></tr><tr><td>learning rate</td><td>2e-5</td><td>2e-4</td><td>2e-4</td></tr></table>


Table 5: Hyperparameter configurations for each mPLM across the XNLI, TyDiQA, and NusaX datasets.


<table><tr><td>Model</td><td>en</td><td>ar</td><td>bg</td><td>de</td><td>el</td><td>es</td><td>fr</td><td>hi</td><td>ru</td><td>sw</td><td>th</td><td>tr</td><td>ur</td><td>vi</td><td>zh</td><td>Avg.</td></tr><tr><td colspan="17">Zero-Shot Cross-lingual Transfer</td></tr><tr><td>XLM-R Large</td><td>87.81</td><td>76.70</td><td>81.37</td><td>80.27</td><td>80.24</td><td>82.58</td><td>81.56</td><td>73.52</td><td>78.31</td><td>65.48</td><td>76.01</td><td>76.04</td><td>69.43</td><td>77.69</td><td>78.07</td><td>76.95</td></tr><tr><td>mT5-XL</td><td>89.04</td><td>77.13</td><td>82.27</td><td>81.34</td><td>81.00</td><td>83.43</td><td>82.74</td><td>74.58</td><td>80.22</td><td>70.19</td><td>75.93</td><td>77.22</td><td>70.35</td><td>76.33</td><td>78.15</td><td>77.92</td></tr><tr><td>Llama 3.1 8B</td><td>91.47</td><td>79.56</td><td>80.56</td><td>83.38</td><td>80.73</td><td>85.06</td><td>84.07</td><td>72.07</td><td>81.66</td><td>60.64</td><td>75.38</td><td>76.51</td><td>62.64</td><td>81.39</td><td>80.01</td><td>77.40</td></tr><tr><td>Gemma 2 9B</td><td>93.05</td><td>80.38</td><td>83.81</td><td>84.42</td><td>83.91</td><td>85.05</td><td>85.52</td><td>77.59</td><td>82.23</td><td>73.96</td><td>76.92</td><td>79.27</td><td>71.31</td><td>81.54</td><td>80.73</td><td>80.47</td></tr><tr><td colspan="17">Translate-Test (translate test data to English using NLLB 3.3B)</td></tr><tr><td>XLM-R Large</td><td>87.81</td><td>76.52</td><td>81.46</td><td>81.31</td><td>81.03</td><td>82.37</td><td>81.57</td><td>74.47</td><td>77.80</td><td>71.96</td><td>72.65</td><td>78.24</td><td>67.87</td><td>78.13</td><td>74.43</td><td>77.13</td></tr><tr><td>mT5-XL</td><td>89.04</td><td>79.04</td><td>83.15</td><td>83.07</td><td>82.56</td><td>83.73</td><td>83.30</td><td>76.69</td><td>80.47</td><td>73.02</td><td>74.69</td><td>79.54</td><td>69.80</td><td>80.26</td><td>77.15</td><td>79.03</td></tr><tr><td>Llama 3.1 8B</td><td>91.47</td><td>78.89</td><td>83.61</td><td>84.00</td><td>82.86</td><td>85.92</td><td>83.97</td><td>76.25</td><td>80.00</td><td>73.30</td><td>73.34</td><td>79.84</td><td>69.03</td><td>79.49</td><td>76.55</td><td>79.08</td></tr><tr><td>Gemma 2 9B</td><td>93.05</td><td>79.76</td><td>84.55</td><td>84.99</td><td>83.98</td><td>87.04</td><td>84.82</td><td>77.01</td><td>81.24</td><td>73.75</td><td>74.04</td><td>81.31</td><td>69.24</td><td>80.57</td><td>77.58</td><td>79.99</td></tr><tr><td colspan="17">Translate-Train (models are trained on training data translated to the target language)</td></tr><tr><td>XLM-R Large w/ LoRA</td><td>87.81</td><td>79.33</td><td>84.13</td><td>82.64</td><td>83.21</td><td>84.50</td><td>83.06</td><td>77.78</td><td>81.42</td><td>74.44</td><td>79.73</td><td>80.72</td><td>74.71</td><td>81.68</td><td>79.44</td><td>80.49</td></tr><tr><td>w/ X-Mixup</td><td>87.81</td><td>78.33</td><td>82.48</td><td>82.16</td><td>80.12</td><td>82.57</td><td>81.23</td><td>76.06</td><td>80.54</td><td>74.11</td><td>79.48</td><td>79.15</td><td>73.67</td><td>81.48</td><td>81.18</td><td>79.47</td></tr><tr><td>w/ input-level fusion</td><td>87.81</td><td>77.29</td><td>81.41</td><td>81.07</td><td>81.36</td><td>82.40</td><td>81.07</td><td>74.89</td><td>77.79</td><td>72.13</td><td>72.76</td><td>78.47</td><td>68.46</td><td>78.11</td><td>74.18</td><td>77.24</td></tr><tr><td>w/ FLARE MT</td><td>87.81</td><td>80.91</td><td>84.76</td><td>84.12</td><td>83.97</td><td>85.03</td><td>83.80</td><td>79.04</td><td>82.07</td><td>76.71</td><td>80.32</td><td>81.70</td><td>76.29</td><td>81.88</td><td>81.76</td><td>81.60</td></tr><tr><td>w/ FLARE</td><td>87.81</td><td>81.04</td><td>84.12</td><td>83.35</td><td>83.44</td><td>83.95</td><td>83.53</td><td>79.26</td><td>79.58</td><td>75.58</td><td>80.40</td><td>80.15</td><td>75.50</td><td>81.30</td><td>82.71</td><td>80.99</td></tr><tr><td>mT5-XL w/ LoRA</td><td>89.04</td><td>79.44</td><td>83.37</td><td>83.23</td><td>81.65</td><td>84.17</td><td>83.76</td><td>76.84</td><td>81.31</td><td>75.97</td><td>76.93</td><td>77.68</td><td>73.00</td><td>79.13</td><td>79.80</td><td>79.73</td></tr><tr><td>w/ X-Mixup</td><td>89.04</td><td>80.14</td><td>82.21</td><td>82.37</td><td>82.73</td><td>82.87</td><td>82.54</td><td>77.16</td><td>79.87</td><td>76.10</td><td>79.03</td><td>78.00</td><td>73.42</td><td>79.97</td><td>78.43</td><td>79.63</td></tr><tr><td>w/ input-level fusion</td><td>89.04</td><td>79.03</td><td>82.76</td><td>82.36</td><td>82.14</td><td>83.43</td><td>82.89</td><td>76.37</td><td>80.28</td><td>72.97</td><td>75.11</td><td>79.01</td><td>69.73</td><td>79.61</td><td>77.70</td><td>78.81</td></tr><tr><td>w/ FLARE MT</td><td>89.04</td><td>80.41</td><td>83.73</td><td>83.45</td><td>82.91</td><td>83.81</td><td>83.57</td><td>78.44</td><td>81.35</td><td>77.02</td><td>78.62</td><td>81.13</td><td>75.46</td><td>80.49</td><td>80.75</td><td>80.80</td></tr><tr><td>w/ FLARE</td><td>89.04</td><td>81.32</td><td>83.72</td><td>83.46</td><td>82.23</td><td>84.87</td><td>83.47</td><td>79.00</td><td>81.28</td><td>77.43</td><td>79.54</td><td>80.50</td><td>74.27</td><td>81.30</td><td>81.66</td><td>81.00</td></tr><tr><td>Llama 3.1 8B w/ LoRA</td><td>91.47</td><td>80.27</td><td>82.88</td><td>84.23</td><td>83.16</td><td>86.85</td><td>85.49</td><td>79.82</td><td>83.97</td><td>67.39</td><td>77.58</td><td>79.62</td><td>76.94</td><td>81.91</td><td>80.37</td><td>80.74</td></tr><tr><td>w/ X-Mixup</td><td>91.47</td><td>79.58</td><td>82.94</td><td>84.11</td><td>81.61</td><td>86.23</td><td>85.39</td><td>79.15</td><td>83.16</td><td>66.49</td><td>76.51</td><td>79.06</td><td>77.20</td><td>81.44</td><td>80.18</td><td>80.22</td></tr><tr><td>w/ input-level fusion</td><td>91.47</td><td>79.17</td><td>85.63</td><td>85.39</td><td>83.61</td><td>87.10</td><td>86.00</td><td>77.55</td><td>81.91</td><td>74.72</td><td>74.75</td><td>82.32</td><td>71.78</td><td>82.07</td><td>77.79</td><td>80.70</td></tr><tr><td>w/ FLARE MT</td><td>91.47</td><td>80.27</td><td>83.17</td><td>84.87</td><td>82.95</td><td>86.75</td><td>85.67</td><td>80.35</td><td>82.70</td><td>67.19</td><td>77.41</td><td>79.95</td><td>77.30</td><td>81.92</td><td>81.15</td><td>80.83</td></tr><tr><td>w/ FLARE</td><td>91.47</td><td>80.00</td><td>83.09</td><td>84.92</td><td>82.90</td><td>86.55</td><td>86.04</td><td>80.80</td><td>83.14</td><td>67.08</td><td>77.21</td><td>79.33</td><td>77.95</td><td>82.33</td><td>81.55</td><td>80.92</td></tr><tr><td>Gemma 2 9B w/ LoRA</td><td>93.05</td><td>85.19</td><td>87.87</td><td>88.03</td><td>87.78</td><td>89.06</td><td>87.13</td><td>82.49</td><td>86.10</td><td>79.52</td><td>83.20</td><td>84.25</td><td>78.07</td><td>85.09</td><td>84.69</td><td>84.89</td></tr><tr><td>w/ X-Mixup</td><td>93.05</td><td>84.70</td><td>87.76</td><td>87.84</td><td>87.32</td><td>88.61</td><td>87.83</td><td>82.44</td><td>85.27</td><td>80.12</td><td>82.60</td><td>83.51</td><td>77.35</td><td>84.50</td><td>84.86</td><td>84.62</td></tr><tr><td>w/ input-level fusion</td><td>93.05</td><td>79.98</td><td>84.84</td><td>85.19</td><td>84.16</td><td>86.65</td><td>85.12</td><td>77.39</td><td>81.74</td><td>74.48</td><td>75.07</td><td>81.98</td><td>70.70</td><td>81.74</td><td>78.34</td><td>80.53</td></tr><tr><td>w/ FLARE MT</td><td>93.05</td><td>85.07</td><td>87.73</td><td>87.89</td><td>87.70</td><td>88.93</td><td>87.90</td><td>82.69</td><td>84.97</td><td>80.52</td><td>82.82</td><td>84.18</td><td>77.36</td><td>84.88</td><td>85.19</td><td>84.84</td></tr><tr><td>w/ FLARE</td><td>93.05</td><td>84.67</td><td>87.93</td><td>88.14</td><td>87.77</td><td>89.23</td><td>88.10</td><td>82.86</td><td>85.97</td><td>79.73</td><td>83.15</td><td>84.08</td><td>78.13</td><td>85.23</td><td>85.19</td><td>85.01</td></tr><tr><td colspan="17">Translate-Train (fusion models are trained on data translated into the target language and evaluated using gold translations from the target language to the source language)</td></tr><tr><td>XLM-R Large w/ input-level fusion</td><td>87.81</td><td>88.41</td><td>88.54</td><td>88.46</td><td>88.36</td><td>88.28</td><td>88.02</td><td>88.38</td><td>85.91</td><td>86.23</td><td>85.91</td><td>85.85</td><td>86.05</td><td>85.85</td><td>86.45</td><td>87.19</td></tr><tr><td>w/ FLARE</td><td>87.81</td><td>88.10</td><td>88.06</td><td>88.04</td><td>88.12</td><td>88.02</td><td>88.08</td><td>88.40</td><td>88.12</td><td>88.46</td><td>88.16</td><td>88.14</td><td>88.22</td><td>88.04</td><td>88.16</td><td>88.15</td></tr><tr><td>mT5-XL w/ input-level fusion</td><td>89.04</td><td>90.04</td><td>89.80</td><td>89.54</td><td>89.70</td><td>89.78</td><td>89.50</td><td>89.80</td><td>89.52</td><td>89.56</td><td>89.84</td><td>89.66</td><td>89.38</td><td>89.52</td><td>89.70</td><td>89.67</td></tr><tr><td>FLARE</td><td>89.04</td><td>88.62</td><td>88.74</td><td>88.80</td><td>85.34</td><td>87.83</td><td>86.19</td><td>84.31</td><td>86.12</td><td>89.66</td><td>88.49</td><td>89.56</td><td>79.22</td><td>85.33</td><td>83.73</td><td>86.57</td></tr></table>


Table 6: Average scores per language in the XNLI dataset. Model performance is evaluated using the Accuracy metric.


<table><tr><td>Model</td><td>en</td><td>ar</td><td>ben</td><td>fi</td><td>ind</td><td>ko</td><td>ru</td><td>sw</td><td>tel</td><td>Avg.</td></tr><tr><td colspan="11">Zero-Shot Cross-lingual Transfer</td></tr><tr><td>XLM-R Large</td><td>49.05</td><td>24.27</td><td>32.78</td><td>30.15</td><td>44.51</td><td>29.92</td><td>30.15</td><td>40.27</td><td>58.42</td><td>36.31</td></tr><tr><td>mT5-XL</td><td>55.23</td><td>30.94</td><td>43.89</td><td>35.56</td><td>49.59</td><td>41.59</td><td>41.47</td><td>50.63</td><td>73.54</td><td>45.90</td></tr><tr><td>Llama 3.1 8B</td><td>55.61</td><td>2.41</td><td>0.00</td><td>5.22</td><td>1.64</td><td>0.58</td><td>5.49</td><td>2.52</td><td>1.06</td><td>2.36</td></tr><tr><td>Gemma 2 9B</td><td>60.08</td><td>1.63</td><td>2.46</td><td>3.20</td><td>0.88</td><td>0.24</td><td>2.77</td><td>1.87</td><td>6.61</td><td>2.46</td></tr><tr><td colspan="11">Translate-Test (translate test data to English using NLLB 3.3B)</td></tr><tr><td>XLM-R Large</td><td>49.05</td><td>28.15</td><td>52.78</td><td>30.54</td><td>47.79</td><td>36.23</td><td>34.92</td><td>52.76</td><td>45.30</td><td>41.06</td></tr><tr><td>mT5-XL</td><td>55.23</td><td>34.01</td><td>49.00</td><td>36.08</td><td>51.97</td><td>42.61</td><td>39.87</td><td>54.95</td><td>74.89</td><td>47.92</td></tr><tr><td>Llama 3.1 8B</td><td>55.61</td><td>1.35</td><td>5.00</td><td>2.21</td><td>0.99</td><td>0.58</td><td>2.71</td><td>4.68</td><td>2.75</td><td>2.53</td></tr><tr><td>Gemma 2 9B</td><td>60.08</td><td>0.68</td><td>3.33</td><td>1.74</td><td>0.66</td><td>0.00</td><td>1.68</td><td>1.08</td><td>9.04</td><td>2.28</td></tr><tr><td colspan="11">Translate-Train (models are trained on training data translated to the target language)</td></tr><tr><td>XLM-R Large w/ LoRA</td><td>49.05</td><td>31.10</td><td>38.97</td><td>30.35</td><td>46.61</td><td>37.11</td><td>24.54</td><td>47.13</td><td>65.34</td><td>40.14</td></tr><tr><td>w/ X-Mixup</td><td>49.05</td><td>26.25</td><td>36.67</td><td>26.65</td><td>43.70</td><td>34.19</td><td>24.40</td><td>45.85</td><td>68.21</td><td>38.24</td></tr><tr><td>w/ input-level fusion</td><td>49.05</td><td>31.56</td><td>40.00</td><td>30.04</td><td>45.63</td><td>33.33</td><td>28.12</td><td>50.29</td><td>64.62</td><td>40.45</td></tr><tr><td>w/ FLARE MT</td><td>49.05</td><td>30.21</td><td>35.00</td><td>33.02</td><td>44.82</td><td>35.90</td><td>25.38</td><td>48.46</td><td>58.26</td><td>38.88</td></tr><tr><td>w/ FLARE</td><td>49.05</td><td>30.83</td><td>41.67</td><td>33.44</td><td>44.61</td><td>35.33</td><td>26.17</td><td>48.47</td><td>66.93</td><td>40.93</td></tr><tr><td>mT5-XL w/ LoRA</td><td>55.23</td><td>33.48</td><td>46.71</td><td>38.90</td><td>49.84</td><td>46.77</td><td>33.20</td><td>51.46</td><td>73.73</td><td>46.76</td></tr><tr><td>w/ X-Mixup</td><td>55.23</td><td>32.55</td><td>54.49</td><td>38.15</td><td>52.17</td><td>49.05</td><td>33.75</td><td>52.04</td><td>73.68</td><td>48.23</td></tr><tr><td>w/ input-level fusion</td><td>55.23</td><td>34.75</td><td>49.74</td><td>39.29</td><td>51.74</td><td>45.29</td><td>30.69</td><td>52.79</td><td>76.32</td><td>47.58</td></tr><tr><td>w/ FLARE MT</td><td>55.23</td><td>46.08</td><td>48.59</td><td>39.31</td><td>53.94</td><td>44.90</td><td>30.14</td><td>49.16</td><td>75.75</td><td>48.48</td></tr><tr><td>w/ FLARE</td><td>55.23</td><td>47.86</td><td>49.55</td><td>40.98</td><td>54.23</td><td>46.05</td><td>30.41</td><td>50.85</td><td>74.82</td><td>49.34</td></tr><tr><td>Llama 3.1 8B w/ LoRA</td><td>55.61</td><td>39.69</td><td>26.11</td><td>44.27</td><td>56.61</td><td>53.56</td><td>37.23</td><td>53.47</td><td>31.75</td><td>42.84</td></tr><tr><td>w/ X-Mixup</td><td>55.61</td><td>23.44</td><td>16.11</td><td>37.26</td><td>30.69</td><td>0.00</td><td>10.96</td><td>0.00</td><td>21.32</td><td>17.47</td></tr><tr><td>w/ input-level fusion</td><td>55.61</td><td>37.48</td><td>21.03</td><td>45.86</td><td>56.68</td><td>62.61</td><td>37.03</td><td>61.82</td><td>46.20</td><td>46.09</td></tr><tr><td>w/ FLARE MT</td><td>55.61</td><td>38.33</td><td>26.11</td><td>37.90</td><td>48.78</td><td>53.85</td><td>34.04</td><td>44.94</td><td>27.63</td><td>38.95</td></tr><tr><td>w/ FLARE</td><td>55.61</td><td>44.48</td><td>26.66</td><td>48.09</td><td>62.80</td><td>56.98</td><td>42.44</td><td>59.73</td><td>40.70</td><td>47.74</td></tr><tr><td>Gemma 2 9B w/ LoRA</td><td>60.08</td><td>43.75</td><td>46.67</td><td>44.69</td><td>57.52</td><td>59.83</td><td>38.82</td><td>59.72</td><td>48.42</td><td>49.93</td></tr><tr><td>w/ X-Mixup</td><td>60.08</td><td>37.71</td><td>20.00</td><td>46.92</td><td>54.07</td><td>39.32</td><td>14.33</td><td>25.71</td><td>45.54</td><td>35.45</td></tr><tr><td>w/ input-level fusion</td><td>60.08</td><td>45.74</td><td>50.14</td><td>40.45</td><td>59.97</td><td>61.87</td><td>41.06</td><td>61.91</td><td>49.14</td><td>51.29</td></tr><tr><td>w/ FLARE MT</td><td>60.08</td><td>43.23</td><td>45.00</td><td>44.05</td><td>59.45</td><td>57.83</td><td>40.05</td><td>58.82</td><td>48.60</td><td>49.63</td></tr><tr><td>w/ FLARE</td><td>60.08</td><td>44.79</td><td>46.67</td><td>47.35</td><td>65.24</td><td>60.68</td><td>41.82</td><td>60.75</td><td>49.83</td><td>52.14</td></tr></table>


Table 7: Average scores per language in the TyDiQA dataset. Model performance is evaluated using the Exact Match metrics.


<table><tr><td>Model</td><td>en</td><td>ace</td><td>ban</td><td>bjn</td><td>bug</td><td>ind</td><td>jav</td><td>mad</td><td>min</td><td>nij</td><td>sun</td><td>Avg.</td></tr><tr><td colspan="13">Zero-Shot Cross-lingual Transfer</td></tr><tr><td>XLM-R Large</td><td>92.04</td><td>68.34</td><td>75.37</td><td>80.37</td><td>51.90</td><td>90.76</td><td>84.69</td><td>69.01</td><td>80.06</td><td>69.23</td><td>82.89</td><td>75.26</td></tr><tr><td>mT5-XL</td><td>91.77</td><td>72.26</td><td>76.42</td><td>79.79</td><td>49.51</td><td>90.61</td><td>87.49</td><td>61.38</td><td>77.71</td><td>65.31</td><td>86.73</td><td>74.72</td></tr><tr><td>Llama 3.1 8B</td><td>89.75</td><td>70.50</td><td>72.00</td><td>80.33</td><td>39.92</td><td>89.75</td><td>77.25</td><td>64.75</td><td>77.75</td><td>65.42</td><td>79.75</td><td>71.74</td></tr><tr><td>Gemma 2 9B</td><td>91.15</td><td>66.42</td><td>71.58</td><td>82.08</td><td>31.92</td><td>91.67</td><td>86.25</td><td>64.00</td><td>80.75</td><td>64.33</td><td>77.08</td><td>71.61</td></tr><tr><td colspan="13">Translate-Test (translate test data to English using NLLB 3.3B)</td></tr><tr><td>XLM-R Large</td><td>92.04</td><td>73.20</td><td>73.88</td><td>82.09</td><td>60.47</td><td>88.85</td><td>84.27</td><td>61.24</td><td>81.19</td><td>59.35</td><td>83.97</td><td>74.85</td></tr><tr><td>mT5-XL</td><td>91.77</td><td>76.27</td><td>73.43</td><td>81.72</td><td>69.29</td><td>86.86</td><td>83.50</td><td>60.63</td><td>82.47</td><td>60.86</td><td>82.68</td><td>75.77</td></tr><tr><td>Llama 3.1 8B</td><td>89.75</td><td>70.83</td><td>73.00</td><td>80.75</td><td>39.92</td><td>89.58</td><td>78.00</td><td>65.25</td><td>81.58</td><td>67.33</td><td>80.42</td><td>72.67</td></tr><tr><td>Gemma 2 9B</td><td>91.15</td><td>66.42</td><td>71.58</td><td>82.08</td><td>31.92</td><td>91.67</td><td>86.25</td><td>64.00</td><td>80.75</td><td>64.33</td><td>77.08</td><td>71.61</td></tr><tr><td colspan="13">Translate-Train (models are trained on training data translated to the target language)</td></tr><tr><td>XLM-R Large w/ LoRA</td><td>92.04</td><td>74.19</td><td>74.55</td><td>81.84</td><td>60.99</td><td>89.40</td><td>85.90</td><td>70.75</td><td>81.15</td><td>67.35</td><td>83.87</td><td>77.00</td></tr><tr><td>w/ X-Mixup</td><td>92.04</td><td>73.10</td><td>73.08</td><td>81.18</td><td>62.22</td><td>88.38</td><td>85.90</td><td>65.79</td><td>82.30</td><td>68.97</td><td>82.74</td><td>76.37</td></tr><tr><td>input-level fusion</td><td>92.04</td><td>77.77</td><td>75.89</td><td>82.67</td><td>69.96</td><td>89.44</td><td>87.92</td><td>66.66</td><td>79.55</td><td>68.47</td><td>87.01</td><td>78.53</td></tr><tr><td>w/ FLARE MT</td><td>92.04</td><td>73.33</td><td>75.95</td><td>81.13</td><td>57.20</td><td>90.76</td><td>86.59</td><td>69.77</td><td>83.42</td><td>68.90</td><td>84.73</td><td>77.18</td></tr><tr><td>w/ FLARE</td><td>92.04</td><td>76.47</td><td>77.27</td><td>80.71</td><td>70.18</td><td>90.54</td><td>87.42</td><td>71.33</td><td>85.15</td><td>70.16</td><td>82.59</td><td>79.18</td></tr><tr><td>mT5-XL w/ LoRA</td><td>91.77</td><td>80.66</td><td>81.92</td><td>85.83</td><td>65.36</td><td>89.78</td><td>90.40</td><td>69.85</td><td>82.30</td><td>69.27</td><td>88.76</td><td>80.41</td></tr><tr><td>w/ X-Mixup</td><td>91.77</td><td>80.34</td><td>74.60</td><td>83.76</td><td>68.87</td><td>88.52</td><td>88.75</td><td>68.25</td><td>83.66</td><td>65.60</td><td>83.76</td><td>78.61</td></tr><tr><td>input-level fusion</td><td>91.77</td><td>81.00</td><td>79.48</td><td>85.54</td><td>71.44</td><td>89.75</td><td>87.58</td><td>66.33</td><td>83.28</td><td>68.02</td><td>88.78</td><td>80.12</td></tr><tr><td>w/ FLARE MT</td><td>91.77</td><td>81.19</td><td>84.12</td><td>85.19</td><td>66.59</td><td>90.14</td><td>89.67</td><td>71.16</td><td>84.80</td><td>71.87</td><td>88.94</td><td>81.37</td></tr><tr><td>w/ FLARE</td><td>91.77</td><td>81.03</td><td>82.03</td><td>85.88</td><td>66.95</td><td>89.55</td><td>89.80</td><td>68.63</td><td>84.20</td><td>69.31</td><td>88.05</td><td>80.54</td></tr><tr><td>Llama 3.1 8B w/ LoRA</td><td>89.75</td><td>76.26</td><td>73.71</td><td>78.10</td><td>62.82</td><td>88.66</td><td>84.29</td><td>62.91</td><td>82.20</td><td>58.04</td><td>80.64</td><td>74.76</td></tr><tr><td>w/ X-Mixup</td><td>89.75</td><td>77.25</td><td>76.58</td><td>79.00</td><td>64.17</td><td>89.92</td><td>85.08</td><td>64.58</td><td>82.17</td><td>59.83</td><td>80.50</td><td>75.91</td></tr><tr><td>input-level fusion</td><td>89.75</td><td>74.83</td><td>66.17</td><td>80.17</td><td>66.67</td><td>89.17</td><td>85.50</td><td>59.63</td><td>82.63</td><td>57.25</td><td>84.00</td><td>74.60</td></tr><tr><td>w/ FLARE MT</td><td>89.75</td><td>78.21</td><td>72.00</td><td>74.29</td><td>64.21</td><td>87.96</td><td>83.04</td><td>64.38</td><td>80.75</td><td>62.38</td><td>77.96</td><td>74.52</td></tr><tr><td>w/ FLARE</td><td>89.75</td><td>78.88</td><td>75.25</td><td>80.25</td><td>64.25</td><td>91.17</td><td>85.88</td><td>65.13</td><td>81.38</td><td>57.88</td><td>80.75</td><td>76.08</td></tr><tr><td>Gemma 2 9B w/ LoRA</td><td>91.15</td><td>77.89</td><td>79.14</td><td>82.30</td><td>66.83</td><td>91.71</td><td>87.56</td><td>69.81</td><td>85.72</td><td>67.61</td><td>85.18</td><td>79.37</td></tr><tr><td>w/ X-Mixup</td><td>91.15</td><td>80.94</td><td>77.92</td><td>82.82</td><td>68.22</td><td>91.46</td><td>88.29</td><td>69.41</td><td>87.64</td><td>67.26</td><td>85.48</td><td>79.94</td></tr><tr><td>input-level fusion</td><td>91.15</td><td>77.67</td><td>76.88</td><td>83.50</td><td>70.58</td><td>89.92</td><td>87.17</td><td>63.92</td><td>84.50</td><td>60.71</td><td>84.92</td><td>77.98</td></tr><tr><td>w/ FLARE MT</td><td>91.15</td><td>79.88</td><td>80.67</td><td>81.88</td><td>62.50</td><td>91.04</td><td>85.67</td><td>66.04</td><td>84.71</td><td>64.42</td><td>84.08</td><td>78.09</td></tr><tr><td>w/ FLARE</td><td>91.15</td><td>82.75</td><td>81.00</td><td>83.17</td><td>65.08</td><td>92.83</td><td>86.83</td><td>73.08</td><td>87.75</td><td>70.58</td><td>85.50</td><td>80.86</td></tr><tr><td colspan="13">Translate-Train (fusion models are trained on data translated into the target language and evaluated using gold translations from the target language to the source language)</td></tr><tr><td>XLM-R Large input-level fusion</td><td>92.04</td><td>91.24</td><td>91.08</td><td>90.55</td><td>90.69</td><td>91.99</td><td>90.88</td><td>91.23</td><td>91.07</td><td>90.07</td><td>90.52</td><td>90.93</td></tr><tr><td>w/ FLARE</td><td>92.04</td><td>89.24</td><td>88.98</td><td>82.55</td><td>90.07</td><td>90.22</td><td>88.15</td><td>71.20</td><td>87.58</td><td>72.93</td><td>85.71</td><td>84.66</td></tr><tr><td>mT5-XL input-level fusion</td><td>91.77</td><td>91.39</td><td>90.39</td><td>91.47</td><td>91.54</td><td>90.88</td><td>89.49</td><td>88.87</td><td>90.86</td><td>89.20</td><td>91.60</td><td>90.57</td></tr><tr><td>w/ FLARE</td><td>91.77</td><td>83.80</td><td>80.55</td><td>84.06</td><td>64.70</td><td>88.32</td><td>90.50</td><td>74.36</td><td>83.64</td><td>69.29</td><td>88.00</td><td>80.72</td></tr></table>


Table 8: Average scores per language in the NusaX dataset. Model performance is evaluated using the Macro F1 metric.


<table><tr><td>Dataset</td><td>Model</td><td>LoRA</td><td>X-Mixup</td><td>Input-level fusion</td></tr><tr><td colspan="5">Translate-Train (models are trained on training data translated to the target language)</td></tr><tr><td rowspan="4">XNLI</td><td>XLM-R w/ FLARE MTw/ FLARE</td><td>0.000***0.139</td><td>0.000***0.001***</td><td>0.000***0.001***</td></tr><tr><td>mT5-XL w/ FLARE MTw/ FLARE</td><td>0.001***0.001***</td><td>0.000***0.000***</td><td>0.000***0.000***</td></tr><tr><td>Llama 3.1 8B w/ FLARE MTw/ FLARE</td><td>0.5640.321</td><td>0.001***0.000***</td><td>0.8850.813</td></tr><tr><td>Gemma 2 9B w/ FLARE MTw/ FLARE</td><td>0.7640.221</td><td>0.006***0.002***</td><td>0.000***0.000***</td></tr><tr><td rowspan="4">TyDiQA</td><td>XLM-R w/ FLARE MTw/ FLARE</td><td>0.3110.296</td><td>0.7100.035**</td><td>0.2130.531</td></tr><tr><td>mT5-XL w/ FLARE MTw/ FLARE</td><td>0.4340.181</td><td>0.9380.804</td><td>0.7720.393</td></tr><tr><td>Llama 3.1 8B w/ FLARE MTw/ FLARE</td><td>0.029**0.007***</td><td>0.007***0.007***</td><td>0.005***0.378</td></tr><tr><td>Gemma 2 9B w/ FLARE MTw/ FLARE</td><td>0.5360.015**</td><td>0.014**0.007***</td><td>0.1130.505</td></tr><tr><td rowspan="4">NusaX</td><td>XLM-R w/ FLARE MTw/ FLARE</td><td>0.7730.026**</td><td>0.3230.009***</td><td>0.4620.514</td></tr><tr><td>mT5-XL w/ FLARE MTw/ FLARE</td><td>0.044**0.693</td><td>0.023**0.032**</td><td>0.1920.543</td></tr><tr><td>Llama 3.1 8B w/ FLARE MTw/ FLARE</td><td>0.7500.015**</td><td>0.094*0.632</td><td>0.9640.265</td></tr><tr><td>Gemma 2 9B w/ FLARE MTw/ FLARE</td><td>0.086*0.041**</td><td>0.027**0.187</td><td>0.9360.088*</td></tr><tr><td>Task</td><td>Language</td><td>ISO Code</td><td>Source</td></tr><tr><td rowspan="14">XNLI</td><td>Arabic</td><td>ar</td><td></td></tr><tr><td>Bulgarian</td><td>bg</td><td colspan="2"></td></tr><tr><td>Chinese</td><td>zh</td><td colspan="2"></td></tr><tr><td>French</td><td>fr</td><td colspan="2"></td></tr><tr><td>German</td><td>de</td><td colspan="2"></td></tr><tr><td>Greek</td><td>el</td><td colspan="2"></td></tr><tr><td>Hindi</td><td>hi</td><td colspan="2" rowspan="2">Crowd-sourced (Williams et al., 2018)</td></tr><tr><td>Russian</td><td colspan="2">ru</td></tr><tr><td>Spanish</td><td>es</td><td colspan="2"></td></tr><tr><td>Swahili</td><td>sw</td><td colspan="2"></td></tr><tr><td>Thai</td><td>th</td><td colspan="2"></td></tr><tr><td>Turkish</td><td>tr</td><td colspan="2"></td></tr><tr><td>Urdu</td><td>ur</td><td colspan="2"></td></tr><tr><td>Vietnamese</td><td>vi</td><td colspan="2"></td></tr><tr><td rowspan="8">TyDiQA</td><td>Arabic</td><td>ar</td><td></td></tr><tr><td>Bengali</td><td>ben</td><td colspan="2"></td></tr><tr><td>Finnish</td><td>fi</td><td colspan="2"></td></tr><tr><td>Indonesian</td><td>ind</td><td colspan="2" rowspan="2">Wikipedia (Clark et al., 2020)</td></tr><tr><td>Korean</td><td colspan="2">ko</td></tr><tr><td>Russian</td><td>ru</td><td colspan="2"></td></tr><tr><td>Swahili</td><td>sw</td><td colspan="2"></td></tr><tr><td>Telugu</td><td>tel</td><td colspan="2"></td></tr><tr><td rowspan="9">NusaX</td><td>Acehnese</td><td>ace</td><td></td></tr><tr><td>Balinese</td><td>ban</td><td colspan="2"></td></tr><tr><td>Banjarese</td><td>bjn</td><td colspan="2"></td></tr><tr><td>Buginese</td><td>bug</td><td colspan="2"></td></tr><tr><td>Indonesian</td><td>ind</td><td colspan="2">SmSA (Purwarianti and Crisdayanti, 2019)</td></tr><tr><td>Javanese</td><td>jav</td><td colspan="2"></td></tr><tr><td>Madurese</td><td>mad</td><td colspan="2"></td></tr><tr><td>Minangkabau</td><td>min</td><td colspan="2"></td></tr><tr><td>Ngaju</td><td>nij</td><td colspan="2"></td></tr></table>


Table 9: P-values from the Pitman permutation test comparing FLARE and FLARE MT (rows) against baseline methods (columns) on the XNLI, TyDiQA, and NusaX datasets. The test is performed on average performance scores per language in the translate-train setting, as reported in Tables 6, 7, and 8. Statistical significance at the 90%, 95%, and 99% confidence levels is indicated by *, **, and ***, respectively.


Table 10: Overview of languages and corresponding source data used in the experiments, categorized by task. 