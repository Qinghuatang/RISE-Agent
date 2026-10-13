# Efficient One-shot Compression via Low-Rank Local Feature Distillation

Yaya Sy yaya.sy@loria.fr 

Christophe Cerisara christophe.cerisara@loria.fr 

Irina Illina illina@loria.fr 

Université de Lorraine, CNRS, Inria, LORIA, F-54000 Nancy, France 

## Abstract

Current structured pruning approaches for large language models typically involve two steps: (1) compression using calibration data and (2) costly continued pretraining on billions of tokens to recover lost performance. This second step is necessary as the first significantly im pacts model accuracy. Moreover, prior research suggests that pretrained Transformer weights are not necessarily low-rank, unlike their activations, making one-shot structured pruning challenging. Based on this observation, we propose Lillama, a compression method that locally distills activations with low-rank weights. Using SVD for initialization and a joint loss combining teacher and student activations, we accelerate convergence and reduce memory use with local gradient updates. Lillama compresses Mixtral-8x7B within minutes on a single A100 GPU, removing 10 billion parameters while retaining over 95% of its original performance. Phi-2 3B can be compressed by 40% with just 13 million calibration tokens, resulting in a small model that competes with recent models of similar size. The method generalizes well to non-transformer architectures, compressing Mamba-3B by 20% while maintaining 99% performance<sup>1</sup>. 

## 1 Introduction

Compressing Large Language Models is a crucial challenge that can be achieved with several complementary approaches: quantization, pruning, compression, and distillation. We propose a new oneshot compression method that locally distills lowrank weights and satisfies the three objectives: 

1. The compression algorithm must be computeefficient, e.g., it should require minimal computational resources and run in a reasonable amount of time. For instance, the iterative magnitude pruning algorithm as done in Frankle and Carbin (2019), is too costly to be applied to large-scale LLMs. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/1a99835f-18ab-4449-94da-363df06bc28e/7f79d2b122ec7c4f2393ab861b701d31ceea6250856ef1bb3c9cc6211e67be4d.jpg)



Figure 1: Lillama approach: STEP 1 selects layers to compress for a target compression ratio (e.g., N%) using various strategies (see Section 5). STEP 2 compresses and initializes the chosen parameters via SVD. STEP 3 distills the low-rank weights with a small calibration dataset.


2. It must achieve fast convergence using as little training data as possible; this objective is often an issue with many model pruning methods that often require significant retraining after pruning to recover the lost performance. 

3. The method should not cause a significant drop in performance; ideally, all the capabilities of the original LLM should remain intact. 

To address the first constraint, we compress the model with a local distillation objective, as opposed to a global objective that is more costly. We further reduce the cost by compressing and distilling only a subset of all layers. About the second constraint, we initialize the compressed layers with 

<table><tr><td>Method</td><td>No Custom Kernel</td><td>No Ampere-only</td><td>Memory Gain</td><td>Is One-Shot</td><td>Calib. Data</td></tr><tr><td>SparseGPT (Frantar and Alistarh, 2023)</td><td>✕</td><td>✕</td><td>✕</td><td>√</td><td>-</td></tr><tr><td>Wanda (Sun et al., 2024)</td><td>✕</td><td>✕</td><td>✕</td><td>√</td><td>-</td></tr><tr><td>Teal (Liu et al., 2024)</td><td>✕</td><td>✕</td><td>✕</td><td>√</td><td>-</td></tr><tr><td>ShearedLLaMA (Xia et al., 2024)</td><td>√</td><td>√</td><td>√</td><td>✕</td><td>52B</td></tr><tr><td>Minitron (Sreenivas et al., 2024)</td><td>√</td><td>√</td><td>√</td><td>✕</td><td>94B</td></tr><tr><td>Lillama</td><td>√</td><td>√</td><td>√</td><td>√</td><td>0.013B</td></tr></table>


Table 1: Positioning of our approach within the compression literature. No Custom Kernel indicates that a custom GPU kernel is not required to observe speedup and No Ampere-only is whether the method requires Ampere GPU to achieve speedup. Calib. Data is the total number of tokens in billions used for compression.


Singular Value Decomposition (SVD), which enables reducing the required number of gradient steps as well as the calibration dataset size. We further improve convergence by combining Teacher and Student activations with a joint distillation loss. We experimentally show the robustness of our approach by compressing several state-of-the-art small and large Transformers, Mixture-of-Experts, and Mamba LLMs. Figure 1 gives an intuitive overview of the proposed method, which is detailed in Section 4 and validated in Section 6. 

## 2 Related Works

We discuss next previous compression methods, highlight their limitations, and explain how our approach addresses them. Table 1 positions our approach to recent compression methods. 

Pruning methods remove unimportant weights in the pre-trained model (LeCun et al., 1989; Han et al., 2015). Structured Pruning removes entire groups of parameters, which results in a smaller and faster model (Xia et al., 2024; Ma et al., 2023). Ma et al. (2023) propose a new gradient-based criterion to eliminate substructures in LLMs, while Xia et al. (2024) use a joint loss combining a pruning mask loss with the language modeling loss. However, optimizing these criteria can be computationally intensive. For example, the pruning step of Sheared-LLaMA (Xia et al., 2023) is 5x expensive compared to standard LM training, according to the authors. In contrast, thanks to the local gradient updates, our approach is computationally efficient, allowing us to compress a 47B model within minutes on a single A100 GPU. Regarding unstructured pruning, these methods do not provide any gains in terms of memory or speedup, at least with current algorithmic implementations. Semi-structured pruning (e.g., 2:4 and 4:8) (Sun et al., 2024; Frantar and Alistarh, 2023; Liu et al., 2024) does not lead to memory gain but can speed up processing on kernels optimized for such matrix structures. On the other hand, our method, which directly shrinks matrices, saves memory across all hardware and leads to speed up, as fewer computations are performed. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/1a99835f-18ab-4449-94da-363df06bc28e/fbea45a206310ab7bea5a211e3be28fcec6c8f62e65d2b43b3280a9cf76c8c9c.jpg)



(a) Teacher


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/1a99835f-18ab-4449-94da-363df06bc28e/0195201cea45506cdd0a6588d16c473513b8cf784100931e7f420ddf38d6bdd7.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/1a99835f-18ab-4449-94da-363df06bc28e/06598f761855082df45c0daf18de6aa9115d4e7f1c941121a63f6bec33f6fc48.jpg)



(b) Student


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/1a99835f-18ab-4449-94da-363df06bc28e/03b2f4f99c9892a13e317a19ae826e655b6e1eec6740904999b69f38758459ce.jpg)



(c) Teacher+Student



Figure 2: Accelerating convergence by learning from Teacher and Student activations through a joint loss. We propose to study the effect of three distillation strategies: (a) Teacher: the input to the compressed student layer comes from the output of the previous teacher layer; (b) Student: the input to the compressed student layer comes from the output of the previous student layer; (c) Teacher+Student: the compressed student layer receives both the output of the previous teacher layer and the output of the previous student layer, and the loss is the sum of two losses.


Low-Rank Decomposition compresses a model by approximating its pre-trained matrices with lowerdimension ones. Based on the observation by Li et al. (2018) that Neural Networks have lower intrinsic dimensions, Aghajanyan et al. (2020) show that Transformer language models also require lower intrinsic dimensions depending on the language task. However, Yu and Wu (2023); Chen et al. (2021) show that the weights in some pretrained Transformer models are full-rank compared to their activations and propose an activation-aware compression method. Our work is related to these approaches, as we also decompose the matrices of the models. However, we apply a lightweight layerwise feature distillation objective to better recover from compression and propose different distillation strategies to accelerate convergence and improve performance. 

Distillation. In Knowledge Distillation (KD), a small student model learns to reproduce the outputs of a larger pre-trained model. Standard KD is dataintensive when the student model’s parameters are randomly initialized. Team et al. (2024) recently distilled Gemma-2 9B from a larger model using 8 trillion tokens, which is as costly as a standard pretraining stage. Another approach is layer-wise distillation, which can be more suitable in some cases, as intermediate layers may contain certain information, such as speaker identity or phonemes in pretrained speech models (Baevski et al., 2020; Pasad et al., 2022; Chang et al., 2022, 2021). For example, TinyBERT (Jiao et al., 2020) initializes the student model by removing layers from the teacher and then distills the student on task-specific knowledge, combining layer-wise and global distillation losses, which took 3.5 days to train. This is costly for LLMs. Instead of removing layers, we leverage the observation that layer activations are low-rank and approximate them with fewer parameters. Additionally, we compress the models to remain generalist rather than task-specific. 

## 3 Background: Low-Rank Approximation of Transformer Models

## 3.1 Low-Rank Approximation from Weights

Given $W \in \mathbb { R } ^ { d _ { 1 } \times d _ { 2 } }$ , a pretrained weight matrix, and an example $\boldsymbol { x } \in \mathbb { R } ^ { d _ { 2 } }$ , the activations $\boldsymbol { y } \in \mathbb { R } ^ { d _ { 1 } }$ are computed as $y = W x$ . It is possible to reduce the computations in this operation by using a lowrank approximation of $W \colon y \simeq \Delta W x = A B x$ with $\Delta W = A B$ the low-rank decomposition of $W .$ , composed of $A \ \in \ \mathbb { R } ^ { d _ { 1 } \times r }$ and $B \ \in \ \mathbb { R } ^ { r \times d _ { 2 } }$ When the rank r is small enough, the number of parameters $r ( d _ { 1 } + d _ { 2 } )$ in the low-rank equation is smaller than $d _ { 1 } d _ { 2 }$ in the full-rank equation. Estimating the low-rank matrix $\Delta W = A B$ can be formulated as a minimization problem to find the low-rank $\Delta W$ that best approximates $W \colon$ 

$$
\widehat {\Delta W} = \underset {\Delta W} {\operatorname{argmin}} \| W - \Delta W \| _ {F}\tag{1}
$$

It is well-known that this minimization problem can be approached using SVD, a low-rank approximation method that offers the optimal r rank approximation of a given matrix with regard to the Frobenius norm $\| \cdot \| _ { F }$ . SVD approximates a matrix W into three matrices: $W = U S V ^ { T }$ , where $S \in \mathbb { R } ^ { d _ { 1 } \times d _ { 2 } }$ is a diagonal matrix containing the singular values of W sorted in descending order, $U \in \mathbb { R } ^ { d _ { 1 } \times d _ { 1 } }$ and $V \in \mathbb { R } ^ { d _ { 2 } \times d _ { 2 } }$ are orthonormal matrices. The optimal low-rank approximation of $W \in \mathbb { R } ^ { d _ { 1 } \times d _ { 2 } }$ can be obtained by keeping the first r singular values, with $r < m i n ( d _ { 1 } , d _ { 2 } )$ 

$$
\widehat {\Delta W} = (U _ {:,: r} S _ {: r,: r}) V _ {: r,:} = A B\tag{2}
$$

with $A = U _ { : , : r } S _ { : r , : r }$ and $B = V _ { : r , : }$ and the notation $_ { : a , : b }$ refers to the slicing operation. 

## 3.2 Low-Rank Approximation from Feature

Previous studies (Chen et al., 2021; Yu and Wu, 2023) have shown that the activations (i.e.,features) of pretrained transformers, are more low-rank than the weights. Recently, Liu et al. (2024) show that transformer activations can be sparsified up to 60% without too much drop in performance. In $\mathsf { A p - }$ pendix $_ { \mathrm { A . 6 , } }$ we also show that the activations of the transformer layers are more low-rank than the weights. All these results suggest transformer activations can be approximated with fewer parameters. Now, let’s examine how previous works (Kaushal et al., 2023; Chen et al., 2021; Yu and Wu, 2023) have approached this. 

Let $\mathcal { D } = \{ x _ { i } \in \mathbb { R } ^ { d _ { 2 } } \} _ { 1 \leq i \leq N }$ be a calibration dataset, and $W \in \mathbb { R } ^ { d _ { 1 } \times d _ { 2 } }$ the parameters of a linear layer. Finding the low-rank matrix $\Delta W$ that best reproduces the activations $\{ y = W x \} \quad \forall x \in { \mathcal { D } }$ involves solving: 

$$
\widehat {\Delta W} = \underset {\Delta W} {\operatorname{argmin}} \frac {1}{N} \sum_ {x \in \mathcal {D}} \| W x - \Delta W x \| _ {F}\tag{3}
$$

An analytic solution to this minimization problem is the eigendecomposition of the covariance matrix of the activations. We first collect all activations $\mathcal { Y } = \{ \boldsymbol { y } = W \boldsymbol { x } \} _ { \forall \boldsymbol { x } \in \mathcal { D } }$ , then the covariance matrix of the activations can be estimated as $\Sigma = \underset { y \in \mathcal { y } } { \mathbb { E } } \left[ y y ^ { T } \right] - \mathbb { E } [ y ] \mathbb { E } [ y ] ^ { T }$ with $\Sigma \in \mathbb { R } ^ { d _ { 1 } \times d _ { 1 } }$ . Since Σ is a diagonalizable matrix, we can apply its eigendecomposition: $\Sigma = U S U ^ { T }$ , where $U \in \mathbb { R } ^ { d _ { 1 } \times d _ { 1 } }$ contains the eigenvectors of Σ and $S \in \mathbb { R } ^ { d _ { 1 } \times d _ { 1 } }$ is the diagonal matrix containing the eigenvalues sorted in decreasing order. As for $\mathrm { E q } { - } 2 ,$ , we can only keep the eigenvectors corresponding to the r largest eigenvalues, which gives us $A = U _ { : ,: r }$ and $B = U _ { : ,: r } ^ { T } W$ . Compared to the low-rank matrices $\operatorname { E q } - 2$ , A and B weights here learned to reproduce the activations of the base weight $W$ , thanks to the minimization objective in $\operatorname { E q } - 3$ . In the next section, we highlight the limitations of this approach and introduce our proposed method. 

## 4 Proposed Approach

We identify and handle next three potential limitations of the approach presented in Section 3.2. 

Non-linear Feature Approximation. The previous analytical solution for Eq-3 is limited to activations from linear layers. To generalize to non-linear modules, we first observe that Eq-3 can be seen as a feature distillation objective that may be optimized numerically by gradient descent rather than analytically by eigendecomposition. Given the input batch $\boldsymbol { X } \in \mathbb { R } ^ { d \times b }$ of b examples, we denote the output activations of the $i ^ { t h }$ Teacher module $\mathcal { T } ^ { ( i ) }$ as 

$$
Y ^ {(i)} = \mathcal {T} ^ {(i)} (X; \Theta^ {(i)})\tag{4}
$$

where $\Theta ^ { ( i ) }$ are the original pretrained matrices of the $i ^ { t h }$ Teacher, and $\bar { Y ^ { ( i ) } } \in \bar { \mathbb { R } } ^ { d \times b }$ its output activations. Similarly, we note the output activations of the $i ^ { t h }$ Student module $\boldsymbol { S } ^ { ( i ) }$ as : 

$$
\widehat {Y} ^ {(i)} = \mathcal {S} ^ {(i)} (X; \Delta \Theta^ {(i)})\tag{5}
$$

where the Student module $\boldsymbol { S } ^ { ( i ) }$ is parametrized with the low-rank matrices $\Delta \Theta ^ { ( i ) }$ and $\widehat { Y } ^ { ( i ) } \in \mathbb { R } ^ { d \times b }$ bare the output activations. We can express the distillation objective of the $i ^ { t h }$ Student module as: 

$$
\widehat {\Delta \Theta^ {(i)}} = \underset {\Delta \Theta^ {(i)}} {\operatorname{argmin}} \mathcal {L} ^ {(i)} (Y ^ {(i)}, \widehat {Y} ^ {(i)})\tag{6}
$$

where $\widehat { \Delta \Theta ^ { ( i ) } }$ are the estimated low-rank matrices of the $i ^ { t h }$ Student module $\boldsymbol { S } ^ { ( i ) }$ and $\mathcal { L } ^ { ( i ) } ( Y ^ { ( i ) } , \ \widehat { Y } ^ { ( i ) } )$ bis the loss measuring the distance between the acti vations of the Teacher and the Student. We opted for the same loss as Chang et al. (2022) because we observed that the $\ell _ { 1 }$ loss yields instabilities: 

$$
\begin{array}{l} \mathcal {L} ^ {(i)} = \mathcal {L} _ {\ell_ {1}} ^ {(i)} + \mathcal {L} _ {\cos} ^ {(i)} \\ = \sum_ {t = 1} ^ {b} \left[ \frac {1}{D} \left\| Y _ {t} ^ {(i)} - \widehat {Y} _ {t} ^ {(i)} \right\| _ {1} - \log \sigma \left(\cos \left(Y _ {t} ^ {(i)}, \widehat {Y} _ {t} ^ {(i)}\right)\right) \right] \end{array} \tag {7}
$$

where D is the hidden vectors dimension, σ is the sigmoid activation and cos( , ) is the cosine similarity. We propose to approximate the low-rank weights of the Students through Gradient Descent. In the linear case, this should converge towards the eigendecomposition solution described in Section 3.2. However, as we show later, we extend the distillation process to non-linear modules, which requires numerical optimization. We also show that initializing the Student’s low-rank parameters with SVD, rather than randomly, improves convergence. Beyond Teacher-only Activations. The second potential limitation of the approach described in Section 3.2 and equations 4 and 5 is that the Student and the Teacher modules take the same input X, which is the output of the precedent layer of the Teacher. This means that the Student is trained from the activations of the Teacher module only, which are not available at inference time. Yu and Wu (2023) use this approach to approximate the Linear Layers of Transformer models, referring to it as Atomic Feature Mimicking. Let $\widehat { Y } _ { T } ^ { ( i ) } =$ $S ^ { ( i ) } ( Y _ { T } ^ { ( i - 1 ) } ; \Delta \Theta ^ { ( i ) } )$ bbe the output activations of the Student module when fed $Y _ { T } ^ { ( i - 1 ) }$ the output activations of the Teacher $T$ at the previous layer $i - 1$ . The loss can then be written as: 

$$
\mathcal {L} _ {T} ^ {(i)} = \mathcal {L} ^ {(i)} (Y _ {T} ^ {(i)}, \widehat {Y} _ {T} ^ {(i)})\tag{8}
$$

where $Y _ { i } ^ { T }$ are the gold activations of the current Teacher. This distillation procedure is illustrated in Figure 2a. This approach can lead to a fast convergence as the Student modules learn from the gold and high-quality activations produced by the Teacher modules. However, it can cause a performance drop during inference as the Teacher activations will not be available. An alternative approach is to have the Student module take as input the activations of the previous Student modules rather than those of the Teacher modules: 

$$
\mathcal {L} _ {S} ^ {(i)} = \mathcal {L} ^ {(i)} (Y _ {T} ^ {(i)}, \widehat {Y} _ {S} ^ {(i)})\tag{9}
$$

where $\widehat { Y } _ { S } ^ { ( i ) } = S ^ { ( i ) } ( \widehat { Y } _ { S } ^ { ( i - 1 ) } ; \Delta \Theta ^ { ( i ) } )$ is the output activab btions of the current Student module S when taking as input the output activations $\widehat { Y } _ { S } ^ { ( i - 1 ) }$ of the Student bmodule of the previous layer, and $Y _ { T } ^ { \left( i \right) }$ are the gold activations of the corresponding Teacher module. This is the standard distillation and is illustrated in Figure 2b. However, although we found this loss performs generally better than the previous one, it leads to a slow convergence due to the errors in the activations of the Student modules. To address both issues, we propose to combine the two losses: 

$$
\mathcal {L} _ {T + S} ^ {(i)} = \mathcal {L} _ {T} ^ {(i)} + \mathcal {L} _ {S} ^ {(i)}\tag{10}
$$

This distillation approach is illustrated in Figure 2c. We don’t introduce any hyperparameter for controlling the importance of one loss over another in the joint loss. We leave this for future work. 

Module-wise Distillation. Finally, we observe that the functions can be any other module in the neural network than linear layers: any sub-part of the model that outputs low-rank activations can be compressed with our method. In our experiments, we distill at the Transformer layer level and leave comparisons with other distillation levels for future work. We also experiment with other models than pretrained Transformer-based language models. 

## 5 Fast and Memory-Efficient Compression

Computing the optimal compression rank of each matrix individually may be costly: while search algorithms can be a solution for smaller models, they can be difficult to scale to LLMs with billions of parameters. Assuming a target compression rate N, we propose to consider three simple strategies: uniform, top, and bottom first compression strategies, which are illustrated in Appendix A.1. 

Uniform. This approach removes N% parameters to each weight matrix in the model. This strategy involves training all LLM weights. 

Bottom. This approach removes N% parameters from the model by prioritizing lower layers as detailed in Algorithm 1. The core of the algorithm is line 7, where the layers are ordered in bottom-top order and ranks are sorted in decreasing order. The argument k controls the spread of compression: the lower its value the more weights in the bottom layer will be compressed first. The algorithm is also illustrated in Steps 1 and 2 in Figure 1. This strategy is efficient as only a subset of weights in the bottom layers will be trained. Consequently, since there is no need to forward through the whole model, only a subset of the weights needs to be loaded into GPU memory, making this approach scalable to larger models. 

Top. This approach removes N% parameters from the model by prioritizing top layers. This consists roughly of reversing the order of layers in the line 7 of Algorithm 1. However, this strategy requires loading all the model’s parameters into GPU memory. It also involves forwarding through the entire model, even if only the top layer weights are being trained. 

## 6 Experiments on Transformers

## 6.1 Setup

Models. We first evaluate our method on various transformer-based LLMs: a 47B mixture-of experts language model (Mixtral-v0.1 8x7B (Jiang et al., 2024)), medium-sized LLMs (Mistralv0.1 7B (Jiang et al., 2023) and Phi-3 14B <sup>2</sup>), and a small language model (Phi-2 3B<sup>3</sup>). 

Data. For calibration data, we use 13 million tokens randomly sampled from Slim-Orca (Lian et al., 2023)<sup>4</sup>, an open-source replication of the Orca instruction dataset (Mukherjee et al., 2023). In preliminary experiments, we also tested other datasets, such as RedPajama (Computer, 2023), but found the results were better with instruction data. Hyperparameters. When compressing models using Algorithm 1, we set a minimum rank of k = 1024 (except for Phi-3 14B, where $k \mathbf { \Sigma } =$ 1536 yielded better results). For all models, the rank increment m is set to 256. We use the Teacher+Student loss (Figure 2c). Section 6.2 ablates these choices. 

Algorithm 1: Bottom Layers First Compression Algorithm

Input: M is the base model; S is the target size of the compressed model; k is the minimum possible rank to set; m the increment when generating ranks.
Result: Low-Rank Compressed M.

1 $R \leftarrow \varnothing$ ; // Stack that will contain ranks and weight matrices.

2 for each weight matrix $W \in R^{d_{1} \times d_{2}}$ in M do
3 $b \leftarrow \min(d_{1}, d_{2})$ 4 for $r = k$ ; $r \leq b$ ; $r = r + m$ do
5 if $r \times (d_{1} + d_{2}) < d_{1} \times d_{2}$ then
6 $R \leftarrow R \cup \{(r, W)\}$ 7 sort R primarily by layer index in increasing order and secondarily by rank in decreasing order.
8 while $|M| > S$ do
9 get the next $(r, W)$ from the stack R
10 compute A, B from W using Eq. 2 with the rank r
11 replace W by A, B in M
12 return M 

Evaluation. We evaluate the speed by measuring the time to forward a batch of 4 prompts of varying sequence lengths. Using lm-evaluationharness (Gao et al., 2023), we evaluate the zeroshot performance of the base and compressed models on 9 downstream tasks: TruthfulQA (TruQA; (Lin et al., 2022)), Social IQa (SIQA; (Sap et al., 2019)), LogiQA (Liu et al., 2020), WinoGrande (WinoG; (Sakaguchi et al., 2019)), Arc Easy (ARC-E; (Clark et al., 2018)), Arc Challenge (ARC-C; (Clark et al., 2018)), BoolQ (Clark et al., 2019), PIQA (Bisk et al., 2020), OpenBookQA (OBQA; (Mihaylov et al., 2018)). 

## 6.2 Results

Our method can retain 97% of the zero-shot performance. As shown in Table 2, models compressed by 20% maintain over 93% of the base models’ performance, regardless of the size. The larger models, Phi-3 14B and Mixtral-8x7B-v0.1 47B, retain 97% and 96% of the performance, respectively. Overall, the models tend to lose the most performance on the ARC-C task, with losses also in the commonsense reasoning task WinoG. We provide additional results for 25% and 30% compression ratios in Appendix A.2. 

<table><tr><td>Model</td><td>Reduction</td><td>TruQA</td><td>SIQA</td><td>LogiQA</td><td>WinoG</td><td>ARC-E</td><td>ARC-C</td><td>BoolQ</td><td>PIQA</td><td>OBQA</td><td>Average</td></tr><tr><td rowspan="2">Phi-3 14B</td><td>0%</td><td>57.63</td><td>57.06</td><td>37.94</td><td>76.01</td><td>81.36</td><td>61.60</td><td>88.56</td><td>81.34</td><td>50.40</td><td>65.77</td></tr><tr><td>20%</td><td>57.88</td><td>50.26</td><td>34.10</td><td>72.53</td><td>83.54</td><td>59.22</td><td>85.32</td><td>80.30</td><td>49.20</td><td>63.59</td></tr><tr><td rowspan="2">Mixtral-8x7B-v0.1 47B</td><td>0%</td><td>48.58</td><td>49.54</td><td>33.18</td><td>76.64</td><td>83.50</td><td>60.07</td><td>85.23</td><td>83.41</td><td>47.00</td><td>63.02</td></tr><tr><td>20%</td><td>44.78</td><td>47.85</td><td>30.41</td><td>72.85</td><td>80.81</td><td>56.74</td><td>81.41</td><td>80.30</td><td>46.60</td><td>60.19</td></tr><tr><td rowspan="2">Phi-2 3B</td><td>0%</td><td>44.40</td><td>55.42</td><td>30.57</td><td>76.16</td><td>78.16</td><td>54.18</td><td>83.21</td><td>79.11</td><td>51.20</td><td>61.38</td></tr><tr><td>20%</td><td>46.04</td><td>52.56</td><td>29.80</td><td>71.51</td><td>72.81</td><td>46.50</td><td>75.05</td><td>76.55</td><td>45.60</td><td>57.38</td></tr><tr><td rowspan="2">Mistral-v0.1 7B</td><td>0%</td><td>42.60</td><td>46.57</td><td>29.80</td><td>73.80</td><td>79.55</td><td>53.92</td><td>83.70</td><td>82.10</td><td>44.00</td><td>59.56</td></tr><tr><td>20%</td><td>44.14</td><td>45.50</td><td>28.26</td><td>66.69</td><td>73.11</td><td>46.33</td><td>77.00</td><td>77.37</td><td>41.20</td><td>55.51</td></tr></table>


Table 2: Compressed models can retain 97% of the zero-shot performance of the base model. Zero-shot performances of the base non-compressed models (0%) and the 20% compressed models.


<table><tr><td>Model</td><td>Reduction</td><td>Model Size</td><td>VRAM</td><td>s = 512</td><td>s = 1024</td><td>s = 2048</td><td>s = 4096</td><td>s = 8192</td><td>s = 16384</td></tr><tr><td rowspan="2">Phi-3 14B</td><td>0%</td><td>14B</td><td>28 GB</td><td>7171 t/s</td><td>7216 t/s</td><td>7197 t/s</td><td>7057 t/s</td><td>7010 t/s</td><td>7036 t/s</td></tr><tr><td>20%</td><td>11B</td><td>22.8 GB</td><td>8622 t/s</td><td>8686 t/s</td><td>8509 t/s</td><td>8349 t/s</td><td>8268 t/s</td><td>8269 t/s</td></tr><tr><td rowspan="2">Mixtral-8x7B-v0.1 47B</td><td>0%</td><td>47B</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td></tr><tr><td>20%</td><td>37B</td><td>73.8 GB</td><td>6515 t/s</td><td>8143 t/s</td><td>8444 t/s</td><td>OOM</td><td>OOM</td><td>OOM</td></tr><tr><td rowspan="2">Phi-2 3B</td><td>0%</td><td>2.8B</td><td>6.8 GB</td><td>29949 t/s</td><td>30895 t/s</td><td>30184 t/s</td><td>30403 t/s</td><td>26636 t/s</td><td>21499 t/s</td></tr><tr><td>20%</td><td>2.2B</td><td>5.7 GB</td><td>32451 t/s</td><td>34561 t/s</td><td>34276 t/s</td><td>32479 t/s</td><td>30351 t/s</td><td>23735 t/s</td></tr><tr><td rowspan="2">Mistral-v0.1 7B</td><td>0%</td><td>7.2B</td><td>15.2 GB</td><td>13385 t/s</td><td>13537 t/s</td><td>13650 t/s</td><td>13265 t/s</td><td>12603 t/s</td><td>12399 t/s</td></tr><tr><td>20%</td><td>5.8B</td><td>12.6 GB</td><td>15416 t/s</td><td>15832 t/s</td><td>15947 t/s</td><td>15446 t/s</td><td>14589 t/s</td><td>14304 t/s</td></tr></table>


Table 3: Compressed models save memory and speedup computation. Inference speed was measured on a single A100 GPU, using the whole test split of Wikitext2 using a batch size of 4 and by varying the sequence length from 512 to 16384 tokens. All the models were loaded in bf16 and used Flash-Attention2. OOM stands for Out-Of-Memory.


Low-Rank models are lighter and faster. Table 3 shows the memory gain (VRAM) and speedup of the compressed models. We can see that 20% compressed models are lighter and up to 20% faster. Notably, while Mixtral-8x7B-v0.1 cannot fit into a single A100-80GB GPU, after compression, it can fit and process up to a 2048 context length with a batch size of 4. This is possible thanks to the memory efficiency of the bottom-first compression strategy and to the local gradient updates approach. Our approach can efficiently create Small Language Models. We compressed Phi-2 3B to 1.7B (40% of parameter reduction) and compare its performance to recent state-of-the-art small language models of equivalent size: StableLM-2 1.6B and Qwen-2 1.5B. Results are presented in Table 4. 

Compressed Models Show Good Recovery with 

Fine-Tuning. Table 5 demonstrates that fine-tuning these compressed models leads to performance recovery. Notably, the 40% compressed Mistral 7Bv0.1 retains 91% of the original performance despite being fine-tuned on only 191 million tokens for fine-tuning. Additional data could likely lead to even greater improvements. 

## 7 Other Architectures and Modalities

In this section, we show the generalizability of our approach by evaluating it on the Mamba architecture for text (Gu and Dao, 2024) and on the speech modality with Whisper (Radford et al., 2022). We further evaluate it on a lower-resource African language in Appendix A.11. 

## 7.1 Mamba Architecture

The attention mechanism in the Transformer is limited by its increasing complexity as a function of the input sequence length. Therefore, Linear Attention Models have been proposed as alternatives to Transformers, in particular State Space Models, such as the Mamba architecture (Gu and Dao, 2024), which have shown promising results. 

Method. We tested our approach on two Mamba 

<table><tr><td>Model</td><td>TruQA</td><td>SIQA</td><td>LogiQA</td><td>WinoG</td><td>ARC-E</td><td>ARC-C</td><td>BoolQ</td><td>PIQA</td><td>OBQA</td><td>Average</td></tr><tr><td>StableLM-2 1.6B</td><td>38.90</td><td>48.52</td><td>26.73</td><td>63.30</td><td>68.39</td><td>38.57</td><td>74.80</td><td>76.93</td><td>39.00</td><td>52.79</td></tr><tr><td>Qwen-1 1.8B</td><td>38.09</td><td>45.19</td><td>31.80</td><td>59.12</td><td>58.33</td><td>34.98</td><td>65.93</td><td>73.23</td><td>33.60</td><td>48.92</td></tr><tr><td>Qwen-2 1.5B</td><td>45.93</td><td>45.85</td><td>31.18</td><td>66.22</td><td>60.56</td><td>36.09</td><td>72.26</td><td>75.35</td><td>36.40</td><td>52.20</td></tr><tr><td>Lillama (Phi-2 1.7B)</td><td>44.08</td><td>47.80</td><td>25.65</td><td>68.27</td><td>63.22</td><td>38.82</td><td>76.02</td><td>72.36</td><td>39.20</td><td>52.82</td></tr></table>


Table 4: 40% compressed Phi-2 3B competes with models of similar size while being compressed using only 13M tokens. Compressing Phi-2 3B to 1.7B (by removing 40% of the parameters) and comparing its zero-shot performance to other recent language models of similar sizes.


<table><tr><td>Model</td><td>Reduction</td><td>TruQA</td><td>SIQA</td><td>LogiQA</td><td>WinoG</td><td>ARC-E</td><td>ARC-C</td><td>BoolQ</td><td>PIQA</td><td>OBQA</td><td>Average</td></tr><tr><td rowspan="3">Phi-2 3B</td><td>0%</td><td>44.40</td><td>55.42</td><td>30.57</td><td>76.16</td><td>78.16</td><td>54.18</td><td>83.21</td><td>79.11</td><td>51.20</td><td>61.38</td></tr><tr><td>40%</td><td>44.08</td><td>47.80</td><td>25.65</td><td>68.27</td><td>63.22</td><td>38.82</td><td>76.02</td><td>72.36</td><td>39.20</td><td>52.82</td></tr><tr><td>40% FT</td><td>45.43</td><td>49.49</td><td>31.80</td><td>70.09</td><td>60.86</td><td>37.29</td><td>67.68</td><td>73.23</td><td>42.00</td><td>53.10</td></tr><tr><td rowspan="3">Mistral-v0.1 7B</td><td>0%</td><td>42.60</td><td>46.57</td><td>29.80</td><td>73.80</td><td>79.55</td><td>53.92</td><td>83.70</td><td>82.10</td><td>44.00</td><td>59.56</td></tr><tr><td>40%</td><td>41.69</td><td>43.65</td><td>30.57</td><td>62.35</td><td>61.66</td><td>34.56</td><td>75.23</td><td>71.49</td><td>34.80</td><td>50.63</td></tr><tr><td>40% FT</td><td>41.60</td><td>52.25</td><td>29.65</td><td>66.61</td><td>66.75</td><td>39.93</td><td>79.39</td><td>74.37</td><td>38.29</td><td>54.32</td></tr></table>


Table 5: Compressed models recover well when finetuned. Zero-shot performances of the base non-compressed models (0%) and the 40% compressed models, without and with fine-tuning (FT). Models were fine-tuned on the whole Slim-Orca dataset (191 million tokens).


<table><tr><td>Model</td><td>Reduction</td><td>TruQA</td><td>SIQA</td><td>LogiQA</td><td>WinoG</td><td>ARC-E</td><td>ARC-C</td><td>BoolQ</td><td>PIQA</td><td>OBQA</td><td>Average</td></tr><tr><td rowspan="2">Falcon-Mamba 7B</td><td>0%</td><td>53.40</td><td>52.66</td><td>31.34</td><td>74.66</td><td>81.73</td><td>58.96</td><td>83.12</td><td>81.56</td><td>48.60</td><td>62.89</td></tr><tr><td>20%</td><td>52.34</td><td>50.10</td><td>30.72</td><td>66.69</td><td>76.73</td><td>49.23</td><td>77.22</td><td>78.02</td><td>44.80</td><td>58.43</td></tr><tr><td rowspan="2">Mamba 3B</td><td>0%</td><td>35.88</td><td>43.24</td><td>26.88</td><td>63.38</td><td>64.02</td><td>36.26</td><td>65.63</td><td>75.90</td><td>39.40</td><td>50.07</td></tr><tr><td>20%</td><td>37.50</td><td>42.43</td><td>27.04</td><td>61.40</td><td>62.16</td><td>35.58</td><td>64.89</td><td>73.29</td><td>40.00</td><td>49.37</td></tr></table>


Table 6: Mamba-based Language Models can be efficiently compressed. Zero-shot evaluation results for the non-compressed (0%) and for the compressed Mamba-based Language Models (20%)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/1a99835f-18ab-4449-94da-363df06bc28e/63853e37fd86069cdaebbc2d071c733054de9fd991f70901ecf5751068311250.jpg)



Figure 3: The joint loss converges generally better. Convergence of the three losses illustrated in Figure 2, evaluated on the Wikitext2 test corpus perplexity during distillation.


architectures: Mamba 3B (Gu and Dao, $2 0 2 4 ) ^ { 5 }$ and Falcon-Mamba $7 \mathrm { B } ^ { 6 }$ . We compressed these models by 20% using the same dataset and hyperparameters as in previous experiments with Transformers, with a minimum rank set to 1024 for both models. Results. Table 6 shows that, as for Transformers, Mamba-based Language Models can be efficiently compressed. Interestingly, the 20% compressed Mamba 3B maintains 99% of the performance. In Section A.4, we also show the compressed Mamba models are lighter and faster. 

## 7.2 Speech modality: Whisper

We evaluate next our approach on a state-of-the-art speech model: whisper-medium.en (Radford et al., 2022), a 764M parameters encoder-decoder speech recognition model trained on English. 

Method. We take inspiration from the deep encoder, shallow decoder approach (Kasai et al., 2021), where the authors observe that the decoder in encoder-decoder models does not need to be very deep to achieve good performance. Therefore, we keep the encoder unchanged and remove 70% of the decoder’s parameters. We then approximate the low-rank matrices using the same method and hyperparameters described in Section 6. However, since Whisper-medium.en is already a model with less than 1 billion parameters, we apply a uniform compression strategy to achieve a 70% reduction in the decoder parameters. 

Data. For calibration, we used 60% of randomly sampled examples from Librispeech-train-100 (Panayotov et al., 2015). We also fine-tuned the compressed model on a subset of 10 hours of speech randomly sampled from Librispeech-train-100. We froze the encoder parameters during this fine-tuning step and trained only the decoder. 

<table><tr><td>Reduction</td><td>Size</td><td>WER (↓)</td><td>VRAM</td><td>Speed (↑)</td></tr><tr><td>0%</td><td>764M</td><td>25.40</td><td>3.5GB</td><td>2395t/s</td></tr><tr><td>37%</td><td>485M</td><td>30.84</td><td>2.4GB</td><td>2717t/s</td></tr><tr><td>0% FT</td><td>764M</td><td>9.20</td><td>3.5GB</td><td>2395t/s</td></tr><tr><td>37% FT</td><td>485M</td><td>12.14</td><td>2.4GB</td><td>2717t/s</td></tr></table>


Table 7: The 37% compressed Whisper is faster, 1.46 lighter. The WER is computed on the English test subset of the Fleurs dataset. The speed (tokens/second) corresponds to the transcription of 400 tokens from a dataset containing 2,048 speech examples, using a batch size of 128. A single A100 GPU is used and the model is loaded in FP32, without Flash-Attention. We also give the GPU Memory (VRAM) occupied by each model.


Results. We evaluate the models’ Word Error Rate (WER), memory footprint and latency on an out-ofdistribution test dataset: the test split of the English Fleurs dataset (Conneau et al., 2022). Table 7 gives the WER for compressed and uncompressed models. We can see that a 37% compressed whisper maintains 82% of the performance. Further finetuning this compressed model on only 10 hours of speech decreases the WER, even outperforming the base non-finetuned model. This compressed model is 31% lighter and 14% faster. 

## 8 Analysis

In this section, we give some ablation studies. In all experiments, the models are compressed for 20%. 

Which distillation loss to use? Table 9 shows that the joint loss depicted in Figure 2c generally produces the best results. Although the differences between the Student and Teacher+Student losses may not seem significant, Figure 3 shows that the Teacher+Student loss leads to faster convergence than the Student loss alone. It also generally results in a lower final perplexity than using the Teacher loss only. We provide the raw values of the plot in Appendix A.3 to better analyze the differences. 

How to choose the rank values? In our bottomfirst compression strategy, a small rank k in Algorithm 1 compresses more severely the bottom layers and leaves more upper layers uncompressed, hence saving GPU memory. However, a small k may also destroy knowledge and the model will struggle to recover during the distillation phase. Figure 4 compares 4 rank values for Mistralv0.1 7B and Phi-3 14B and shows that higher ranks lead to better performance, but are also more costly as more layers are distilled. 

Is SVD initialization necessary? Figure 5 shows the perplexity curves for Phi-2 3B and Phi-3 14b when the low-rank matrices are initialized with SVD or randomly. SVD initialization enables faster convergence, with 8 million tokens being sufficient, i.e., roughly 4k examples of 2048 tokens. 

Cost. All models in Section 6 are compressed in less than one hour on a single A100 GPU. The precise cost depends on the number of layers distilled determined by the compression strategy (bottom, top, or uniform) and the minimum rank k of Algorithm 1: see Appendix A.1 for details. 

Comparisons. Table 8 shows that our approach outperforms SliceGPT (Ashkboos et al., 2024), a one-shot compression method for LLMs. Our approach also compares favorably with SVD-LLM (Wang et al., 2024) in Section A.5. 

## 9 Conclusion

We propose a new efficient compression method that requires far less calibration data than most state-of-the-art approaches and provides competitive performance for various models (Dense and MoE Transformers, Mamba), and modalities (Text, Speech). We show that combining SVD initialization with a joint Teacher and Student loss for local distillation enables fast convergence, and a bottomup layer selection approach enables cost-effective compression. In future work, we plan to study the complementary of our method with quantization, and explore how the performance of compressed models improve with continued pretraining. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/1a99835f-18ab-4449-94da-363df06bc28e/b89d6a3b46de66d34068647d415f937d243b0962c2b04018014ca79d15436ddd.jpg)



Figure 4: Higher ranks give generally better results. Mean accuracy over our test benchmarks as a function of the minimum rank k in Algorithm 1


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/1a99835f-18ab-4449-94da-363df06bc28e/720de4b0317dcd53f6c03d0df269934be04a7f0bfddb826088b3191d0f04c104.jpg)



Figure 5: SVD Initialization improves the sample-efficiency. Convergence when initializing low-rank weights randomly (without SVD) or with SVD.


<table><tr><td>Reduction</td><td>Method</td><td>PIQA</td><td>WinoG</td><td>HellaSwag</td><td>ARC-E</td><td>ARC-C</td><td>Avg.</td></tr><tr><td rowspan="2">0%</td><td>SliceGPT</td><td>79.11</td><td>75.77</td><td>73.83</td><td>78.32</td><td>54.18</td><td>72.24</td></tr><tr><td>Reproduced</td><td>79.11</td><td>76.01</td><td>73.60</td><td>78.41</td><td>54.35</td><td>72.30</td></tr><tr><td rowspan="2">24%</td><td>SliceGPT</td><td>74.05</td><td>62.12</td><td>53.31</td><td>67.26</td><td>39.42</td><td>59.23</td></tr><tr><td>Lillama</td><td>74.76</td><td>69.77</td><td>60.60</td><td>72.10</td><td>47.10</td><td>64.86</td></tr></table>


Table 8: Comparison of our method with SliceGPT on zeroshot task evaluation for Phi-2 3B. We give the evaluation scores before compression (0%), both as reported by SliceGPT and as reproduced by us. Both methods use the Alpaca calibration dataset for compression.


<table><tr><td>Model</td><td>Teacher</td><td>Student</td><td>Tea+Stu</td></tr><tr><td>Phi-2 3B</td><td>57.38</td><td>56.44</td><td>57.38</td></tr><tr><td>Phi-3 14B</td><td>62.54</td><td>63.20</td><td>63.36</td></tr><tr><td>Mistral-v0.1 7B</td><td>51.25</td><td>55.46</td><td>55.51</td></tr><tr><td>Mixtral-8x7B 47B</td><td>58.87</td><td>60.49</td><td>60.19</td></tr><tr><td>Average</td><td>57.51</td><td>58.90</td><td>59.11</td></tr></table>


Table 9: The joint loss generally performs better: Zero-shot performances of the three losses using the same setup for each model, as described in Section 6.


## 10 Limitations

Compatibility with quantization: Our pruning/compression method is theoretically complementary to quantization, since a model compressed with our approach can be further quantized. However, each step removes information, making it crucial to balance these approaches optimally. Addressing this challenge is essential for large-scale adoption in LLM libraries but is beyond the scope of this work and warrants a dedicated study. 

Compromise between specificity and genericity: Efficient LLM compression requires trade-offs between speed and generality. We prioritize generality over raw speed, avoiding GPU-specific kernels as used in semi-structured sparsity approaches. Efficiency also ties to carbon cost: our method works on CPU as well as on GPU, enabling the reuse of existing heritage hardware and not relying on carbon-costly GPU. 

## 11 Acknowledgments

This work was granted access to the HPC resources of IDRIS under the allocation 2024- AD011011668R4 made by GENCI. The authors acknowledge the ANR (French National Research Agency) for its financial support of the LLM4ALL project n°ANR-23-IAS1-0008. 

## References



David Ifeoluwa Adelani, Jessica Ojo, Israel Abebe Azime, Jian Yun Zhuang, Jesujoba O. Alabi, Xuanli He, Millicent Ochieng, Sara Hooker, Andiswa Bukula, En-Shiun Annie Lee, Chiamaka Chukwuneke, Happy Buzaaba, Blessing Sibanda, Godson Kalipe, Jonathan Mukiibi, Salomon Kabongo, Foutse Yuehgoh, Mmasibidi Setaka, Lolwethu Ndolela, Nkiruka Odu, Rooweither Mabuya, Shamsuddeen Hassan Muhammad, Salomey Osei, Sokhar Samb, Tadesse Kebede Guge, and Pontus Stenetorp. 2024. Irokobench: A new benchmark for african languages in the age of large language models. Preprint, arXiv:2406.03368. 





Armen Aghajanyan, Luke Zettlemoyer, and Sonal Gupta. 2020. Intrinsic dimensionality explains the effectiveness of language model fine-tuning. Preprint, arXiv:2012.13255. 





Saleh Ashkboos, Maximilian L. Croci, Marcelo Gennari do Nascimento, Torsten Hoefler, and James Hensman. 2024. SliceGPT: Compress Large Language Models by Deleting Rows and Columns. arXiv preprint. ArXiv:2401.15024 [cs]. 





Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, and Michael Auli. 2020. wav2vec 2.0: A framework for self-supervised learning of speech representations. Preprint, arXiv:2006.11477. 





Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. 2020. Piqa: Reasoning about physical commonsense in natural language. In Thirty Fourth AAAI Conference on Artificial Intelligence. 





Heng-Jui Chang, Shu wen Yang, and Hung yi Lee. 2022. Distilhubert: Speech representation learning by layer-wise distillation of hidden-unit bert. Preprint, arXiv:2110.01900. 





Xuankai Chang, Takashi Maekaku, Pengcheng Guo, Jing Shi, Yen-Ju Lu, Aswin Shanmugam Subramanian, Tianzi Wang, Shu wen Yang, Yu Tsao, Hung yi Lee, and Shinji Watanabe. 2021. An exploration of self-supervised pretrained representations for end-toend speech recognition. Preprint, arXiv:2110.04590. 





Patrick Chen, Hsiang-Fu Yu, Inderjit Dhillon, and Cho-Jui Hsieh. 2021. Drone: Data-aware low-rank compression for large nlp models. In Advances in Neural Information Processing Systems, volume 34, pages 29321–29334. Curran Associates, Inc. 





Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. 2019. Boolq: Exploring the surprising difficulty of natural yes/no questions. ArXiv, abs/1905.10044. 





Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have solved question answering? try arc, the ai2 reasoning challenge. ArXiv, abs/1803.05457. 





Together Computer. 2023. Redpajama: An open source recipe to reproduce llama training dataset. 





Alexis Conneau, Min Ma, Simran Khanuja, Yu Zhang, Vera Axelrod, Siddharth Dalmia, Jason Riesa, Clara Rivera, and Ankur Bapna. 2022. Fleurs: Few-shot learning evaluation of universal representations of speech. Preprint, arXiv:2205.12446. 





Jonathan Frankle and Michael Carbin. 2019. The lottery ticket hypothesis: Finding sparse, trainable neural networks. Preprint, arXiv:1803.03635. 





Elias Frantar and Dan Alistarh. 2023. Sparsegpt: Massive language models can be accurately pruned in one-shot. Preprint, arXiv:2301.00774. 





Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. 2023. A framework for few-shot language model evaluation. 





Albert Gu and Tri Dao. 2024. Mamba: Lineartime sequence modeling with selective state spaces. Preprint, arXiv:2312.00752. 





Song Han, Jeff Pool, John Tran, and William J. Dally. 2015. Learning both weights and connections for efficient neural networks. Preprint, arXiv:1506.02626. 





Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2023. Mistral 7b. Preprint, arXiv:2310.06825. 





Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Théophile Gervet, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2024. Mixtral of Experts. arXiv preprint. ArXiv:2401.04088 [cs]. 





Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li, Fang Wang, and Qun Liu. 2020. TinyBERT: Distilling BERT for Natural Language Understanding. arXiv preprint. ArXiv:1909.10351 [cs]. 





Jungo Kasai, Nikolaos Pappas, Hao Peng, James Cross, and Noah A. Smith. 2021. Deep encoder, shallow decoder: Reevaluating non-autoregressive machine translation. Preprint, arXiv:2006.10369. 





Ayush Kaushal, Tejas Vaidhya, and Irina Rish. 2023. LORD: Low Rank Decomposition Of Monolingual Code LLMs For One-Shot Compression. arXiv preprint. ArXiv:2309.14021 [cs]. 





Yann LeCun, John Denker, and Sara Solla. 1989. Optimal brain damage. In Advances in Neural Information Processing Systems, volume 2. Morgan-Kaufmann. 





Chunyuan Li, Heerad Farkhoor, Rosanne Liu, and Jason Yosinski. 2018. Measuring the intrinsic dimension of objective landscapes. Preprint, arXiv:1804.08838. 





Wing Lian, Guan Wang, Bleys Goodson, Eugene Pentland, Austin Cook, Chanvichet Vong, and "Teknium". 2023. Slimorca: An open dataset of gpt-4 augmented flan reasoning traces, with verification. 





Stephanie Lin, Jacob Hilton, and Owain Evans. 2022. TruthfulQA: Measuring how models mimic human falsehoods. In Proceedings ofthe 60th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 3214–3252, Dublin, Ireland. Association for Computational Linguistics. 





Xin Wang and Yu Zheng and Zhongwei Wan and Mi Zhang. 2024. SVD-LLM: Truncation-aware Singular Value Decomposition for Large Language Model Compression arXiv preprint arXiv:2403.07378. 





James Liu, Pragaash Ponnusamy, Tianle Cai, Han Guo, Yoon Kim, and Ben Athiwaratkun. 2024. Trainingfree activation sparsity in large language models. Preprint, arXiv:2408.14690. 





Jian Liu, Leyang Cui, Hanmeng Liu, Dandan Huang, Yile Wang, and Yue Zhang. 2020. Logiqa: A challenge dataset for machine reading comprehension with logical reasoning. Preprint, arXiv:2007.08124. 





Xinyin Ma, Gongfan Fang, and Xinchao Wang. 2023. Llm-pruner: On the structural pruning of large language models. Preprint, arXiv:2305.11627. 





Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. 2018. Can a suit of armor conduct electricity? a new dataset for open book question answering. In EMNLP. 





Subhabrata Mukherjee, Arindam Mitra, Ganesh Jawahar, Sahaj Agarwal, Hamid Palangi, and Ahmed Awadallah. 2023. Orca: Progressive learning from complex explanation traces of gpt-4. Preprint, arXiv:2306.02707. 





Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur. 2015. Librispeech: An asr corpus based on public domain audio books. In 2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 5206–5210. 





Ankita Pasad, Ju-Chieh Chou, and Karen Livescu. 2022. Layer-wise analysis of a self-supervised speech representation model. Preprint, arXiv:2107.04734. 





Alec Radford, Jong Wook Kim, Tao Xu, Greg Brock man, Christine McLeavey, and Ilya Sutskever. 2022. Robust speech recognition via large-scale weak supervision. Preprint, arXiv:2212.04356. 





Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. 2019. Winogrande: An adversarial winograd schema challenge at scale. arXiv preprint arXiv:1907.10641. 





Maarten Sap, Hannah Rashkin, Derek Chen, Ronan Le Bras, and Yejin Choi. 2019. Social iqa: Com monsense reasoning about social interactions. In Proceedings of the 2019 Conference on Empirical 





Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 4463–4473. 





Shivalika Singh, Freddie Vargus, Daniel Dsouza, Börje F. Karlsson, Abinaya Mahendiran, Wei-Yin Ko, Herumb Shandilya, Jay Patel, Deividas Mataciunas, Laura OMahony, Mike Zhang, Ramith Hettiarachchi, Joseph Wilson, Marina Machado, Luisa Souza Moura, Dominik Krzeminski, Hakimeh´ Fadaei, Irem Ergün, Ifeoma Okoh, Aisha Alaagib, Oshan Mudannayake, Zaid Alyafeai, Vu Minh Chien, Sebastian Ruder, Surya Guthikonda, Emad A. Alghamdi, Sebastian Gehrmann, Niklas Muennighoff, Max Bartolo, Julia Kreutzer, Ahmet Üstün, Marzieh Fadaee, and Sara Hooker. 2024. Aya dataset: An open-access collection for multilingual instruction tuning. Preprint, arXiv:2402.06619. 





Sharath Turuvekere Sreenivas, Saurav Muralidharan, Raviraj Joshi, Marcin Chochowski, Mostofa Patwary, Mohammad Shoeybi, Bryan Catanzaro, Jan Kautz, and Pavlo Molchanov. 2024. Llm pruning and distillation in practice: The minitron approach. Preprint, arXiv:2408.11796. 





Mingjie Sun, Zhuang Liu, Anna Bair, and J. Zico Kolter. 2024. A simple and effective pruning approach for large language models. Preprint, arXiv:2306.11695. 



Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, Léonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ramé, Johan Ferret, Peter Liu, Pouya Tafti, Abe Friesen, Michelle Casbon, Sabela Ramos, Ravin Kumar, Charline Le Lan, Sammy Jerome, Anton Tsitsulin, Nino Vieillard, Piotr Stanczyk, Sertan Girgin, Nikola Momchev, Matt Hoffman, Shantanu Thakoor, Jean-Bastien Grill, Behnam Neyshabur, Olivier Bachem, Alanna Walton, Aliaksei Severyn, Alicia Parrish, Aliya Ahmad, Allen Hutchison, Alvin Abdagic, Amanda Carl, Amy Shen, Andy Brock, Andy Coenen, Anthony Laforge, Antonia Paterson, Ben Bastian, Bilal Piot, Bo Wu, Brandon Royal, Charlie Chen, Chintu Kumar, Chris Perry, Chris Welty, Christopher A. Choquette-Choo, Danila Sinopalnikov, David Weinberger, Dimple Vijaykumar, Dominika Rogozinska,´ Dustin Herbison, Elisa Bandy, Emma Wang, Eric Noland, Erica Moreira, Evan Senter, Evgenii Eltyshev, Francesco Visin, Gabriel Rasskin, Gary Wei, Glenn Cameron, Gus Martins, Hadi Hashemi, Hanna Klimczak-Plucinska, Harleen Batra, Harsh Dhand,´ Ivan Nardini, Jacinda Mein, Jack Zhou, James Svensson, Jeff Stanway, Jetha Chan, Jin Peng Zhou, Joana Carrasqueira, Joana Iljazi, Jocelyn Becker, Joe Fernandez, Joost van Amersfoort, Josh Gordon, Josh Lipschultz, Josh Newlan, Ju yeong Ji, Kareem Mohamed, Kartikeya Badola, Kat Black, Katie Millican, Keelin McDonell, Kelvin Nguyen, Kiranbir Sodhia, Kish Greene, Lars Lowe Sjoesund, Lauren Usui, Laurent Sifre, Lena Heuermann, Leticia Lago, Lilly McNealus, Livio Baldini Soares, Logan Kilpatrick, Lucas Dixon, Luciano Martins, 



Machel Reid, Manvinder Singh, Mark Iverson, Martin Görner, Mat Velloso, Mateo Wirth, Matt Davidow, Matt Miller, Matthew Rahtz, Matthew Watson, Meg Risdal, Mehran Kazemi, Michael Moynihan, Ming Zhang, Minsuk Kahng, Minwoo Park, Mofi Rahman, Mohit Khatwani, Natalie Dao, Nenshad Bardoliwalla, Nesh Devanathan, Neta Dumai, Nilay Chauhan, Oscar Wahltinez, Pankil Botarda, Parker Barnes, Paul Barham, Paul Michel, Pengchong Jin, Petko Georgiev, Phil Culliton, Pradeep Kuppala, Ramona Comanescu, Ramona Merhej, Reena Jana, Reza Ardeshir Rokni, Rishabh Agarwal, Ryan Mullins, Samaneh Saadat, Sara Mc Carthy, Sarah Perrin, Sébastien M. R. Arnold, Sebastian Krause, Shengyang Dai, Shruti Garg, Shruti Sheth, Sue Ronstrom, Susan Chan, Timothy Jordan, Ting Yu, Tom Eccles, Tom Hennigan, Tomas Kocisky, Tulsee Doshi, Vihan Jain, Vikas Yadav, Vilobh Meshram, Vishal Dharmadhikari, Warren Barkley, Wei Wei, Wenming Ye, Woohyun Han, Woosuk Kwon, Xiang Xu, Zhe Shen, Zhitao Gong, Zichuan Wei, Victor Cotruta, Phoebe Kirk, Anand Rao, Minh Giang, Ludovic Peran, Tris Warkentin, Eli Collins, Joelle Barral, Zoubin Ghahramani, Raia Hadsell, D. Sculley, Jeanine Banks, Anca Dragan, Slav Petrov, Oriol Vinyals, Jeff Dean, Demis Hassabis, Koray Kavukcuoglu, Clement Farabet, Elena Buchatskaya, Sebastian Borgeaud, Noah Fiedel, Armand Joulin, Kathleen Kenealy, Robert Dadashi, and Alek Andreev. 2024. Gemma 2: Improving open language models at a practical size. Preprint, arXiv:2408.00118. 





Omkar Thawakar, Ashmal Vayani, Salman Khan, Hisham Cholakkal, Rao Muhammad Anwer, Michael Felsberg, Timothy Baldwin, Eric P. Xing, and Fahad Shahbaz Khan. 2024. Mobillama: Towards accurate and lightweight fully transparent gpt. Preprint, arXiv:2402.16840. 





Atnafu Lambebo Tonja, Bonaventure FP Dossou, Jessica Ojo, Jenalea Rajab, Fadel Thior, Eric Peter Wairagala, Aremu Anuoluwapo, Pelonomi Moiloa, Jade Abbott, Vukosi Marivate, et al. 2024. Inkubalm: A small language model for low-resource african lan guages. arXiv preprint arXiv:2408.17024. 





Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. 2020. Hug gingface’s transformers: State-of-the-art natural language processing. Preprint, arXiv:1910.03771. 





Mengzhou Xia, Tianyu Gao, Zhiyuan Zeng, and Danqi Chen. 2023. Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning. arXiv preprint. ArXiv:2310.06694 [cs]. 





Mengzhou Xia, Tianyu Gao, Zhiyuan Zeng, and Danqi Chen. 2024. Sheared llama: Accelerating language 





model pre-training via structured pruning. Preprint, arXiv:2310.06694. 





Hao Yu and Jianxin Wu. 2023. Compressing Transformers: Features Are Low-Rank, but Weights Are Not! Proceedings of the AAAI Conference on Artificial Intelligence, 37(9):11007–11015. 



## A Appendix

## A.1 On bottom, top, and uniform compression strategies

Figure 6 intuitively illustrates the three compression strategies evaluated in our method: uniform, bottom-first, and top-first, explaining their respective advantages and drawbacks. We present in this section the empirical results of the differences in their efficiency and performance. All the experiments presented here use the same hyperparameters as in Section 6. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/1a99835f-18ab-4449-94da-363df06bc28e/297ce35aa2741c6ea61e5fdc665f8607b6c9863fcfcbbfc8aec5274c8edeb650.jpg)



Figure 6: Illustration of 20% compression when using the three strategies: bottom, top, and uniform. Green boxes are compressed layers with low-rank matrices while gray boxes are non-compressed ones. $l _ { 1 } , l _ { 2 } , . . . ,$ $l _ { n - 1 } , l _ { n }$ are the layer indexes.


Differences in running time and memory. The bottom-first compression strategy allows to compress all models, including Mixtral-8x7B-v0.1 47B. While the top-first strategy can compress Phi-3 14B, the uniform compression strategy fails to compress Phi-3 14B and Mixtral-8x7B-v0.1 (47B). Table 10 gives the distillation time for each strategy and each model. 

Differences in performances. Table 11 presents the mean accuracies for each compression strategy. The bottom-first compression strategy performs well and is also more memory efficient. The top-first strategy does not improve accuracy while having higher time and memory complexity compared to the bottom-first approach. 

<table><tr><td>Model</td><td>Bottom</td><td>Top</td><td>Uniform</td></tr><tr><td>Phi-2 3B</td><td>25min</td><td>32min</td><td>50min</td></tr><tr><td>Mistral-v0.1 7B</td><td>26min</td><td>46min</td><td>105min</td></tr><tr><td>Phi-3 14B</td><td>30min</td><td>50min</td><td>OOM</td></tr><tr><td>Mixtral-8x7B-v0.1 47B</td><td>47min</td><td>OOM</td><td>OOM</td></tr></table>


Table 10: Compression times for the three compression strategies at 20% parameter reduction on a single A100 GPU. For the bottom and top first compression strategies, we used the same hyperparameters as in Section 6.


<table><tr><td>Model</td><td>Bottom</td><td>Top</td><td>Uniform</td></tr><tr><td>Phi-2 3B</td><td>57.38</td><td>55.71</td><td>58.21</td></tr><tr><td>Mistral-v0.1 7B</td><td>55.51</td><td>55.63</td><td>54.25</td></tr><tr><td>Phi-3 14B</td><td>63.59</td><td>60.80</td><td>OOM</td></tr><tr><td>Mixtral-8x7B-v0.1 47B</td><td>60.19</td><td>OOM</td><td>OOM</td></tr></table>


Table 11: Average zero-shot performance for the three compression strategies at 20% compression. For the bottom and top first compression strategies, we used the same hyperparameters as in Section 6.


## A.2 Compression at 20%, 25%, and 30%

To show how performance is affected by different compression ratios, we compressed the models Phi-2 3B, Phi-3 14B, and Mistral-v0.1 7B at 20%, 25%, and 30%. For the 20% compression ratio, we used the same hyperparameters as described in Section 6. For compression ratios >20%, we compressed Phi-3 14B using the bottom-first compression strategy with a minimum rank of k = 2048. For Phi-2 3B and Mistral-v0.1 7B, we applied the uniform compression strategy. As shown in Table 12, our method remains robust under severe compression ratios, with both Phi-3 14B and Phi-2 3B compressed at 30% retaining 93% of their base performance. 

## A.3 Convergence when using Teacher, Student or Teacher+Student loss

As shown in Figure 3, the Teacher+Student loss combines the best of both worlds: fast convergence due to the Teacher activations and high performance at inference thanks to the Student activations. We illustrate this in Table 13 for the first 2M training tokens. It shows that the Teacher+Student loss converges faster than the Student-only loss. While the Teacher-only loss also converges fast, it reaches a higher plateau and does not converge beyond that point, as shown in Figure 3. 

<table><tr><td>Model</td><td>Reduction</td><td>TruQA</td><td>SIQA</td><td>LogiQA</td><td>WinoG</td><td>ARC-E</td><td>ARC-C</td><td>BoolQ</td><td>PIQA</td><td>OBQA</td><td>Average</td></tr><tr><td rowspan="4">Phi-3 14B</td><td>0%</td><td>57.63</td><td>57.06</td><td>37.94</td><td>76.01</td><td>81.36</td><td>61.60</td><td>88.56</td><td>81.34</td><td>50.40</td><td>65.77</td></tr><tr><td>20%</td><td>57.88</td><td>50.26</td><td>34.10</td><td>72.53</td><td>83.54</td><td>59.22</td><td>85.32</td><td>80.30</td><td>49.20</td><td>63.59</td></tr><tr><td>25%</td><td>56.37</td><td>51.38</td><td>32.26</td><td>70.56</td><td>80.98</td><td>57.51</td><td>83.88</td><td>79.76</td><td>47.00</td><td>62.19</td></tr><tr><td>30%</td><td>55.32</td><td>50.41</td><td>31.18</td><td>70.32</td><td>78.70</td><td>55.55</td><td>84.10</td><td>76.99</td><td>45.00</td><td>60.98</td></tr><tr><td rowspan="4">Phi-2 3B</td><td>0%</td><td>44.40</td><td>55.42</td><td>30.57</td><td>76.16</td><td>78.16</td><td>54.18</td><td>83.21</td><td>79.11</td><td>51.20</td><td>61.38</td></tr><tr><td>20%</td><td>46.04</td><td>52.56</td><td>29.80</td><td>71.51</td><td>72.81</td><td>46.50</td><td>75.05</td><td>76.55</td><td>45.60</td><td>57.38</td></tr><tr><td>25%</td><td>44.55</td><td>51.64</td><td>30.88</td><td>71.35</td><td>71.93</td><td>45.99</td><td>79.79</td><td>74.43</td><td>43.20</td><td>57.08</td></tr><tr><td>30%</td><td>44.12</td><td>51.59</td><td>30.41</td><td>70.72</td><td>69.99</td><td>44.37</td><td>78.17</td><td>74.54</td><td>43.20</td><td>56.35</td></tr><tr><td rowspan="4">Mistral-v0.1 7B</td><td>0%</td><td>42.60</td><td>46.57</td><td>29.80</td><td>73.80</td><td>79.55</td><td>53.92</td><td>83.70</td><td>82.10</td><td>44.00</td><td>59.56</td></tr><tr><td>20%</td><td>44.14</td><td>45.50</td><td>28.26</td><td>66.69</td><td>73.11</td><td>46.33</td><td>77.00</td><td>77.37</td><td>41.20</td><td>55.51</td></tr><tr><td>25%</td><td>40.52</td><td>44.93</td><td>30.11</td><td>68.11</td><td>65.07</td><td>40.61</td><td>78.44</td><td>73.99</td><td>40.20</td><td>53.55</td></tr><tr><td>30%</td><td>40.08</td><td>44.37</td><td>30.57</td><td>66.93</td><td>65.53</td><td>40.61</td><td>77.98</td><td>73.18</td><td>39.00</td><td>53.14</td></tr></table>


Table 12: Zero-shot evaluation results for the base non-compressed models (0%), and the 20%, 25%, and 30% compressed models.


<table><tr><td>Model</td><td>Seen Tokens</td><td>Teacher</td><td>Student</td><td>Tea+Stu</td></tr><tr><td rowspan="5">Phi-2 3B</td><td>0</td><td>7185.31</td><td>7185.31</td><td>7185.31</td></tr><tr><td>532480</td><td>20.74</td><td>36.56</td><td>20.34</td></tr><tr><td>1056768</td><td>17.69</td><td>25.11</td><td>17.26</td></tr><tr><td>1559879</td><td>16.00</td><td>18.97</td><td>15.92</td></tr><tr><td>2000506</td><td>15.21</td><td>18.19</td><td>15.51</td></tr><tr><td rowspan="5">Mistral-v0.1 7B</td><td>0</td><td>12365.44</td><td>12365.44</td><td>12365.44</td></tr><tr><td>532480</td><td>43.79</td><td>270.16</td><td>114.02</td></tr><tr><td>1056768</td><td>19.07</td><td>53.90</td><td>21.90</td></tr><tr><td>1579121</td><td>18.56</td><td>22.40</td><td>13.64</td></tr><tr><td>2056609</td><td>18.55</td><td>15.72</td><td>13.11</td></tr><tr><td rowspan="5">Phi-3 14B</td><td>0</td><td>46362.82</td><td>46362.82</td><td>46362.82</td></tr><tr><td>532480</td><td>9.77</td><td>12.94</td><td>8.89</td></tr><tr><td>1056768</td><td>8.33</td><td>8.76</td><td>7.99</td></tr><tr><td>1580799</td><td>7.70</td><td>7.85</td><td>7.44</td></tr><tr><td>2069488</td><td>7.61</td><td>7.77</td><td>7.34</td></tr></table>


Table 13: Comparing the Wikitext perplexity convergence of the three losses for 2M training tokens.


## A.4 Speed up for Mamba architectures

Table 14 shows that compressed Mamba models are also lighter and faster than their original model. 

## A.5 Comparisons with SVD-LLM

We compressed Mistral-v0.1 7B by 25% using the SVD-LLM method<sup>7</sup> and compared it to our approach. We can see in Table A.5 that our approach outperforms SVD-LLM by a large margin. 

## A.6 Activations are more low-rank than weights

We use stable rank (srank) as a proxy measure for rank: 

$$
\operatorname{srank} (X) = \frac {\sum_ {i = 1} ^ {r} \sigma_ {i} ^ {2}}{\sigma_ {1} ^ {2}}
$$

with $X ~ \in ~ \mathbb { R } ^ { d _ { 1 } \times d _ { 2 } }$ being the input matrix (activations or weights), $r = \operatorname* { m i n } ( d _ { 1 } , d _ { 2 } )$ , and $\sigma _ { 1 } \geq$ $\cdots \geq \sigma _ { r }$ the singular values of X. Figure 7 shows the stable rank of various weight matrices and layer activations for three models: Falcon-Mamba, Mistral-v0.1 7b and Phi-2 3b. For all models, activations are significantly lower rank than weight matrices, which confirm previously published results (Yu and Wu, 2023). This also explains why naive SVD solely doesn’t work. 

## A.7 Ease of implementation of our approach

Our proposed pruning approach gives good performance, is cost-efficient, supports various model architectures and modalities, and is further straightforward to implement. Figure 8 shows a simple pseudo-code implementation that can be easily added into most recent PyTorch models. 

## A.8 Example of generated texts

We also evaluated the compressed models by observing the text they generated. In general, at 20% compression of Phi-2 3B, the generation ability and knowledge remained nearly intact. At 40% compression, the model still retained its general knowledge and generation abilities but tended to be more repetitive and verbose at times. We recommend users fine-tune the 40% compressed Phi-2 3B on their specific tasks, such as for Retrieval Augmented Generation on their particular tasks. Table 16 and 17 give some generated texts from 20% and 40% compressed Phi-2 3B. 

Evaluation results. We evaluate the models’ Word Error Rate (WER), memory footprint and latency on an out-of-distribution test dataset: the test split of the English Fleurs dataset (Conneau et al., 2022). Table 7 gives the WER for compressed and uncompressed models. We can see that a 37% compressed whisper maintains 82% of the performance. Further fine-tuning this compressed model on only 10 hours of speech decreases the WER, even outperforming the base model. This compressed model is 31% lighter and 14% faster. 

<table><tr><td>Model</td><td>Reduction</td><td>Model Size</td><td>VRAM</td><td>s = 512</td><td>s = 1024</td><td>s = 2048</td><td>s = 4096</td><td>s = 8192</td><td>s = 16384</td></tr><tr><td rowspan="2">Falcon-Mamba 7B</td><td>0%</td><td>7.27B</td><td>15.3 GB</td><td>8725 t/s</td><td>9346 t/s</td><td>9390 t/s</td><td>9695 t/s</td><td>9908 t/s</td><td>9486 t/s</td></tr><tr><td>20%</td><td>5.82B</td><td>12.6 GB</td><td>10227 t/s</td><td>10427 t/s</td><td>10477 t/s</td><td>10814 t/s</td><td>11108 t/s</td><td>10647 t/s</td></tr><tr><td rowspan="2">Mamba 3B</td><td>0%</td><td>2.77B</td><td>6.7 GB</td><td>17994 t/s</td><td>18556 t/s</td><td>18819 t/s</td><td>19618 t/s</td><td>19231 t/s</td><td>18602 t/s</td></tr><tr><td>20%</td><td>2.22B</td><td>5.6 GB</td><td>19013 t/s</td><td>19993 t/s</td><td>20378 t/s</td><td>21218 t/s</td><td>20808 t/s</td><td>20100 t/s</td></tr></table>


Table 14: Compressing mamba models saves memory and speeds up computation. Inference speed was measured on a single A100 GPU, using the whole test split of Wikitext2 with a batch size of 4 and by varying the sequence length from 512 to 16384 tokens. All models were loaded in bf16 and used Flash-Attention2.


<table><tr><td>Reduction</td><td>Method</td><td>PIQA</td><td>WinoG</td><td>HellaSwag</td><td>ARC-E</td><td>ARC-C</td><td>Avg.</td></tr><tr><td>0%</td><td>Mistral 7B</td><td>82.10</td><td>73.80</td><td>81.04</td><td>79.55</td><td>53.92</td><td>74.08</td></tr><tr><td>25%</td><td>SVD-LLM</td><td>66.27</td><td>62.51</td><td>50.87</td><td>59.68</td><td>33.28</td><td>54.52</td></tr><tr><td>25%</td><td>Lillama</td><td>73.99</td><td>68.11</td><td>67.68</td><td>65.07</td><td>40.61</td><td>63.09</td></tr></table>


Table 15: Comparison of our method with Mistral 7B and SVD-LLM on zero-shot task evaluation for compressed models. Evaluation scores are given for different tasks before and after compression.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/1a99835f-18ab-4449-94da-363df06bc28e/d2dcea85c1bc1431f6b6cf79f2bca2631ac3bf0559ad5988996b4ee367345eb1.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/1a99835f-18ab-4449-94da-363df06bc28e/ccf0d16e935fe3f86ced0211bc65a1138b6013ec2974f31c4b856c45e7fda07c.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/1a99835f-18ab-4449-94da-363df06bc28e/34b05f92f0464b6f95bca5fb11b8fa7c6fab3c2130669b70935d87c2ffb3afa7.jpg)



Figure 7: Activations are low-rank. Comparison of the stable rank of weights (*_proj) and layer activations (layer_activations). Each bar is the average stable rank across all layers.


## A.9 General hyper-parameters

The following general hyper-parameters are used in all our experiments, except when stated otherwise. 

Implementation. We implement our approach using version 4.44.2 of the Huggingface Transformers library (Wolf et al., 2020) and PyTorch version 2.4.0+cu121. For all experiments, we use the AdamW optimizer with a learning rate of 8.6e−<sup>4</sup>. Note that since our approach is local, each student layer has its own optimizer. This also allows us to perform local gradient updates and avoid storing the entire large PyTorch computation graph in memory. All experiments were conducted on a single A100 GPU, regardless of the model. 

Hyperparameters. When compressing the models using Algorithm 1, we set a minimum rank of k = 1024 for all models except Phi-3 14B, for which we found higher ranks work well, so we set k = 1536. For all models, we set m = 256 for the increment to generate ranks. We apply the Teacher+Student loss, as illustrated in Figure 2c, to all models, as we found that this loss converges faster and generally performs better. Additionally, we use the bottom-first compression strategy, which proved to be effective and less memoryintensive than other strategies. Section 6.2 ablates these choices. To measure inference speed, we measure the time to forward a batch of prompts through the entire model and present the results in tokens per second. 

<table><tr><td>Base Model</td><td>-20%</td><td>-40%</td></tr><tr><td>Steve Jobs was a young man who had a passion for technology and design. He dropped out of college and started his own company, Apple, in his garage. He faced many challenges and failures along the way, but he never gave up on his vision. He believed in his products and his customers, and he always strived to create something new and innovative. He was not afraid to take risks and to challenge the status quo. He was a leader who inspired others to follow his example. He was a visionary who changed the world.</td><td>Steve Jobs was a co-founder of Apple, a company that produces computers, software, and other products. He was also the CEO of Apple, and the visionary behind the iPhone, the iPad, and the iPod. He was known for his innovation, creativity, and leadership. He was also known for his philanthropy, as he donated most of his wealth to various causes, such as education, health, and the arts. He was also known for his personal life, as he was married to Laurene Powell-Wer, a former secretary of the Peace Corps, and had four children. He died in 2011 at the age of 56, after suffering from pancreatic cancer.Questions:- What was Steve Jobs' occupation?- What were some of the products that Apple produced? - What were some of the causes that Steve Jobs donated to? - How many children did Steve Jobs have? - When and how did Steve Jobs die?Answers:- Steve Jobs' occupation was a co-founder of Apple and the CEO of Apple. - Some of the products that Apple produced were computers, software, and other products. - Some of the causes that Steve Jobs donated to were education,</td><td>- He was born in California in 1955.- He dropped out of college and started working at a computer company.- He became the co-founder of Apple.- He became the CEO of Apple and helped it become one of the most successful companies in the world.- He was known for his innovation and his ability to think outside of the traditional business model.- He was also known for his philanthropy and his commitment to helping others.- He died in 2011.</td></tr><tr><td>The Civil War was caused by a number of factors, including economic and social differences between the North and the South, as well as disagreements over states' rights and slavery. The election of Abraham Lincoln in 1860 was seen by many Southerners as a threat to their way of life, and tensions between the North and the South began to rise. The war began in 1861 when Confederate forces attacked Fort Sumter in South Carolina. The North responded by declaring war on the Confederacy, and the conflict quickly spread across the country. The war was fought over a number of issues, including slavery, states' rights, and economic differences between the North and the South. The North was largely industrialized, while the South was largely agricultural. This led to tensions over tariffs and other economic issues. The war was also fought over the issue of slavery. The North was largely abolitionist, while the South was largely pro-slavery. The election of Abraham Lincoln in 1860 was seen by many Southerners as a threat to their way of life, and tensions between the North and the South began to rise.</td><td>The American Civil War was a conflict that took place from 1861 to 1865. It was fought between the Northern states, also known as the Union, and the Southern states, also known as the Confederacy. The main cause of the war was the issue of slavery. The Southern states wanted to keep slavery, while the Northern states wanted to abolish it. This led to tensions and eventually led to the war.What were the major battles of the Civil War? Here is the story.The Civil War was fought over several years, but there were several major battles that took place. The most famous battles were the Battle of Gettysburg, the Battle of Antieta, and the Battle of Bull Run. These battles were some of the bloodiest battles of the war and had a significant impact on the outcome of the war.What was the Emancipation Proclamation? Here is the story.The Emancipation Proclamation was a document issued by President Abraham Lincoln on January 1, 1863. It declared that all slaves in Confederate territory were to be set free. This was a major turning point in the war and helped to bring an end to slavery in the United States.</td><td>The Civil War was a conflict between the United States and the Confederate States, which lasted from 1861 to 1865. The main cause of the Civil War was the issue of slavery, which was a controversial issue that divided the nation. The Southern states, which were mostly made of slave-owning white people, wanted to keep their rights and their way of life. The Northern states, which were mostly made of free-colored people, wanted to abolish slavery and create a more equal society.The Civil War began when the Confederate States declared their secession from the United States, which means they wanted to break away from the country. The United States responded by sending troops to the South, which started the war. The war lasted for four years, and it was one of the most important events in American history. The war ended with the defeat of the Confederate States, and the United States became a unified nation.The Civil War had many consequences, both positive and negative. Some positive consequences were that the United States became more united and more democratic, and that the issue of slavery was finally resolved. Some negative consequences were that many people died, and that the war caused a lot of damage and destruction.</td></tr></table>


Table 16: Steve Jobs Story by the 20% and 40% compressed Phi-2 3B models, without any finetuning. Prompt: "Here the story of Steve Jobs:"


Table 17: The Civil War Story by the 20% and 40% compressed models, without any finetuning. Prompt: "What is the cause of the Civil War? Here is the story." 

```python
def low_rank_layer(layer, ranks):
    for name, module in layer.named_modules():
    if isinstance(module, torch.nn.Linear):
    # get the rank of this matrix, as computed with Algorithm 1
    r = ranks[name]
    # create the Up-Down lines for the low-rank matrices
    l1 = torch.nn.Linear(module.in_features, r, bias=False)
    l2 = torch.nn.Linear(r, module.out_features, bias=False)
    # initialize the Low-Rank matrices with SVD
    u, s, v = torch.linalg.svd(module.weight)
    w1 = torch.diag(s)[:r, :r] @ v[:r, :]
    w2 = u[:, :r]
    l1.weight = torch.nn.Parameter(w1)
    l2.weight = torch.nn.Parameter(w2)
    # replace the old linear layers with the low-rank linear layer
    low_rank_linear = torch.nn.Sequential(l1, l2)
    setattr(layer, name, low_rank_linear)

# 'ranks' are the rank values computed using Algorithm 1
# 'base_layer' is the uncompressed pretrained layer.
compressed_layer = low_rank_layer(copy(base_layer), ranks)
optimizer = torch.optim.AdamW(compressed_layer.parameters())

def feature_approximation_hook(module, inputs, outputs):
    optimizer.zero_grad()
    outputs_ = compressed_layer(inputs)
    loss = loss_fn(outputs_, outputs)
    loss.backward()
    optimizer.step()

# attach the function to the base layer as a forward hook
# so the function is called at each forward call of the base layer
base_layer.register_forward_hook(feature_approximation_hook) 
```


Figure 8: An example of PyTorch implementation of our approach with the Teacher loss (see Figure 2a).


## A.10 Metrics used when evaluating with lm-evaluation-harness

We provide in Table 18 the metrics we used to evaluate the models in Section 6. 

<table><tr><td>Task</td><td>Metric</td></tr><tr><td>arc_challenge</td><td>acc_norm</td></tr><tr><td>arc_easy</td><td>acc_norm</td></tr><tr><td>piqa</td><td>acc_norm</td></tr><tr><td>social_iqa</td><td>acc</td></tr><tr><td>logiqa</td><td>acc_norm</td></tr><tr><td>truthfulqa_mc2</td><td>acc</td></tr><tr><td>winogrande</td><td>acc</td></tr><tr><td>boolq</td><td>acc</td></tr><tr><td>openbookqa</td><td>acc_norm</td></tr></table>


Table 18: Benchmarks and corresponding metrics that were used to evaluate the models in Section 6.


## A.11 A Tiny Model for a Low-Resource Language

For most human languages, there is not enough data (generally less than 1B tokens) to pretrain large LLMs. In such cases, data-efficient approaches are preferred. We experimented compressing a tiny language model for Hausa, a low-resource language for which there is not billions of tokens available for continued pretraining. We used InkubaLM (Tonja et al., 2024), a 422M parameter language model designed for five low-resource languages. Since 60% of the parameters are in the embeddings, we focused on compressing the input embedding $W _ { e }$ and the prediction head $W _ { o }$ This is easily achieved with our approach by locally distilling the low-rank embeddings and prediction head: 

$$
\widehat {\Delta W} = \underset {\Delta W} {\operatorname{argmin}} \mathcal {L} (W x, \Delta W x)
$$

where $\Delta W$ are the low-rank embedding matrices or prediction head, and x is an input example. 

We used a rank of 1024, compressing the model by 30%. Then, we applied a lightweight local distillation using our approach, with 64k randomly sampled Hausa sentences from InkubaMono<sup>8</sup> and the Aya-Dataset (Singh et al., 2024). We compared the models with MobiLLaMA (Thawakar et al., 2024) and Smo $\mathbf { { \mathbf { { L M } } } ^ { 9 } }$ using the Afrimmlu benchmark (Adelani et al., 2024). As shown in Table 19, the compressed InkubaLM model for Hausa retains 93% of its base performance. 

<table><tr><td>Model</td><td>Model Size (Billion)</td><td>Accuracy</td></tr><tr><td>SmoLLM</td><td>1.7</td><td>21.80</td></tr><tr><td>MobiLLaMA</td><td>1.3</td><td>21.40</td></tr><tr><td>InkubaLM-base</td><td>0.422</td><td>29.20</td></tr><tr><td>InkubaLM-30%</td><td>0.299</td><td>27.40</td></tr></table>


Table 19: Zero-shot performance on afrimmlu for the 30% compressed InkubaLM model for the Hausa language.
