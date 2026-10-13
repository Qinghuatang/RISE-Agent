# BOOSTING ENTROPY WITH BELL BOX QUANTIZATION

Ningfeng Yang University of British Columbia nxyang@ece.ubc.ca 

Tor M. Aamodt University of British Columbia aamodt@ece.ubc.ca 

## ABSTRACT

Quantization-Aware Pre-Training (QAPT) is an effective technique to reduce the compute and memory overhead of Deep Neural Networks while improving their energy efficiency on edge devices. Existing QAPT methods produce models stored in compute-efficient data types (e.g. integers) that are not information theoretically optimal (ITO). On the other hand, existing ITO data types (e.g. Quantile/NormalFloat Quantization) are not compute-efficient. We propose BBQ, the first ITO quantization method that is also compute-efficient. BBQ builds on our key insight that since learning is domain-agnostic, the output of a quantizer does not need to reside in the same domain as its input. BBQ performs ITO quantization in its input domain, and returns its output in a compute-efficient domain where ITO data types are mapped to compute-efficient data types. Without sacrificing compute efficiency, BBQ outperforms prior SOTA QAPT methods by a perplexity reduction of up to 2 points for 4-bit models, up to 4 points for 3-bit models, up to 5 points for 2-bit models, and up to 18 points for 1-bit models. Code is available at https://github.com/1733116199/bbq. 

## 1 INTRODUCTION

Quantization is an effective method to reduce the computation/memory/energy consumption of Deep Neural Networks (DNNs), allowing DNNs to be deployed to edge devices with limited hardware resources. However, quantization often degrades model quality. While Post-Training Quantization (PTQ) methods (Lin et al., 2024; Shao et al., 2024; Ma et al., 2024b; Liu et al., 2024a; Kim et al., 2024) can mitigate quality degradation without re-training for higher precisions (4+-bit), they struggle to maintain quality when weights and activations are quantized to 4-bit and below (Panferov et al., 2025; Esser et al., 2020; Ma et al., 2024a; Kumar et al., 2025). Quantization-Aware Training (QAT) methods (Panferov et al., 2025; Esser et al., 2020; Ma et al., 2024a; Kumar et al., 2025; Shen et al., 2024; Liu et al., 2025; Yamamoto, 2021), on the other hand, can achieve higher accuracy than PTQ methods under the same precision (Du et al., 2024; Panferov et al., 2025; Liu et al., 2024b) by introducing quantization in the training loop. 

QAT can be further divided into Quantization-Aware Pre-Training (Panferov et al., 2025; Yang & Aamodt, 2025) (QAPT) and Quantization-Aware Fine-Tuning (Malinovskii et al., 2024; Du et al., 2024) (QAFT). QAFT initializes a low-precision model from a full-precision pre-trained checkpoint, and trains the model for a short duration to fit a downstream task, which is typically much smaller than the pre-training dataset (Du et al., 2024). On the other hand, QAPT initializes a low-precision model from scratch, aims to fit a much larger dataset, and typically trains for much longer durations. Compared to first pre-training in full-precision and subsequently applying PTQ/QAFT, QAPT may have higher pre-training speed (Kumar et al., 2025; Xi et al., 2023; Castro et al., 2025) as the forward pass of training may be performed in low precision. 

This work aims to improve the accuracy of QAPT without compromising the resulting model’s efficiency on edge devices. Specifically, we target the case when a low-precision model is initialized randomly, trained for long durations on large datasets, and is intended to be deployed to edge devices with constraints on memory capacity, inference latency, and energy consumption. The energy constraint requires the model to be compute-efficient, or that expensive high-precision matrix multiplications be substituted with low-precision arithmetic, while the memory and latency constraints require the model to be small in size, or to have a low parameter count and precision. A main challenge of QAPT is that models with limited memory footprint cannot fit large datasets well due to a lack of learning capacity (Kumar et al., 2025). 

<table><tr><td>Signed INT4</td><td>-8, ±7, ±6, ±5, ±4, ±3, ±2, ±1, 0</td></tr><tr><td>MX FP4</td><td>±6, ±4, ±3, ±2, ±1.5, ±1, ±0.5, ±0</td></tr></table>


Table 1: Possible values of the INT4 and the MX FP4 data type.


Using the Shannon entropy (Shannon, 1948; Cover & Thomas, 2006) of quantized weights as a proxy for the amount of information/knowledge present in a model trained with QAPT, we observe that SOTA QAPT methods QuEST (Panferov et al., 2025) and LSQ (Esser et al., 2020) produce quantized models that under-utilize the available learning capacity. The under-utilization is because LSQ and QuEST use compute-efficient data types (e.g. integers and floats) which are not information theoretically optimal (ITO). While existing ITO data types (Dettmers et al., 2023; 2022) can reduce this under-utilization of learning capacity, they lack compute efficiency on modern CPUs/GPUs, which limits their applicability on edge devices with limited energy. 

To maximally utilize the limited learning capacity while preserving compute efficiency, we propose Bell Box Quantization (BBQ). BBQ is designed based on our key insight: since learning is domain-agnostic, the output of a quantizer does not need to live in the same domain as its input. BBQ performs ITO quantization in its input domain to maximally preserve information, but returns its output in a different domain, where ITO data types are mapped to compute-efficient data types. BBQ achieves higher capacity utilization and better prediction quality than QuEST and LSQ. Unlike existing ITO data types, a BBQ-quantized model can still accelerate expensive matrix multiplications with low-precision arithmetic. 

## 2 BACKGROUND AND MOTIVATION

In this section, we discuss compute-efficient data types on modern GPUs, existing ITO quantization methods, and lastly, the domain-agnostic property of learning which is the key inspiration of BBQ. 

Quantization is the discretization of a continuous random<sup>1</sup> variable x to a discrete random variable xˆ, such that xˆ is sufficiently close to x. We present a generic template for quantization as follows: 

$$
\hat {x} = f ^ {- 1} (q), \text {   and   } q = r (f (x))\tag{1}
$$

where $f : \mathbb { R } ^ { d }  \mathbb { R } ^ { d }$ is a transform function that converts $x \in \mathbb { R } ^ { d }$ to an intermediate domain suitable for quantization; $r : \mathbb { R } ^ { d }  \mathbb { T } ^ { d }$ is a rounding function that takes real-valued inputs and returns outputs $q$ in a set of $2 ^ { b }$ real values, i.e. $q \in \mathbb { T } ^ { d } , \mathbb { T } \subseteq \mathbb { R }$ , and $\left| \mathbb { T } \right| = 2 ^ { b }$ ; and $\scriptstyle { \dot { \boldsymbol { f } } } ^ { - 1 }$ , the inverse function of $f ,$ is responsible for converting the rounded variable $q$ back to the original domain of x. Throughout this work, x, q, and xˆ represent neural network weights or activations that are raw, quantized, and de-quantized, respectively. 

## 2.1 COMPUTE EFFICIENT DATA TYPES

A data type T is compute-efficient if hardware offers low-precision multiply-accumulate instructions that directly operate on members of T without first decoding to high-precision data types. Table 1 shows members of $\mathbb { T }$ for INT4, which is supported by the Nvidia Ampere (Nvidia, 2021) and Turing (Nvidia, 2018) architectures, and members of T for MX FP4 (Rouhani et al., 2023), which is supported by the Blackwell architecture (Nvidia, 2025). When $f ^ { - 1 }$ is linear or affine, a computeefficient data type can be used to accelerate matrix multiplication and convolution (Yao et al., 2021). For example, if $f ^ { - 1 } ( q ) = s q$ for a constant scalar s, then the multiplication of activation matrix $\hat { X } = f ^ { - 1 } ( Q _ { x } ) = s Q _ { \ l }$ <sub>x</sub> and weight matrix $\hat { W } = f ^ { - 1 } ( Q _ { w } ) = s Q _ { w }$ can be simplified to $s ^ { 2 } ( Q _ { x } \cdot Q _ { w } )$ In other words, the matrix multiplication can be computed using solely low-precision representations of activations and weights, and later de-quantized to the original domain by multiplication of $s ^ { 2 }$ 

Prior SOTA QAPT methods QuEST (Panferov et al., 2025) and LSQ (Esser et al., 2020) use linear/affine $f ^ { - 1 }$ and compute-efficient data types. We show the definition of $f ,$ r, and $f ^ { - 1 }$ for LSQ 

and QuEST as follows: 

$$
\begin{array}{l} \text { LSQ: } f (x) = x / s, r (v) = \lfloor \operatorname{clip} (v, - 2 ^ {b - 1}, 2 ^ {b - 1} - 1) \rceil , f ^ {- 1} (q) = s q \\ \text { QuEST: } f (x) = \frac {\mathrm{HT} (x)}{\alpha^ {*} \sigma} - \frac {1}{2}, r (v) = \lfloor \operatorname{clip} (v, - 2 ^ {b - 1}, 2 ^ {b - 1} - 1) \rceil , f ^ {- 1} (q) = \mathrm{HT} (\alpha^ {*} \sigma (q + \frac {1}{2})) \end{array}\tag{2}
$$

where $s , \sigma ,$ and $\alpha ^ { * }$ are scalars; all division operations are element-wise; HT is the Hadamard transform, a linear operation; and the clip operation and the round-to-nearest-integer operation ⌊·⌉ enforce the output of r to be a b-bit signed integer. Since $f ^ { - 1 }$ is linear or affine in LSQ and QuEST, matrix multiplications can be accelerated with low-precision arithmetic. 

In short, a compute-efficient data type T and a linear/affine dequantization function $f ^ { - 1 }$ are two key features that enable quantization methods like LSQ and QuEST to run efficiently on modern GPUs. 

## 2.2 INFORMATION THEORETICALLY OPTIMAL QUANTIZATION METHODS

An ITO quantization method is one that ensures each member of T is used equally often (Dettmers et al., 2023). For example, if $b = 2 ,$ , then 25% of elements of x should be assigned to each of the 4 possible values in T. While there are many possible mappings that evenly divide the elements of x, a trivial mapping is to use the 25th, the 50th, and the 75th percentiles as rounding boundaries. In other words, elements of x less than the 25th percentile of x are rounded to the smallest value in T; elements larger than the 25th percentile but less than the 50th percentile are rounded to the second smallest value in T, etc. If all d elements of x are i.i.d. sampled from $N ( 0 , \sigma ^ { 2 } )$ and d is sufficiently large, this trivial mapping can be expressed as: 

$$
f (x) = x / \sigma , r (v) = T [ \mathrm{floor} (2 ^ {b} \Phi (v)) ], f ^ {- 1} (q) = \sigma q\tag{3}
$$

where $T [ i ]$ is the ith smallest element in T. Equation 3 only requires that T contains $2 ^ { b }$ unique real values, regardless of what these real values may be. As such, Equation 3 describes a set of ITO methods rather than a unique one. By evenly dividing the elements of $x ,$ all ITO data types maximize the empirical Shannon entropy of q defined as 

$$
H (q) = \sum_ {t \in \mathbb {T}} - P (q, t) \log_ {2} P (q, t)\tag{4}
$$

where $P ( \boldsymbol { q } , t )$ is the probability that a (uniformly) randomly chosen element of q is equal to t. Note that entropy was shown to positively correlate with model accuracy/prediction quality (Cheng et al., 2025; Shen et al., 2024). 

Inspired by ITO quantization methods, Dettmers et al. (2023) proposed NF4, a 4-bit NormalFloat whose values are a list of abs-max-normalized Gaussian quantiles (Yoshida, 2023; Dotzel et al., 2024). Due to lack of hardware support, NF4 must be de-quantized to full-precision floats and then used in computation, which limits its applicability on energy-constrained edge devices. 

This presents a dilemma: while ITO quantization methods maximally preserve information, they are not compute-efficient; compute-efficient quantization methods, on the other hand, are not ITO for Gaussianly distributed weights/activations. Can we obtain the best of both worlds? 

## 2.3 LEARNING IS DOMAIN-AGNOSTIC

Since neural networks are universal function approximators (Hornik et al., 1989), they are capable of learning from transformed/augmented data, or even data encoded in latent spaces that are not humanunderstandable. To list a few examples: DNNs can correctly classify images that are rotated (He et al., 2015); DNNs can classify images that are transformed to the frequency domain (Xu et al., 2020); When autoencoders (Gao et al., 2025) are trained just to reconstruct the data, the resulting latent embedding of the data can be used by other DNNs to complete downstream tasks. These examples are all evidence that, as long as information is preserved, simply projecting/transforming data to a different domain does not prevent learning. 

The domain-agnostic property of learning provides an opportunity to circumvent the inefficiency of ITO quantization methods. Rather than treating quantizers as data compressors and reconstructors, we could treat them as feature extractors that return a set of compact latent features of the original data. Critically, these latent features reside in an output domain, that is, not necessarily the same domain as the input, but is designed to be a compute-efficient domain, where features can be efficiently used by matrix multiplication operators. While N2UQ (Liu et al., 2022) also returns outputs in a compute-efficient but distinct-from-input domain, BBQ is the first method to apply such a technique to ITO quantization for both activations and weights. In addition, N2UQ assumes raw weights are uniformly distributed, while BBQ is ITO without making any assumptions on the distribution shape of raw weights or raw activations. In Section 4, we quantitatively compare BBQ against N2UQ, showing BBQ achieves lower perplexity under the same precision. Inspired by the domain-agnostic property of learning, we ask the research question: During QAPT, can ITO quantization outperform non-ITO methods ifwe return outputs in a compute-efficient domain? 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/67335314-6215-4a10-a04e-e66551ddf124/5f912d306ad0f4fe5eed453d314569e5d55435f500b507932f1f8db59493d053.jpg)



Weights or Activations


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/67335314-6215-4a10-a04e-e66551ddf124/46964025ef3342c5ebbdc005613429138ea5c18af122f6649fdccac99b43dad0.jpg)



(a) HT & RMS


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/67335314-6215-4a10-a04e-e66551ddf124/12e20287cb29c1b7cf42a64415d409887f1461e173159b0ad7bd868967167af6.jpg)



(b) PIT


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/67335314-6215-4a10-a04e-e66551ddf124/2c1c941143f54e48d61cd3f60861ed11fede8c2584dbcaa7bdbe7ce0678e5863.jpg)



(c) Uniform Quantization



Figure 1: The three steps of the BBQ quantization formula (Equation 5) for b = 4. Step a is the Hadamard Transform (HT) followed by RMS Normalization. Step b is the probability integral transform (PIT). Step c is uniform quantization. We name our method Bell Box Quantization because Figure 1a looks like a bell and Figure 1b looks like a box (rectangle) which is quantized in Figure 1c.


## 3 BELL BOX QUANTIZATION

In this section, we present the Bell Box Quantizer (BBQ). BBQ is designed to be ITO, or to maximally preserve information from its input x (activations or weights), while also returning computeefficient outputs that can be accelerated by modern hardware. The BBQ quantization and dequantization formulas are shown in Equations 5 and $^ { 6 , }$ respectively: 

$$
q = \left\lfloor 2 ^ {b} \Phi (v) \right\rfloor - 2 ^ {b - 1} - z, \text { where } v = \mathrm{HT} (x) / \sigma\tag{5}
$$

$$
\hat {x} = \frac {\gamma}{2 ^ {b - 1}} q\tag{6}
$$

where HT $\colon \mathbb { R } ^ { d }  \mathbb { R } ^ { d }$ is the Hadamard transform operation; the root-mean-square (RMS) normalization factor, $\sigma ,$ is defined as $\sigma = ( 1 / d \cdot \textstyle \sum _ { i = 1 } ^ { d } ( \mathrm { H T } ( x ) _ { i } ) ^ { 2 } ) ^ { 1 / 2 } ;$ v is a variable introduced for notational convenience; $\Phi : \mathbb { R } ^ { d }  ( 0 , 1 ) ^ { d }$ is the element-wise Gaussian CDF operation; $\lfloor \cdot \rfloor$ is the element-wise floor operation; $z \in \{ 0 , - 0 . 5 \}$ is a hyperparameter zero point; and $\gamma$ is a learnable scaling factor. Figure 1 illustrates the three major steps of Equation 5: Hadamard Transform combined with RMS Normalization (step a), probability integral transform (step b), and uniform quantization (step c). We discuss steps a, b, and c in Sections 3.1, 3.2, and 3.3, respectively, and the dequantization formula Equation 6 in Section 3.4. 

## 3.1 STEP A: HADAMARD TRANSFORM AND RMS NORMALIZATION (FIGURE 1A)

Since ITO quantization is ill-defined without knowing a distribution, the first step of BBQ is to convert its input x from an unknown distribution to a Gaussian by applying the Hadamard Transform (Panferov et al., 2025; Yang et al., 2025). After the transform, we assume elements of HT(x) behave like samples from $N ( 0 , \overleftarrow { \boldsymbol { \sigma } ^ { 2 } } )$ following Panferov et al. (2025) and Yang et al. (2025). Instead of transforming all elements of $x ,$ we follow QuEST and perform the Hadamard Transform on every H elements of x along the input channel dimension. During training, the Hadamard Transform is a differentiable operation, and we simply use autograd frameworks to perform its backward pass. During inference, the Hadamard transform is implemented with a vector-matrix multiplication between an H-element slice of x and the pre-computed $H \times H$ Hadamard matrix. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/67335314-6215-4a10-a04e-e66551ddf124/9aef75f51e824a6bba8501610e6fc767d75f58f376215555091f8c5e6a806112.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/67335314-6215-4a10-a04e-e66551ddf124/f0486e61f3a325b07213fd15518aeda88963fedd098ef1ef12fb6218f7b79772.jpg)



(a) Graph of clip and Φ. Both functions take input in $( - \infty , \infty )$ and return output in a finite range, and therefore both qualify as clipping functions.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/67335314-6215-4a10-a04e-e66551ddf124/db57afff9f0bf10f27cf2f9842a54efe1132ec03f8b30f143db5768c19fe5854.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/67335314-6215-4a10-a04e-e66551ddf124/088c9432c471ed3ea2bdc408ddad964eabc48171b5c7ffbcadd8bbf29b80c1b9.jpg)



φ(v)



(b) PDF of clip(v) and Φ(v) for v ∼ N(0, 1). Φ(v) has maximum (differential) entropy as its probability mass is evenly spread out.



Figure 2: Comparison of clip (blue) and the standard Gaussian CDF Φ (orange).


Next, we apply RMS normalization by multiplying the Gaussian-like data by $1 / \sigma$ . As a result, elements of $\boldsymbol { v } = \mathrm { H T } ( \boldsymbol { x } ) / \sigma$ behave like samples from $N ( 0 , 1 )$ . During training, we use autograd frameworks to perform the backward pass of RMS Normalization. Following standard quantization practice (Nagel et al., 2021), we perform channel-wise RMS normalization for weights, and pertensor RMS normalization for activations. 

We discuss an optional variant of BBQ, named BBQ-Fast, which achieves identical perplexity (Section A.3) but has better inference speed. During training, BBQ-Fast has identical behaviour to BBQ, except BBQ-Fast additionally keeps track of the exponential moving average of $1 / \sigma$ , denoted as $E _ { 1 / \sigma }$ , using the update rule in Equation 7, which is executed every training iteration with $\beta = 0 . 9 9$ 

$$
E _ {1 / \sigma} \leftarrow \beta \cdot E _ {1 / \sigma} + (1 - \beta) \cdot (1 / \sigma)\tag{7}
$$

During inference, BBQ-Fast performs activation RMS normalization by multiplying $\operatorname { H T } ( x )$ by the final value of $E _ { 1 / \sigma } ,$ instead of by $1 / \sigma ,$ , since measuring σ requires the execution of the root-meansquare operation, which often incurs expensive cross-thread-block communication for large activation tensors. 

## 3.2 STEP B: PROBABILITY INTEGRAL TRANSFORM (FIGURE 1B)

The probability integral transform (David & Johnson, 1948; Fisher, 1932) converts any continuous distribution to a uniform one by applying its CDF on the data. We apply the standard Gaussian CDF Φ to v which behaves like samples from the standard Gaussian $N ( 0 , 1 )$ , creating $\Phi ( v )$ which behaves like samples from $U ( 0 , 1 )$ . Function Φ is designed to clip data within a finite range (Figure 2a) while maximizing entropy (Figure 2b) for Gaussianly distributed data, and is meant to replace the vanilla clip function in QuEST and LSQ. Figure 2a shows that Φ is much smoother than clip. This is because Φ is infinitely differentiable and clip is piecewise linear. A smoother operation can be more suitable (Nesterov, 2014; Bottou et al., 2018) for optimization methods like Gradient Descent. For example, the GELU function (Hendrycks & Gimpel, 2023) defined as $\mathrm { G E L U } ( x ) = x \cdot \Phi ( x )$ is smoother than the piecewise linear ReLU (Agarap, 2018), and GELU can outperform ReLU empirically (Hendrycks & Gimpel, 2023). During training, Φ is a differentiable operation and we use the autograd framework to perform BackProp. During inference, we combine Φ with the subsequent floor operation using the following property, which is true $\forall v \in \mathbb { R } , \forall i \in \{ 0 , 1 , 2 , . . . 2 ^ { b } - 1 \}$ 

$$
\lfloor 2 ^ {b} \Phi (v) \rfloor = i \iff \Phi^ {- 1} \left(\frac {i}{2 ^ {b}}\right) \leq v <   \Phi^ {- 1} \left(\frac {i + 1}{2 ^ {b}}\right)\tag{8}
$$

Property 8 allows us to pre-compute $\Phi ^ { - 1 } ( i / 2 ^ { b } )$ for all $2 ^ { b }$ possible values of $^ { \cdot } i ,$ and at runtime perform a binary search that requires b floating point comparisons. Our profiling results in Section 4.2 show this binary search incurs negligible runtime overhead. The pseudocode of a 3-bit inference kernel implementation of BBQ is shown in Algorithm 1. 

## 3.3 STEP C: UNIFORM QUANTIZATION (FIGURE 1C)

After applying the Probability Integral Transform, since $\Phi ( v )$ behaves like samples from $U ( 0 , 1 )$ , we can achieve ITO quantization by applying uniform quantization, obtaining $\lfloor \dot { 2 } ^ { b } \Phi ( v ) \rfloor$ which can be stored as a b-bit unsigned integer. Since zero-mean activations can contribute to faster convergence (Bjorck et al., 2018; Ioffe & Szegedy, 2015), we convert the non-negative data $\lfloor 2 ^ { b } \Phi ( v ) \rfloor$ to symmetric data by subtracting $2 ^ { b - 1 }$ and the hyperparameter zero point $z ,$ producing the final $q = \dot { [ 2 ^ { b } \Phi ( v ) ] } - 2 ^ { b - \bar { 1 } } - z$ stored as a signed compute-efficient data type. Inspired by NF4 (Dettmers et al., 2023), we use $z = 0$ for $b \in \{ 3 , 4 \}$ , allowing the data type to represent zero exactly while sacrificing a value on the positive side. We use $z = - 0 . 5$ for $b \in \{ 1 , 2 \}$ to ensure activations remain zero-mean. Table 2 lists the values supported by the BBQ data type. For $b \in \{ 1 , 2 , 3 \}$ , BBQ can be represented using the MX FP4 data type on existing Blackwell GPUs. For $b \in \{ 3 , 4 \}$ }, BBQ can be represented using INT4 on existing Ampere and Turing GPUs. During training, we use the Straight-Through Estimator (Bengio et al., 2013) for the floor operation, and the autograd framework for all other differentiable operations. 

Algorithm 1 Example 3-bit Inference Quantization Kernel for a Thread Block
1: function BBQ(block_id, x_ptr, h_ptr, q_ptr, $E_{1/\sigma}$ )
2:    x = x_ptr[block_id:block_id+1, :] # x_ptr is an M × N matrix reshaped to (MN/H, H)
3:    h = h_ptr[:, :] # h_ptr is a pre-computed Hadamard matrix with shape (H, H)
4:    xh = x @ h # Step a: vector-matrix multiplication between shapes (1, H) and (H, H)
5:    v = xh · $E_{1/\sigma}$ # Step a: element-wise multiplication by pre-computed scalar (BBQ-Fast)
6:    q = an array with H INT4/MX FP4 elements
7:    for t ← 0 to H - 1 do # can be done in parallel across warps and threads
8:    if v[t] ≥ Φ $^{-1}$ (4/8) then # Step b: Binary search with pre-computed Φ $^{-1}$ values
9:    if v[t] ≥ Φ $^{-1}$ (6/8) then
10:    q[t] = 3 if v[t] ≥ Φ $^{-1}$ (7/8) else 2 # Step c: quantize to INT4/MX FP4
11:    else
12:    q[t] = 1 if v[t] ≥ Φ $^{-1}$ (5/8) else 0 # Step c: quantize to INT4/MX FP4
13:    end if
14:    else
15:    if v[t] ≥ Φ $^{-1}$ (2/8) then
16:    q[t] = -1 if v[t] ≥ Φ $^{-1}$ (3/8) else -2 # Step c: quantize to INT4/MX FP4
17:    else
18:    q[t] = -3 if v[t] ≥ Φ $^{-1}$ (1/8) else -4 # Step c: quantize to INT4/MX FP4
19:    end if
20:    end if
21:    end for
22:    q_ptr[block_id:block_id+1, :] = q # store quantized INT4/MX FP4 results to global memory
23: end function 

<table><tr><td>Precision b</td><td>Zero Point z</td><td>Possible Values of q</td><td>INT4</td><td>MX FP4</td></tr><tr><td>4</td><td>0</td><td>-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7</td><td>√</td><td>✕</td></tr><tr><td>3</td><td>0</td><td>-4, -3, -2, -1, 0, 1, 2, 3</td><td>√</td><td>√</td></tr><tr><td>2</td><td>-0.5</td><td>-1.5, -0.5, 0.5, 1.5</td><td>✕</td><td>√</td></tr><tr><td>1</td><td>-0.5</td><td>-0.5, 0.5</td><td>✕</td><td>√</td></tr></table>


Table 2: The BBQ data type: zero point z, possible values of $q ,$ and whether BBQ can be encoded as INT4 and MX FP4 for each precision $b \in \{ 1 , 2 , 3 , 4 \}$


## 3.4 DEQUANTIZATION

Inspired by LSQ, the dequantization formula of BBQ, Equation $^ { 6 , }$ includes a learnable scaling parameter $s \stackrel { \cdot } { = } \gamma / \bar { 2 } ^ { b - 1 }$ . A learnable scaling factor enables control of the magnitude of xˆ. To ensure the learnable scaling factor can be initialized to the same value regardless of precision $b ,$ we do not use s directly but instead decouple s into the ratio of a precision-independent learnable scaling factor $\gamma$ and a constant precision-dependent scaling factor $2 ^ { \overset { \cdot } { b } - 1 }$ . Following LSQ, we apply gradient scaling (Esser et al., 2020) to reduce the gradient of $\gamma$ by a factor of ${ \check { \sqrt { d } } } ,$ , where d is the number of weights/activations in x. Following Nagel et al. (2021), we use channel-wise $\gamma$ for weights and per-tensor γ for activations. 

<table><tr><td rowspan="2">Params</td><td rowspan="2">Tokens</td><td rowspan="2">Full</td><td rowspan="2">Bits</td><td colspan="2">BBQ</td><td colspan="2">QuEST</td><td colspan="2">LSQ</td></tr><tr><td>Entropy</td><td>Perplexity</td><td>Entropy</td><td>Perplexity</td><td>Entropy</td><td>Perplexity</td></tr><tr><td>95M</td><td>3B</td><td>24.75</td><td>4</td><td>3.93</td><td>25.51</td><td>3.61</td><td>26.37</td><td>3.59</td><td>27.46</td></tr><tr><td>95M</td><td>3B</td><td>24.75</td><td>3</td><td>2.96</td><td>26.55</td><td>2.78</td><td>29.04</td><td>2.74</td><td>30.27</td></tr><tr><td>95M</td><td>3B</td><td>24.75</td><td>2</td><td>1.97</td><td>31.34</td><td>1.92</td><td>35.58</td><td>1.69</td><td>36.58</td></tr><tr><td>95M</td><td>3B</td><td>24.75</td><td>1</td><td>1.00</td><td>49.22</td><td>1.00</td><td>67.78</td><td>-</td><td>-</td></tr><tr><td>125M</td><td>5B</td><td>21.51</td><td>4</td><td>3.93</td><td>22.15</td><td>3.61</td><td>22.98</td><td>3.60</td><td>23.77</td></tr><tr><td>125M</td><td>5B</td><td>21.51</td><td>3</td><td>2.96</td><td>23.22</td><td>2.78</td><td>25.21</td><td>2.74</td><td>26.28</td></tr><tr><td>125M</td><td>5B</td><td>21.51</td><td>2</td><td>1.98</td><td>27.34</td><td>1.93</td><td>31.32</td><td>1.81</td><td>31.42</td></tr><tr><td>125M</td><td>5B</td><td>21.51</td><td>1</td><td>1.00</td><td>44.58</td><td>1.00</td><td>72.54</td><td>-</td><td>-</td></tr><tr><td>200M</td><td>10B</td><td>17.93</td><td>4</td><td>3.93</td><td>18.79</td><td>3.61</td><td>19.06</td><td>2.73</td><td>1778</td></tr><tr><td>200M</td><td>10B</td><td>17.93</td><td>3</td><td>2.96</td><td>19.74</td><td>2.78</td><td>20.82</td><td>2.50</td><td>140.9</td></tr><tr><td>200M</td><td>10B</td><td>17.93</td><td>2</td><td>1.98</td><td>23.08</td><td>1.93</td><td>25.46</td><td>1.63</td><td>78.19</td></tr><tr><td>200M</td><td>10B</td><td>17.93</td><td>1</td><td>1.00</td><td>38.27</td><td>1.00</td><td>52.37</td><td>-</td><td>-</td></tr><tr><td>300M</td><td>20B</td><td>15.43</td><td>4</td><td>3.93</td><td>16.10</td><td>3.61</td><td>16.26</td><td>-</td><td>-</td></tr><tr><td>300M</td><td>20B</td><td>15.43</td><td>3</td><td>2.96</td><td>16.90</td><td>2.78</td><td>17.67</td><td>-</td><td>-</td></tr><tr><td>300M</td><td>20B</td><td>15.43</td><td>2</td><td>1.98</td><td>19.75</td><td>1.93</td><td>21.53</td><td>-</td><td>-</td></tr></table>


Table 3: Entropy and Perplexity of LLaMA models pre-trained on the C4 dataset. The headers are the number of parameters (Params), the number of tokens (Tokens), perplexity without any quantization (Full), activation/weight precision (Bits), and the entropy and perplexity of BBQ, QuEST, and LSQ.


The initialization of $\gamma$ plays a critical role, as overly large initialization can lead to gradient explosion while overly small initialization can lead to gradient vanishing. We initialize $\gamma$ to $\zeta ^ { * } \sigma _ { 0 }$ , where $\sigma _ { 0 }$ is the σ measured at the first iteration of training, and $\zeta ^ { * }$ is defined as: 

$$
\zeta^ {*} = \underset {\zeta} {\arg \min} E _ {v \sim N (0, 1)} [ (v - \zeta (2 \Phi (v) - 1)) ^ {2} ]\tag{9}
$$

See Section A.2 for a detailed explanation of $\zeta ^ { * }$ and Equation 9. In short, by initializing $\gamma \mathrm { t o } \zeta ^ { * } \sigma _ { 0 } .$ xˆ can have approximately the same magnitude as x (in the first iteration) to prevent the activation magnitude from exploding or vanishing as tokens travel deeper into the network. After the first iteration, we simply let $\gamma$ update itself based on gradient descent. We do not apply weight decay to $\gamma ,$ as an overly decayed $\gamma$ may lead to gradient vanishing. 

## 4 EVALUATION AND DISCUSSION

We run experiments to evaluate the perplexity of BBQ against other SOTA QAPT methods on representative LLM architectures. We use the publicly available source code of QuEST (Panferov et al., 2025), add the implementation of BBQ, and train on LLaMA (Touvron et al., 2023; Vaswani et al., 2017) models with n non-embedding parameters plus e embedding parameters, where $n \in \{ 3 0 M , 5 0 M , 1 0 0 M , 2 0 0 M \}$ and $e \in \{ { \bar { 6 5 } } { \bar { M } } , 7 5 M , 1 0 { \bar { 0 } } M , 1 0 0 M \}$ . For each model, we pre-train with 100n C4 tokens while quantizing the weights and activations of all linear layers to b-bit following QuEST. We compare BBQ against QuEST/LSQ for each $b \in \{ 1 , 2 , 3 , 4 \}$ , and present the evaluation quality (perplexity) and weight entropy<sup>2</sup> (bits) in Table 3. The omitted entries in Table 3 are because LSQ does not support 1-bit quantization and LSQ diverges for larger models (unlike BBQ which works well for 1-bit and converges even for larger models). We show zero-shot results in Table 9, and training loss curves in Section A.7. Each experiment for $n + e \leq 2 0 0 M$ is conducted on one Nvidia RTX 5090 and lasts for up to 1 day. Each experiment for $n + e > 2 0 0 M$ is conducted on one Nvidia A100 80GB and lasts for 3.5 days. In total, all experiments took approximately 1.5 GPU months. Lastly, we evaluate BBQ against QuEST, LSQ, ParetoQ-SEQ (Liu et al., 2025), N2UQ (Liu et al., 2022), and NormalFloat (Dettmers et al., 2023) on GPT (Brown et al., 2020) models and present our results in Table 4. 

Results in Table 3 and 4 suggest BBQ can consistently achieve higher entropy and lower perplexity than QuEST and LSQ at the same precision, and that entropy can be a good proxy to prediction quality. A special case is when $b = 1$ . In this case, both BBQ and QuEST achieve the maximal entropy of 1 bit. We attribute the performance gain of BBQ in the 1-bit case to the fact that Φ from BBQ is smoother than the clip operation from QuEST (as discussed in Section 3.2). In addition, we note that LSQ diverges for LLaMA-200M, which is reflected in its entropy. Notably, when LSQ does not diverge, like in the case of LLaMA-95M and LLaMA-125M, we see that LSQ can achieve an entropy of 3.6 bits for 4-bit precision, 2.74 bits for 3-bit precision, and 1.7 bits for 2-bit precision. However, when LSQ diverges, like in the case of LLaMA-200M, we see that its entropy drops to 2.73 bits for 4-bit precision, 2.50 bits for 3-bit precision, and 1.63 bits for 2-bit precision. This shows that entropy is a good indicator of a quantized model’s prediction quality. 

<table><tr><td>Params</td><td>95M</td><td>95M</td><td>95M</td><td>95M</td><td>95M</td><td>95M</td><td>95M</td><td>95M</td><td>95M</td><td>95M</td><td>95M</td><td>95M</td></tr><tr><td>Bits WA</td><td>16</td><td>4</td><td>4</td><td>4</td><td>4</td><td>3</td><td>3</td><td>3</td><td>3</td><td>3</td><td>1</td><td>1</td></tr><tr><td>Method</td><td>-</td><td>BBQ</td><td>QuEST</td><td>LSQ</td><td>NF4</td><td>BBQ</td><td>QuEST</td><td>LSQ</td><td>NF3</td><td>N2UQ</td><td>BBQ</td><td>QuEST</td></tr><tr><td>Perplex</td><td>25</td><td>26.7</td><td>26.83</td><td>27.55</td><td>28.5</td><td>28.61</td><td>29.7</td><td>30.85</td><td>36.16</td><td>29.87</td><td>53.8</td><td>77.96</td></tr><tr><td>Params</td><td>95M</td><td>95M</td><td>95M</td><td>95M</td><td>95M</td><td>95M</td><td>125M</td><td>125M</td><td>125M</td><td>125M</td><td>125M</td><td>125M</td></tr><tr><td>Bits WA</td><td>2</td><td>2</td><td>2</td><td>2</td><td>2</td><td>2</td><td>3</td><td>3</td><td>2</td><td>2</td><td>1</td><td>1</td></tr><tr><td>Method</td><td>BBQ</td><td>QuEST</td><td>LSQ</td><td>NF2</td><td>SEQ</td><td>N2UQ</td><td>BBQ</td><td>QuEST</td><td>BBQ</td><td>QuEST</td><td>BBQ</td><td>QuEST</td></tr><tr><td>Perplex</td><td>34.36</td><td>36.7</td><td>38.98</td><td>246.6</td><td>40.07</td><td>35.59</td><td>24.78</td><td>26.05</td><td>29.38</td><td>31.47</td><td>50.46</td><td>69.78</td></tr></table>


Table 4: Perplexity of GPT models pre-trained on the C4 dataset.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/67335314-6215-4a10-a04e-e66551ddf124/3419d72ab614053993337ae02fb16dd5aca78a0c42c8366788c46c39f14d230d.jpg)



Figure 3: Quantized weight entropy vs. training iterations for LLaMA-300M with 2-bit weight and activations, pre-trained on 20 billion C4 tokens (batched into 80 thousand training iterations).


## 4.1 ENTROPY AND LEARNING CAPACITY

In this section, we discuss, based on empirical measurements of entropy during training, the improvement of BBQ over QuEST in terms of learning capacity utilization. Figure 3 shows the behaviour of weight entropy of BBQ and QuEST during 2-bit training. We note a few observations. First, BBQ can maximize entropy as it can achieve the theoretical maximum entropy of 2 bits at the beginning of training. Second, during training, a BBQ-quantized model can adjust (increase or decrease) its entropy. Third, excluding the first 8 thousand iterations when learning rate is warming up, BBQ tends to increase weight entropy for the last 72 thousand iterations. This suggests, as more training tokens are presented to the model, more information is accumulated in the weights. Lastly, the entropy of QuEST seems to have an empirical ceiling at around 1.93 bits. We attribute the empirical entropy ceiling of QuEST to the following: QuEST applies uniform quantization to $\mathrm { H T } ( x ) { \bar { / } } \sigma ,$ and therefore QuEST is only ITO if $\operatorname { H T } ( x )$ follows a uniform distribution, which is unlikely as the Hadamard transform has a tendency to Gaussianize (Yang et al., 2025; Panferov et al., 2025). See Section A.5 for a visualization of the difference between BBQ and QuEST. 

By having a lower empirical entropy upper bound of 1.93 bits, QuEST limits the learning capacity of the model, while BBQ does not suffer from such capacity under-utilization. Suppose a model has d parameters, each quantized to b bits. This means the model has a total of $2 ^ { d b }$ possible unique states. Some of the $2 ^ { d b }$ states have higher entropy, while others have lower entropy. QAPT’s objective is to find a (locally) optimal state out of all $2 ^ { \aa \Vec { d b } }$ possible states. However, if QuEST enforces its weight entropy to be lower than 1.93 bits, then some of the high-entropy states cannot be explored by the training algorithm. Therefore, the model’s learning capacity is effectively reduced. 

## 4.2 INFERENCE SPEEDUP

In this section, we discuss our profiling results of the inference latency of BBQ, compared against FP16 and NF4. We perform our latency measurements on NVIDIA RTX 5090, a Blackwell GPU that supports MX FP4 matrix multiplication, and NVIDIA A100 80GB, an Ampere GPU that supports INT4 matrix multiplication. At inference time, for each linear layer, BBQ launches an activation quantization kernel to compute the quantized activation matrix $Q _ { x } ,$ a low-precision matrix multiplication kernel between $Q _ { x }$ and the quantized weight matrix $Q _ { w } ,$ , and lastly an element-wise scaling kernel to scale the result $Q _ { x } Q _ { w }$ by a factor of $s _ { x } s _ { w }$ . We note that BBQ performs its weight quantization offline and therefore weight quantization does not incur inference latency overhead. Since compute-efficiency, the feature that distinguishes BBQ from NF4, manifests in latency savings during the execution of compute-bound kernels, we focus on profiling the prefill phase of inference. However, we note both BBQ and NF4 can achieve latency reduction during the memory-bound decode phase as both can reduce memory bandwidth consumption. In addition, during the decode phase, BBQ may save energy compared to NF4 as computation is performed in low precision. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/67335314-6215-4a10-a04e-e66551ddf124/15f45b3a67a7d5281c50ca8f4a457f1dde3c5a5334ab51bd7c3d299fd9351a7b.jpg)



(a) BBQ quantization kernel of an $N \times N$ matrix


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/67335314-6215-4a10-a04e-e66551ddf124/0165e98f21b7577d25b895cf0e7aa2ed5b3f2584b1e86f4f12903a7e678d6b85.jpg)



(b) Matrix multiplication of two N × N matrices



Figure 4: Kernel latency on Nvidia RTX 5090 (y-axis) vs. matrix size N (x-axis).


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/67335314-6215-4a10-a04e-e66551ddf124/2f1ffb4a4dcb130c13525ea142531a58ae38422cfb990c800d3c66bca2dad1b4.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/67335314-6215-4a10-a04e-e66551ddf124/491d4edbdbc55d8b96dadaf577250df38488314bf56ab9679934439e666c58e5.jpg)



Figure 5: End-to-end LLaMA inference latency of BBQ, FP16 and NF4 on RTX 5090 and A100 GPUs. For each linear layer, BBQ launches an activation quantization kernel (green region), an fp4/int4 matrix multiplication kernel (part of blue regions), and an element-wise scaling kernel (part of orange region).


We first show at the kernel level, the latency of the BBQ activation quantization kernel is negligible compared to the time saved by performing matrix multiplication in low-precision. Figure 4a shows the latency of a Triton (Tillet et al., 2019) implementation (shown in Section A.1) of Algorithm 1 on $N \times N$ activation matrices, for $N \in \{ 2 5 6 \cdot \bar { i } \mid i \in \mathbb { Z } , 2 \leq i \leq 3 2 \}$ . Note that $N = 2 5 6 \cdot 3 2 = 8 1 9 2$ corresponds to the dimension of LLaMA-65B (Touvron et al., 2023), an important LLM benchmark. Figure 4a shows that when $N = 8 1 9 2$ , our BBQ quantization kernel takes 0.3 milliseconds to quantize an $N \times N$ activation matrix to FP4. However, Figure 4b shows the latency saving of FP4 matrix multiplication over FP16 matrix multiplication is 4 milliseconds when $\dot { N } = 8 \bar { 1 } 9 2 .$ Thus, for LLaMA-65B, BBQ’s activation quantization adds a latency overhead that is only onetenth the latency savings enabled by BBQ’s reduced precision matrix multiplication. We attribute the low latency of the activation quantization kernel to the following: since quantization kernels are element-wise and highly memory-bound, compute units on GPUs are often idle waiting for data to arrive from global memory, and therefore additional computation (such as the Hadamard transform and binary search) can be fused into the quantization kernel without incurring latency overhead. 

We next show BBQ can outperform FP16 and NF4 in terms of end-to-end inference latency, using LLaMA-700M and LLaMA-2.4B as our benchmarks. Figure 5 illustrates the overall inference latency, with contributions from different types of kernels shown in different colours. While BBQ adds additional latency overhead to quantize activations (green region) and perform element-wise scaling (orange region), BBQ significantly reduces the latency of matrix multiplication (blue region), which is the latency bottleneck in both FP16 and NF4. In general, BBQ shows a 40% speedup over FP16 and a 48% speedup over NF4. While both BBQ and NF4 can compress large models, BBQ is compute-efficient and has shorter latency than FP16. In contrast, during prefill, NF4 has longer latency than FP16, since NF4-quantized activations and weights must be de-quantized before being consumed by full-precision matrix multiplication kernels. 

<table><tr><td>Row</td><td>HT</td><td>RMS</td><td>PIT</td><td>Learn γ</td><td>γ Init</td><td>Perplexity</td><td>Entropy</td><td>Notes</td></tr><tr><td>1</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>31.34</td><td>1.97</td><td>BBQ</td></tr><tr><td>2</td><td>✕</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>35.79</td><td>1.98</td><td></td></tr><tr><td>3</td><td>✓</td><td>✕</td><td>✓</td><td>✓</td><td>✓</td><td>35.93</td><td>1.98</td><td></td></tr><tr><td>4</td><td>✓</td><td>✓</td><td>✕</td><td>✕</td><td>✕</td><td>35.58</td><td>1.92</td><td>QuEST</td></tr><tr><td>5</td><td>✓</td><td>✓</td><td>✓</td><td>✕</td><td>✕</td><td>138.3</td><td>1.92</td><td></td></tr><tr><td>6</td><td>✓</td><td>✓</td><td>✓</td><td>✕</td><td>✓</td><td>31.46</td><td>1.98</td><td></td></tr></table>


Table 5: Ablation of BBQ features: Hadamard Transform (HT), RMS Normalization (RMS), Probability Integral Transform (PIT), making γ a learnable parameter (Learn γ), and γ initialization (γ Init). A ✓ means a feature is present, and ✗ means the feature is not included.


## 4.3 ABLATION

In this section, we present ablation studies of BBQ’s individual features in Table 5. Specifically, we study the effects of Hadamard Transform, RMS Normalization, probability integral transform (Φ) instead of the regular clipping function, having a learnable scaling factor γ, and initializing γ to $\zeta ^ { * } \sigma _ { 0 }$ instead of a dummy value (e.g. 1). We evaluate on a 2-bit LLaMA-95M pre-trained with 3B tokens. Results show Hadamard Transform and RMS normalization are two solid features BBQ inherited from QuEST, as removing them from BBQ results in a perplexity increase of 4.45 (row 2 compared to row 1) and 4.59 (row 3 compared to row 1). The probability integral transform (Φ) is meant to replace QuEST’s clip function. We see that naively substituting clip with Φ (row 4 compared to row 5) without γ initialization leads to divergence (a perplexity increase from 35.58 to 138.3), but the combination of Φ and γ initialization (row 6) can outperform QuEST (row 4) by a perplexity decrease of 4.12. Lastly, making γ learnable (row 1 compared to row 6) can further reduce the perplexity by a smaller margin of 0.12. 

## 5 LIMITATIONS AND FUTURE WORK

While BBQ can achieve lower perplexity than QuEST and LSQ for QAPT, BBQ is, by design, a quantizer that makes no attempt to reduce/bound the Euclidean distance between x and xˆ, since they live in separate domains. Therefore, when applied to QAFT, BBQ’s unbounded quantization error will drastically reduce model quality and cannot catch up to “same-domain” quantizers (ones that follow Equation 1) like QuEST and LSQ within the short time frame of QAFT. For similar reasons, BBQ is not suitable for PTQ. 

BBQ can maximize entropy under the assumption that the Hadamard Transform can Gaussianize (Panferov et al., 2025; Yang et al., 2025) BBQ’s input data. Empirically, we find HT(x) to be exactly Gaussian at the beginning of training, but not perfectly Gaussian afterwards. As future work, it may be possible to replace Φ with a more accurate smooth approximation of the non-differentiable empirical CDF of HT(x), to more closely approximate true probability integral transform. 

## 6 CONCLUSION

In this work, we identify existing SOTA QAPT methods under-utilize learning capacity. While existing ITO quantization methods can maximize entropy, they are not compute-efficient, which limits their applicability to edge devices with energy constraints. Utilizing our key insight that learning is domain-agnostic, we propose BBQ, which performs ITO quantization in the input domain, while returning outputs in a compute-efficient domain. Empirical results show BBQ outperforms QuEST and LSQ without sacrificing compute efficiency. 

## 7 REPRODUCIBILITY STATEMENT

A link to our source code is provided in the abstract of the paper. Our code should allow readers to reproduce tables and figures in the paper. 

## 8 ACKNOWLEDGMENTS

This research was supported by the Natural Sciences and Engineering Research Council of Canada through a Discovery Grant. We thank Renjie Liao for providing access to additional GPU resources at UBC. Tor Aamodt has recently served as a consultant for NVIDIA, HPE, Cisco and IBM. 

## REFERENCES



Abien Fred Agarap. Deep learning using rectified linear units (relu), 2018. URL http://arxiv. org/abs/1803.08375. cite arxiv:1803.08375Comment: 7 pages, 11 figures, 9 tables. 





Yoshua Bengio, Nicholas Leonard, and Aaron Courville. Estimating or propagating gradients´ through stochastic neurons for conditional computation, 2013. URL https://arxiv.org/ abs/1308.3432. 





Johan Bjorck, Carla Gomes, Bart Selman, and Kilian Q. Weinberger. Understanding batch normalization. In Proceedings of the 32nd International Conference on Neural Information Processing Systems, NIPS’18, pp. 7705–7716, Red Hook, NY, USA, 2018. Curran Associates Inc. 





Leon Bottou, Frank E. Curtis, and Jorge Nocedal. Optimization methods for large-scale machine´ learning, 2018. URL https://arxiv.org/abs/1606.04838. 





Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners, 2020. URL https://arxiv.org/abs/2005.14165. 





Roberto L. Castro, Andrei Panferov, Soroush Tabesh, Oliver Sieberling, Jiale Chen, Mahdi Nikdan, Saleh Ashkboos, and Dan Alistarh. Quartet: Native fp4 training can be optimal for large language models. CoRR, abs/2505.14669, May 2025. URL https://doi.org/10.48550/arXiv. 2505.14669. 





Feng Cheng, Cong Guo, Chiyue Wei, Junyao Zhang, Changchun Zhou, Edward Hanson, Jiaqi Zhang, Xiaoxiao Liu, Hai Li, and Yiran Chen. Ecco: Improving memory bandwidth and capacity for llms via entropy-aware cache compression. In Proceedings of the 52nd Annual International Symposium on Computer Architecture, ISCA ’25, pp. 793–807, New York, NY, USA, 2025. Association for Computing Machinery. ISBN 9798400712616. doi: 10.1145/3695053.3731024. URL https://doi.org/10.1145/3695053.3731024. 





Yubei Chen Chun-Hsiao Yeh. IN100pytorch: Pytorch implementation: Training resnets on imagenet-100. https://github.com/danielchyeh/ImageNet-100-Pytorch, 2022. 





Thomas M. Cover and Joy A. Thomas. Elements of Information Theory (Wiley Series in Telecommunications and Signal Processing). Wiley-Interscience, USA, 2006. ISBN 0471241954. 





F. N. David and N. L. Johnson. The probability integral transformation when parameters are estimated from the sample. Biometrika, 35(1-2):182–190, 05 1948. ISSN 0006-3444. doi: 10.1093/biomet/35.1-2.182. URL https://doi.org/10.1093/biomet/35.1-2.182. 





Tim Dettmers, Mike Lewis, Sam Shleifer, and Luke Zettlemoyer. 8-bit optimizers via block-wise quantization. In International Conference on Learning Representations, 2022. URL https: //openreview.net/forum?id=shpkpVXzo3h. 





Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. Qlora: efficient finetuning of quantized llms. In Proceedings of the 37th International Conference on Neural Information Processing Systems, NIPS ’23, Red Hook, NY, USA, 2023. Curran Associates Inc. 





Jordan Dotzel, Yuzong Chen, Bahaa Kotb, Sushma Prasad, Gang Wu, Sheng Li, Mohamed S. Abdelfattah, and Zhiru Zhang. Learning from students: applying t-distributions to explore accurate and efficient formats for llms. In Proceedings of the 41st International Conference on Machine Learning, ICML’24. JMLR.org, 2024. 





DaYou Du, Yijia Zhang, Shijie Cao, Jiaqi Guo, Ting Cao, Xiaowen Chu, and Ningyi Xu. BitDistiller: Unleashing the potential of sub-4-bit LLMs via self-distillation. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 102–116, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.7. URL https://aclanthology.org/2024.acl-long.7/. 





Steven K. Esser, Jeffrey L. McKinstry, Deepika Bablani, Rathinakumar Appuswamy, and Dharmendra S. Modha. Learned step size quantization. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id=rkgO66VKDS. 





Ronald A. Fisher. Statistical methods for research workers. Oliver and Boyd., 1932. 





Leo Gao, Tom Dupre la Tour, Henk Tillman, Gabriel Goh, Rajan Troll, Alec Radford, Ilya Sutskever, Jan Leike, and Jeffrey Wu. Scaling and evaluating sparse autoencoders. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/ forum?id=tcsZt9ZNKD. 





Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. CoRR, abs/1512.03385, 2015. URL http://arxiv.org/abs/1512.03385. 





Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus), 2023. URL https:// arxiv.org/abs/1606.08415. 





Kurt Hornik, Maxwell Stinchcombe, and Halbert White. Multilayer feedforward networks are universal approximators. Neural Networks, 2(5):359–366, 1989. ISSN 0893-6080. doi: https: //doi.org/10.1016/0893-6080(89)90020-8. URL https://www.sciencedirect.com/ science/article/pii/0893608089900208. 





Sergey Ioffe and Christian Szegedy. Batch normalization: accelerating deep network training by reducing internal covariate shift. In Proceedings of the 32nd International Conference on International Conference on Machine Learning - Volume 37, ICML’15, pp. 448–456. JMLR.org, 2015. 





Sehoon Kim, Coleman Hooper, Amir Gholami, Zhen Dong, Xiuyu Li, Sheng Shen, Michael W. Mahoney, and Kurt Keutzer. Squeezellm: dense-and-sparse quantization. In Proceedings of the 41st International Conference on Machine Learning, ICML’24. JMLR.org, 2024. 





Tanishq Kumar, Zachary Ankner, Benjamin Frederick Spector, Blake Bordelon, Niklas Muennighoff, Mansheej Paul, Cengiz Pehlevan, Christopher Re, and Aditi Raghunathan. Scaling laws for precision. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=wg1PCg3CUP. 





Haokun Lin, Haobo Xu, Yichen Wu, Jingzhi Cui, Yingtao Zhang, Linzhan Mou, Linqi Song, Zhenan Sun, and Ying Wei. Duquant: Distributing outliers via dual transformation makes stronger quantized LLMs. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/forum?id=mp8u2Pcmqz. 





Jing Liu, Ruihao Gong, Xiuying Wei, Zhiwei Dong, Jianfei Cai, and Bohan Zhuang. QLLM: Accurate and efficient low-bitwidth quantization for large language models. In The Twelfth International Conference on Learning Representations, 2024a. URL https://openreview.net/ forum?id=FIplmUWdm3. 





Zechun Liu, Kwang-Ting Cheng, Dong Huang, Eric Xing, and Zhiqiang Shen. Nonuniform-touniform quantization: Towards accurate quantization via generalized straight-through estimation. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4932– 4942, 2022. doi: 10.1109/CVPR52688.2022.00489. 





Zechun Liu, Barlas Oguz, Changsheng Zhao, Ernie Chang, Pierre Stock, Yashar Mehdad, Yangyang Shi, Raghuraman Krishnamoorthi, and Vikas Chandra. LLM-QAT: Data-free quantization aware training for large language models. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Findings of the Association for Computational Linguistics: ACL 2024, pp. 467–484, Bangkok, Thailand, August 2024b. Association for Computational Linguistics. doi: 10.18653/v1/2024. findings-acl.26. URL https://aclanthology.org/2024.findings-acl.26/. 





Zechun Liu, Changsheng Zhao, Hanxian Huang, Sijia Chen, Jing Zhang, Jiawei Zhao, Scott Roy, Lisa Jin, Yunyang Xiong, Yangyang Shi, Lin Xiao, Yuandong Tian, Bilge Soran, Raghuraman Krishnamoorthi, Tijmen Blankevoort, and Vikas Chandra. Paretoq: Improving scaling laws in extremely low-bit LLM quantization. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id=PMSNd8xTHp. 





Shuming Ma, Hongyu Wang, Lingxiao Ma, Lei Wang, Wenhui Wang, Shaohan Huang, Li Dong, Ruiping Wang, Jilong Xue, and Furu Wei. The era of 1-bit llms: All large language models are in 1.58 bits, 2024a. URL https://arxiv.org/abs/2402.17764. 





Yuexiao Ma, Huixia Li, Xiawu Zheng, Feng Ling, Xuefeng Xiao, Rui Wang, Shilei Wen, Fei Chao, and Rongrong Ji. Affinequant: Affine transformation quantization for large language models. In ICLR, 2024b. URL https://openreview.net/forum?id=of2rhALq8l. 





Vladimir Malinovskii, Denis Mazur, Ivan Ilin, Denis Kuznedelev, Konstantin Pavlovich Burlachenko, Kai Yi, Dan Alistarh, and Peter Richtarik. PV-tuning: Beyond straight-through´ estimation for extreme LLM compression. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/forum?id= YvA8UF0I37. 





Markus Nagel, Marios Fournarakis, Rana Ali Amjad, Yelysei Bondarenko, Mart van Baalen, and Tijmen Blankevoort. A white paper on neural network quantization, 2021. URL https:// arxiv.org/abs/2106.08295. 





Yurii Nesterov. Introductory Lectures on Convex Optimization: A Basic Course. Springer Publishing Company, Incorporated, 1 edition, 2014. ISBN 1461346916. 





Nvidia. Nvidia turing gpu architecture, 2018. URL https://images.nvidia. com/aem-dam/en-zz/Solutions/design-visualization/technologies/ turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf. 





Nvidia. Nvidia ampere ga102 gpu architecture, 2021. URL https://www.nvidia.com/ content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2. pdf. 





Nvidia. Nvidia rtx blackwell gpu architecture, 2025. URL https:// images.nvidia.com/aem-dam/Solutions/geforce/blackwell/ nvidia-rtx-blackwell-gpu-architecture.pdf. 





Andrei Panferov, Jiale Chen, Soroush Tabesh, Roberto L. Castro, Mahdi Nikdan, and Dan Alistarh. QuEST: Training accurate LLMs over highly-compressed weights and activation. In Sparsity in LLMs (SLLM): Deep Dive into Mixture ofExperts, Quantization, Hardware, and Inference, 2025. URL https://openreview.net/forum?id=ElgAzv9fNk. 





Bita Darvish Rouhani, Nitin Garegrat, Tom Savell, Ankit More, Kyung-Nam Han, Ritchie Zhao, Mathew Hall, Eric Chung Jasmine Klar, Yuan Yu, Michael Schulte, Ralph Wittig, Ian Bratt, Nigel Stephens, Jelena Milanovic, John Brothers, Pradeep Dubey, Marius Cornea, Alexander Heinecke, Andres Rodriguez, Martin Langhammer, Summer Deng, Maxim Naumov, Paulius Micikevicius, Michael Siu, and Colin Verrilli. Ocp microscaling formats (mx) specification, 2023. URL https://www.opencompute.org/documents/ ocp-microscaling-formats-mx-v1-0-spec-final-pdf. 





Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Li Fei-Fei. Imagenet large scale visual recognition challenge. Int. J. Comput. Vision, 115(3):211–252, December 2015. ISSN 0920-5691. doi: 10.1007/s11263-015-0816-y. URL https://doi. org/10.1007/s11263-015-0816-y. 





C. E. Shannon. A mathematical theory of communication. Bell System Technical Journal, 27(3):379–423, 1948. doi: https://doi.org/10.1002/j.1538-7305.1948.tb01338.x. URL https://onlinelibrary.wiley.com/doi/abs/10.1002/j.1538-7305. 1948.tb01338.x. 





Wenqi Shao, Mengzhao Chen, Zhaoyang Zhang, Peng Xu, Lirui Zhao, Zhiqian Li, Kaipeng Zhang, Peng Gao, Yu Qiao, and Ping Luo. Omniquant: Omnidirectionally calibrated quantization for large language models. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=8Wuvhh0LYW. 





Xuan Shen, Zhenglun Kong, Changdi Yang, Zhaoyang Han, Lei Lu, Peiyan Dong, Cheng Lyu, Chih hsiang Li, Xuehang Guo, Zhihao Shu, Wei Niu, Miriam Leeser, Pu Zhao, and Yanzhi Wang. Edgeqat: Entropy and distribution guided quantization-aware training for the acceleration of lightweight llms on the edge. CoRR, abs/2402.10787, 2024. URL https://doi.org/ 10.48550/arXiv.2402.10787. 





Philippe Tillet, H. T. Kung, and David Cox. Triton: an intermediate language and compiler for tiled neural network computations. In Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages, MAPL 2019, pp. 10–19, New York, NY, USA, 2019. Association for Computing Machinery. ISBN 9781450367196. doi: 10.1145/3315508.3329973. URL https://doi.org/10.1145/3315508.3329973. 





Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Herve Jegou. Training data-efficient image transformers & distillation through attention. In Marina Meila and Tong Zhang (eds.), Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pp. 10347– 10357. PMLR, 18–24 Jul 2021. URL https://proceedings.mlr.press/v139/ touvron21a.html. 





Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothee´ Lacroix, Baptiste Roziere, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Ar-` mand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models, 2023. URL https://arxiv.org/abs/2302.13971. 





Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (eds.), Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017. URL https://proceedings.neurips.cc/paper_files/paper/2017/ file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf. 





Haocheng Xi, ChangHao Li, Jianfei Chen, and Jun Zhu. Training transformers with 4-bit integers. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https: //openreview.net/forum?id=H9hWlfMT6O. 





Kai Xu, Minghai Qin, Fei Sun, Yuhao Wang, Yen-Kuang Chen, and Fengbo Ren. Learning in the frequency domain. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1737–1746, 2020. doi: 10.1109/CVPR42600.2020.00181. 





Kohei Yamamoto. Learnable Companding Quantization for Accurate Low-bit Neural Networks . In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5027–5036, Los Alamitos, CA, USA, June 2021. IEEE Computer Society. doi: 10. 1109/CVPR46437.2021.00499. URL https://doi.ieeecomputersociety.org/10. 1109/CVPR46437.2021.00499. 





Kaicheng Yang, Xun Zhang, Haotong Qin, Yucheng Lin, Kaisen Yang, Xianglong Yan, and Yulun Zhang. Robuq: Pushing dits to w1.58a2 via robust activation quantization, 2025. URL https: //arxiv.org/abs/2509.23582. 





Ningfeng Yang and Tor M. Aamodt. Improving the straight-through estimator with zeroth-order information. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id=ceTeM2Xl1n. 





Zhewei Yao, Zhen Dong, Zhangcheng Zheng, Amir Gholami, Jiali Yu, Eric Tan, Leyuan Wang, Qijing Huang, Yida Wang, Michael Mahoney, and Kurt Keutzer. Hawq-v3: Dyadic neural network quantization. In Marina Meila and Tong Zhang (eds.), Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pp. 11875–11886. PMLR, 18–24 Jul 2021. URL https://proceedings.mlr.press/ v139/yao21a.html. 





Davis Yoshida. Nf4 isn’t information theoretically optimal (and that’s good), 2023. URL https: //arxiv.org/abs/2306.06965. 



## A APPENDIX

## A.1 KERNEL IMPLEMENTATION

In this section, we present an example implementation of Algorithm 1 using the Triton (Tillet et al., 2019) programming language. 

Nvidia GPUs provide extensive support for low-precision tensor core operations, allowing expensive high-precision matrix multiplications to be replaced with low-precision alternatives, leading to higher compute throughput and lower energy consumption. For example, the Nvidia Blackwell architecture (Nvidia, 2025) provides tensor core support for the MX FP4(Rouhani et al., 2023) data type. As another example, the Turing (Nvidia, 2018) and Ampere (Nvidia, 2021) architectures provide support for the INT4 data type. We list in Table 6 the representable values for both data types. 

<table><tr><td>Binary Representation</td><td>Signed INT4</td><td>MX FP4</td></tr><tr><td>0b0000</td><td>0</td><td>0</td></tr><tr><td>0b0001</td><td>1</td><td>0.5</td></tr><tr><td>0b0010</td><td>2</td><td>1</td></tr><tr><td>0b0011</td><td>3</td><td>1.5</td></tr><tr><td>0b0100</td><td>4</td><td>2</td></tr><tr><td>0b0101</td><td>5</td><td>3</td></tr><tr><td>0b0110</td><td>6</td><td>4</td></tr><tr><td>0b0111</td><td>7</td><td>6</td></tr><tr><td>0b1000</td><td>-8</td><td>0</td></tr><tr><td>0b1001</td><td>-7</td><td>-0.5</td></tr><tr><td>0b1010</td><td>-6</td><td>-1</td></tr><tr><td>0b1011</td><td>-5</td><td>-1.5</td></tr><tr><td>0b1100</td><td>-4</td><td>-2</td></tr><tr><td>0b1101</td><td>-3</td><td>-3</td></tr><tr><td>0b1110</td><td>-2</td><td>-4</td></tr><tr><td>0b1111</td><td>-1</td><td>-6</td></tr></table>


Table 6: Signed INT4 vs. MX FP4



Table 7 shows the values that BBQ can represent.


<table><tr><td>Precision</td><td>Possible Values of q</td></tr><tr><td>4</td><td>-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7</td></tr><tr><td>3</td><td>-4,-3,-2,-1,0,1,2,3</td></tr><tr><td>2</td><td>-1.5,-0.5,0.5,1.5</td></tr><tr><td>1</td><td>-0.5,0.5</td></tr></table>


Table 7: Values that BBQ can represent. As shown, for b = 4, BBQ can be encoded as INT4. For b = 3, BBQ can be encoded as INT4 or FP4. For $b \in \{ 1 , 2 \}$ , BBQ can be encoded as FP4.


Since b = 3 can be encoded by both INT4 and FP4, we present an example triton kernel for BBQ with 3-bit quantization following Algorithm 1. 

```python
@triton.jit
def bbq(
    x_ptr,
    h_ptr,
    rsigma_ptr,
    qmz_ptr,
    ROW_SIZE: tl.constexpr,
    BLOCKSIZE: tl.constexpr=128,
    DTYPE: tl.constexpr = "int4"
):
    rid = tl.program_id(axis=0)
    cid = tl.program_id(axis=1)
    xrids = rid * BLOCKSIZE + tl.arange(0, BLOCKSIZE)[:, None]
    xcids = cid * BLOCKSIZE + tl.arange(0, BLOCKSIZE)[None, :]
    xids = xrids * ROW_SIZE + xcids

    hrids = tl.arange(0, BLOCKSIZE)[:, None]
    hcids = tl.arange(0, BLOCKSIZE)[None, :]
    hids = hrids * BLOCKSIZE + hcids

    # load a block of x
    x = tl.load(x_ptr + xids)
    # load pre-computed hadamard matrix shared across the entire kernel
    h = tl.load(h_ptr + hids)
    # load reciprocal of sigma
    rsigma = tl.load(rsigma_ptr)

    # hadamard transform
    v = tl.dot(x, h) * rsigma

    # accelerated normal CDF and rounding
    qmz = normal_cdf_and_round_3bit(v, DTYPE)

    # pack two 4-bit data type into a single byte
    shift = tl.arange(0, 2)[None, None, :] * 4
    qmz = tl.reshape(qmz, (BLOCKSIZE, BLOCKSIZE // 2, 2)) << shift
    qmz = tl.xor_sum(qmz, axis=-1, keep_dims=False)

    # write quantized data to memory
    qmzrids = rid * BLOCKSIZE + tl.arange(0, BLOCKSIZE)[:, None]
    qmzcids = cid * (BLOCKSIZE // 2) + tl.arange(0, BLOCKSIZE // 2)[None, :]
    qmzids = qmzrids * (ROW_SIZE // 2) + qmzcids
    tl.store(qmz_ptr + qmzids, qmz) 
```

Next, we show the definition of function normal cdf and round 3bit. We pre-compute $\Phi ^ { - 1 } \left( \frac { i } { 2 ^ { b } } \right)$ for all possible values of i, and use a binary search method to decide which i should be the result of quantization, and lastly, directly use the binary representation of $i - 2 ^ { b - 1 } - z$ according to Table 6. For 3-bit quantization, the 8 pre-computed values of $\Phi ^ { - 1 } \left( \frac { i } { 2 ^ { b } } \right)$ , for $i \in \{ 0 , 1 , 2 , 3 , 4 , 5 , \bar { 6 } , 7 \}$ , are $\{ - \infty , -$ 1.1503493803760083, -0.6744897501960818, -0.3186393639643752, 0.0, 0.3186393639643752, 0.6744897501960818, 1.1503493803760083}. The corresponding values of $i - 2 ^ { b - 1 } - z$ are {−4, −3, −2, −1, 0, 1, 2, 3}. 

```python
@triton.jit
def normal_cdf_and_round_3bit(v: tl.tensor, DTYPE: tl.constexpr):
    if DTYPE == "int4":
    qmz = tl.where(
    v >= 0.0,
    tl.where(
    v >= 0.6744897501960818,
    tl.where(v >= 1.1503493803760083, 0b0011, 0b0010),
    tl.where(v >= 0.3186393639643752, 0b0001, 0b0000),
    ),
    tl.where(
    v >= -0.6744897501960818,
    tl.where(v >= -0.3186393639643752, 0b1111, 0b1110),
    tl.where(v >= -1.1503493803760083, 0b1101, 0b1100),
    )
).to(tl.int8)
return qmz
elif DTYPE == "fp4":
    qmz = tl.where(
    v >= 0.0,
    tl.where(
    v >= 0.6744897501960818,
    tl.where(v >= 1.1503493803760083, 0b0101, 0b0100),
    tl.where(v >= 0.3186393639643752, 0b0010, 0b0000),
    ),
    tl.where(
    v >= -0.6744897501960818,
    tl.where(v >= -0.3186393639643752, 0b1010, 0b1100),
    tl.where(v >= -1.1503493803760083, 0b1101, 0b1110),
    )
).to(tl.int8)
return qmz 
```

## A.2 EXPLANATION OF EQUATION 9

In this section, we discuss why Equation 9 is useful in ensuring that elements of x and elements of xˆ have approximately the same average magnitude. We first re-iterate the definitions of xˆ: 

$$
\begin{array}{l} v = \mathrm{HT} (x) / \sigma \\ q = \left\lfloor 2 ^ {b} \Phi (v) \right\rfloor - 2 ^ {b - 1} - z \\ \hat {x} = \frac {\gamma}{2 ^ {b - 1}} q \end{array}\tag{10}
$$

We first assume elements of $\operatorname { H T } ( x )$ has approximately the same average magnitude as elements of x, since the Hadamard matrix is orthogonal. This means elements of $\bar { v ^ { } } = \mathrm { H } \bar { \mathrm { T } } ( x ) / \sigma$ has an average magnitude that is σ times smaller than that of x. We next remove the floor operation and the zero point z for simplicity, arriving at the following simplified equation: 

$$
\begin{array}{l} v = \mathbf {H T} (x) / \sigma \\ q = 2 ^ {b} \Phi (v) - 2 ^ {b - 1} \\ \hat {x} = \frac {\gamma}{2 ^ {b - 1}} q = \gamma (2 \Phi (v) - 1) \end{array}\tag{11}
$$

We assume the average magnitude of elements of $2 \Phi ( v ) - 1$ to be $\zeta ^ { * }$ times smaller than that of v. In other words, the average magnitude of elements of $2 \Phi ( v ) - 1$ is approximately $\zeta ^ { * } \sigma$ times smaller than that of x. Therefore, we compensate by setting $\gamma = \zeta ^ { * } \sigma$ , ensuring elements of ${ \hat { x } } =$ $\gamma ( 2 \Phi ( v ) - 1 )$ has approximately the same average magnitude as x. 

To estimate the value of $\zeta ^ { * }$ , we use Equation 9, which expresses $\zeta ^ { * }$ as the optimal ζ that minimizes the MSE between v and $\zeta ( 2 \Phi ( v ) - \bar { 1 } )$ when v follows a Gaussian distribution. Using Monte-Carlo sampling to estimate the MSE, and gradient descent to solve for the arg min, we estimate $\zeta ^ { * } \approx 1 . 6 9 4$ 

## A.3 ABLATION ON EXPONENTIAL MOVING AVERAGE OF $1 / \sigma$

In this section, we show BBQ and BBQ-Fast achieve identical perplexity. 

<table><tr><td></td><td>4-bit</td><td>3-bit</td><td>2-bit</td><td>1-bit</td></tr><tr><td>BBQ</td><td>25.51</td><td>26.55</td><td>31.35</td><td>49.80</td></tr><tr><td>BBQ-Fast</td><td>25.40</td><td>26.55</td><td>31.43</td><td>49.72</td></tr></table>


Table 8: Perplexity of BBQ vs BBQ-Fast, on LLaMA-95M pre-trained with 3 billion C4 tokens.


## A.4 ZERO-SHOT RESULTS

In this section, we present the zero-shot evaluation perplexity in Table 9, which corresponds to the models pre-trained in Table 3. 

<table><tr><td>Model</td><td>Dataset</td><td>Params n+e</td><td>Method</td><td>Bits WA</td><td>Perplexity</td></tr><tr><td>LLaMA</td><td>wikitext</td><td>95M</td><td>None</td><td>16</td><td>56.11</td></tr><tr><td>LLaMA</td><td>wikitext</td><td>95M</td><td>BBQ</td><td>4</td><td>57.87</td></tr><tr><td>LLaMA</td><td>wikitext</td><td>95M</td><td>QuEST</td><td>4</td><td>58.91</td></tr><tr><td>LLaMA</td><td>wikitext</td><td>95M</td><td>LSQ</td><td>4</td><td>61.62</td></tr><tr><td>LLaMA</td><td>wikitext</td><td>95M</td><td>BBQ</td><td>3</td><td>60.18</td></tr><tr><td>LLaMA</td><td>wikitext</td><td>95M</td><td>QuEST</td><td>3</td><td>64.21</td></tr><tr><td>LLaMA</td><td>wikitext</td><td>95M</td><td>LSQ</td><td>3</td><td>68.35</td></tr><tr><td>LLaMA</td><td>wikitext</td><td>95M</td><td>BBQ</td><td>2</td><td>70.19</td></tr><tr><td>LLaMA</td><td>wikitext</td><td>95M</td><td>QuEST</td><td>2</td><td>77.32</td></tr><tr><td>LLaMA</td><td>wikitext</td><td>95M</td><td>LSQ</td><td>2</td><td>80.11</td></tr><tr><td>LLaMA</td><td>wikitext</td><td>95M</td><td>BBQ</td><td>1</td><td>109.66</td></tr><tr><td>LLaMA</td><td>wikitext</td><td>95M</td><td>QuEST</td><td>1</td><td>172.77</td></tr><tr><td>LLaMA</td><td>wikitext</td><td>125M</td><td>None</td><td>16</td><td>50.26</td></tr><tr><td>LLaMA</td><td>wikitext</td><td>125M</td><td>BBQ</td><td>4</td><td>50.50</td></tr><tr><td>LLaMA</td><td>wikitext</td><td>125M</td><td>QuEST</td><td>4</td><td>51.31</td></tr><tr><td>LLaMA</td><td>wikitext</td><td>125M</td><td>LSQ</td><td>4</td><td>53.72</td></tr><tr><td>LLaMA</td><td>wikitext</td><td>125M</td><td>BBQ</td><td>3</td><td>50.94</td></tr><tr><td>LLaMA</td><td>wikitext</td><td>125M</td><td>QuEST</td><td>3</td><td>54.37</td></tr><tr><td>LLaMA</td><td>wikitext</td><td>125M</td><td>LSQ</td><td>3</td><td>58.61</td></tr><tr><td>LLaMA</td><td>wikitext</td><td>125M</td><td>BBQ</td><td>2</td><td>60.47</td></tr><tr><td>LLaMA</td><td>wikitext</td><td>125M</td><td>QuEST</td><td>2</td><td>69.45</td></tr><tr><td>LLaMA</td><td>wikitext</td><td>125M</td><td>LSQ</td><td>2</td><td>70.61</td></tr><tr><td>LLaMA</td><td>wikitext</td><td>125M</td><td>BBQ</td><td>1</td><td>97.34</td></tr><tr><td>LLaMA</td><td>wikitext</td><td>125M</td><td>QuEST</td><td>1</td><td>180.33</td></tr><tr><td>LLaMA</td><td>wikitext</td><td>200M</td><td>None</td><td>16</td><td>40.42</td></tr><tr><td>LLaMA</td><td>wikitext</td><td>200M</td><td>BBQ</td><td>4</td><td>42.56</td></tr><tr><td>LLaMA</td><td>wikitext</td><td>200M</td><td>QuEST</td><td>4</td><td>41.67</td></tr><tr><td>LLaMA</td><td>wikitext</td><td>200M</td><td>LSQ</td><td>4</td><td>4133</td></tr><tr><td>LLaMA</td><td>wikitext</td><td>200M</td><td>BBQ</td><td>3</td><td>41.82</td></tr><tr><td>LLaMA</td><td>wikitext</td><td>200M</td><td>QuEST</td><td>3</td><td>43.95</td></tr><tr><td>LLaMA</td><td>wikitext</td><td>200M</td><td>LSQ</td><td>3</td><td>426.8</td></tr><tr><td>LLaMA</td><td>wikitext</td><td>200M</td><td>BBQ</td><td>2</td><td>49.53</td></tr><tr><td>LLaMA</td><td>wikitext</td><td>200M</td><td>QuEST</td><td>2</td><td>54.12</td></tr><tr><td>LLaMA</td><td>wikitext</td><td>200M</td><td>LSQ</td><td>2</td><td>208.5</td></tr><tr><td>LLaMA</td><td>wikitext</td><td>200M</td><td>BBQ</td><td>1</td><td>80.34</td></tr><tr><td>LLaMA</td><td>wikitext</td><td>200M</td><td>QuEST</td><td>1</td><td>123.19</td></tr><tr><td>LLaMA</td><td>wikitext</td><td>300M</td><td>None</td><td>16</td><td>34.95</td></tr><tr><td>LLaMA</td><td>wikitext</td><td>300M</td><td>BBQ</td><td>4</td><td>38.57</td></tr><tr><td>LLaMA</td><td>wikitext</td><td>300M</td><td>QuEST</td><td>4</td><td>35.26</td></tr><tr><td>LLaMA</td><td>wikitext</td><td>300M</td><td>BBQ</td><td>3</td><td>36.77</td></tr><tr><td>LLaMA</td><td>wikitext</td><td>300M</td><td>QuEST</td><td>3</td><td>37.26</td></tr><tr><td>LLaMA</td><td>wikitext</td><td>300M</td><td>BBQ</td><td>2</td><td>41.24</td></tr><tr><td>LLaMA</td><td>wikitext</td><td>300M</td><td>QuEST</td><td>2</td><td>45.28</td></tr></table>


Table 9: Zero-shot wikitext perplexity.


## A.5 BBQ VS. QUEST

In this section, we illustrate the difference between BBQ and QuEST in Figure 6. 

• BBQ and QuEST share the Hadamard transform (⃝1 ) and RMS normalization (⃝2 ) to reshape the distribution to better match the standard Gaussian. 

• In step ⃝3 , QuEST scales by $\alpha ^ { * }$ , shifts by 0.5, and uses the clip function to limit values within a finite range. BBQ uses the Φ function to limit values within a finite range, while simultaneously maximizing the entropy. 

• In step ⃝4 , both methods use uniform quantization to quantize data. Both methods ensure the quantized data can be stored in a compute-efficient data type (integer). 

• In step ⃝5 , BBQ applies linear scaling, while QuEST reverses all of the operations it did before (RMS Normalization, scaling by $\alpha ^ { * }$ , and shifting by 0.5), as QuEST is a “samedomain” quantizer. BBQ is a “cross-domain” quantizer, so BBQ does not reverse its previous operation such as Φ, shifting by $2 ^ { b } - 1$ and z, multiplication by $2 ^ { b } ,$ , division by σ etc. Instead, BBQ applies linear scaling to scale its range to be within $[ - \gamma , \gamma ]$ where γ is a learnable parameter. 

• In step ⃝6 , QuEST reverses its Hadamard transform (a unitary operation). BBQ does not reverse the Hadamard transform because it is a “cross-domain” quantizer. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/67335314-6215-4a10-a04e-e66551ddf124/25f57dccad01381bda97ac193427d8a0fc0a71f944088efc676589ef413d5467.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/67335314-6215-4a10-a04e-e66551ddf124/55adc1eaaf04ab63c90adc8b7613fd947bc0ed429ec31b61ce83a38f01fea6c0.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/67335314-6215-4a10-a04e-e66551ddf124/64449e1848ae26094c5fbccb48537ca802eac5daf9c2696fa1f9cee0cba1d423.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/67335314-6215-4a10-a04e-e66551ddf124/a395a46b30d28e986e9935e82ecb17a01824095c5a2435c01793bd2a238822b9.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/67335314-6215-4a10-a04e-e66551ddf124/e89d197d414c5e1d9c48477ba39931063511f19f5725c219233586acf01d3809.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/67335314-6215-4a10-a04e-e66551ddf124/4dec9575fa1d11056f72755d33c8364b3b90b8c7be42da67e2ec33bde4802ba0.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/67335314-6215-4a10-a04e-e66551ddf124/2932efcd0256d72174ed56cb9b0443c156fd7a8881abcadb9f93ec93ed65a2f2.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/67335314-6215-4a10-a04e-e66551ddf124/93b872535bfbc2f3b28cadc1fc2bef3660f81807440f4b23e895cc7aecab4f35.jpg)



Figure 6: BBQ vs. QuEST.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/67335314-6215-4a10-a04e-e66551ddf124/cea6c0adf17d6b28cefff1c4b3147e780c1eaf9ed7ef063be85660c4f7b77d03.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/67335314-6215-4a10-a04e-e66551ddf124/8921b5807fc14517c56cdb4370aaf10b045156e131be2ff423d0b07f459e1bbf.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/67335314-6215-4a10-a04e-e66551ddf124/acae8d13e64178d78c983e40bea04c6f4d72a578e77173ffc7488ceae7623d0f.jpg)



(a) Language model: 2-bit LLaMA-95M.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/67335314-6215-4a10-a04e-e66551ddf124/489c6f9a297705a60a9e185f4a27ff355040a30cd8174df69a5986c00579c365.jpg)



(b) Vision model: 2-bit DeiT-Tiny.



Figure 7: Histogram of the absolute value of learned γ in a language model versus a vision model.


## A.6 EXTENSION TO VISION MODELS

While BBQ is designed for language models, we discuss and evaluate a potential extension of BBQ to vision models. When we naively apply BBQ to vision models without any modification, we notice, as shown in Figure 7b, a subset of learned γ have magnitudes extremely close to 0. Since every γ is assigned to a weight channel, $\operatorname { i f } \gamma$ is small, weights from the corresponding channel suffers from gradient vanishing. As shown in Figure 7a, language models do not suffer from such γ collapsing. Instead of making γ a learnable parameter and initializing γ to $\zeta ^ { * } \sigma _ { 0 }$ as discussed in Section 3.4, we propose a BBQ variant, BBQ-Vision, that instead dynamically calculates the value of γ as ζσ on every forward pass, where ζ is a constant hyperparameter. BBQ-Vision prevents γ from collapsing to 0, since for γ to be 0, all weights in that channel must also be 0. In addition, BBQ-Vision still preserves some degree of learnability as the network can modify weights/activations to indirectly control the value of σ. We compare BBQ-Vision against QuEST and LSQ on DeiT (Touvron et al., 2021) and Resnet (He et al., 2015) with various sizes and show our results in Table 10. We use ζ = 2.45. Our preliminary results show BBQ-Vision can consistently outperform QuEST and LSQ, especially at lower precisions (1-bit and 2-bit). 

<table><tr><td rowspan="2">Model</td><td rowspan="2">Params</td><td rowspan="2">Bits WA</td><td colspan="3">BBQ-Vision</td><td colspan="3">QuEST</td><td colspan="3">LSQ</td></tr><tr><td>Ent</td><td>TL</td><td>VA</td><td>Ent</td><td>TL</td><td>VA</td><td>Ent</td><td>TL</td><td>VA</td></tr><tr><td>DEIT-T</td><td>5M</td><td>4</td><td>3.85</td><td>2.84</td><td>74.68</td><td>3.61</td><td>2.91</td><td>74.3</td><td>3.67</td><td>2.88</td><td>74.88</td></tr><tr><td>DEIT-T</td><td>5M</td><td>3</td><td>2.93</td><td>2.89</td><td>74.24</td><td>2.78</td><td>2.98</td><td>72.2</td><td>2.81</td><td>3.01</td><td>71.36</td></tr><tr><td>DEIT-T</td><td>5M</td><td>2</td><td>1.94</td><td>3.05</td><td>70.16</td><td>1.93</td><td>3.18</td><td>67.24</td><td>1.90</td><td>3.23</td><td>65.32</td></tr><tr><td>DEIT-T</td><td>5M</td><td>1</td><td>1.00</td><td>3.52</td><td>53.54</td><td>1.00</td><td>3.67</td><td>47.55</td><td>-</td><td>-</td><td>-</td></tr><tr><td>DEIT-S</td><td>20M</td><td>4</td><td>3.82</td><td>2.53</td><td>79.46</td><td>3.61</td><td>2.51</td><td>80.00</td><td>3.57</td><td>2.52</td><td>80.04</td></tr><tr><td>DEIT-S</td><td>20M</td><td>3</td><td>2.91</td><td>2.48</td><td>80.88</td><td>2.78</td><td>2.62</td><td>79.34</td><td>2.74</td><td>2.64</td><td>78.96</td></tr><tr><td>DEIT-S</td><td>20M</td><td>2</td><td>1.93</td><td>2.76</td><td>77.02</td><td>1.93</td><td>2.82</td><td>76.18</td><td>1.83</td><td>2.87</td><td>74.42</td></tr><tr><td>DEIT-S</td><td>20M</td><td>1</td><td>1.00</td><td>3.18</td><td>66.16</td><td>1.00</td><td>3.37</td><td>59.94</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Resnet-10</td><td>5M</td><td>1</td><td>1.00</td><td>3.37</td><td>63.6</td><td>1.00</td><td>3.40</td><td>62.4</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Resnet-18</td><td>11M</td><td>1</td><td>1.00</td><td>3.12</td><td>73.13</td><td>1.00</td><td>3.14</td><td>72.28</td><td>-</td><td>-</td><td>-</td></tr></table>


Table 10: Entropy (Ent), Training Cross Entropy Loss (TL), and Validation top-1 Accuracy (VA) of vision models pretrained on Imagenet-100 (Chun-Hsiao Yeh, 2022; Russakovsky et al., 2015).


## A.7 VALIDATION LOSS VS. TRAINING PROGRESS

In this section we present validation loss vs. training iteration curves for all of our experiments in Section 4. For visualization purposes, we only show the validation loss for the last 90% iterations, hide the first 10% iterations when learning rate is warming up, and don’t show loss curves for methods that diverged. For all figures, the y-axis is the validation cross entropy loss, and x-axis is iterations (batches). For all experiments, we quantize both weights and activations of linear layers to b bits. In addition, for all QuEST experiments, we use a trust factor of $T = \alpha ^ { * } / ( 2 ^ { b } - 1 )$ as discuss in QuEST (Panferov et al., 2025). 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/67335314-6215-4a10-a04e-e66551ddf124/dcb47ade51fe95fe3dec530407b837f5b0169a7334ed5914434b3520f8bfdc49.jpg)



Figure 8: LLaMA-95M (4-bit) pre-trained on 3 billion C4 tokens (batched over 12 thousand iterations). LSQ is green, QuEST is gray, and BBQ is blue.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/67335314-6215-4a10-a04e-e66551ddf124/527839d06dff32c5f2150e199691c9a219ba88d51a6f8b5ddcb4ae3bb5ce39eb.jpg)



Figure 9: LLaMA-95M (3-bit) pre-trained on 3 billion C4 tokens (batched over 12 thousand iterations). LSQ is pink, QuEST is red, and BBQ is green.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/67335314-6215-4a10-a04e-e66551ddf124/c0ad85b6a531f696ea98452c5a4cd8b67d2fb080733d55fedefeddf9cfe691a1.jpg)



Figure 10: LLaMA-95M (2-bit) pre-trained on 3 billion C4 tokens (batched over 12 thousand iterations). LSQ is yellow, QuEST is purple, and BBQ is orange.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/67335314-6215-4a10-a04e-e66551ddf124/aa65320a8c491ec78a89186deaddccd7cdcbba5c477ce7a299d774b8722a955f.jpg)



Figure 11: LLaMA-95M (1-bit) pre-trained on 3 billion C4 tokens (batched over 12 thousand iterations). QuEST is purple and BBQ is orange.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/67335314-6215-4a10-a04e-e66551ddf124/1baaa454a67bb3b720faeaec1d665957a85ed458044f3862ec354ae9a3b528ee.jpg)



Figure 12: LLaMA-125M (4-bit) pre-trained on 5 billion C4 tokens (batched over 20 thousand iterations). LSQ is red, QuEST is brown, and BBQ is pink.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/67335314-6215-4a10-a04e-e66551ddf124/7010692d10f9084412162140c1602eee675bbcd7d24a06474309c870d8d6924f.jpg)



Figure 13: LLaMA-125M (3-bit) pre-trained on 5 billion C4 tokens (batched over 20 thousand iterations). LSQ is green, QuEST is gray, and BBQ is blue.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/67335314-6215-4a10-a04e-e66551ddf124/e1c7d83be9937cb95d899dfbc91b24c70e1b1ec3ae5deb87e610931dae8a622a.jpg)



Figure 14: LLaMA-125M (2-bit) pre-trained on 5 billion C4 tokens (batched over 20 thousand iterations). LSQ is purple, QuEST is orange, and BBQ is green.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/67335314-6215-4a10-a04e-e66551ddf124/f56d96d2cdb4278800803b006cd9c4d32dd686185ed8acf7473b64549a8af5b1.jpg)



Figure 15: LLaMA-125M (1-bit) pre-trained on 5 billion C4 tokens (batched over 20 thousand iterations). QuEST is pink and BBQ is orange.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/67335314-6215-4a10-a04e-e66551ddf124/61252b6c35481fe01da61c44964cada0d685ca0062282ffef98b26d27e18cdc8.jpg)



Figure 16: LLaMA-200M (4-bit) pre-trained on 10 billion C4 tokens (batched over 40 thousand iterations). QuEST is green and BBQ is brown.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/67335314-6215-4a10-a04e-e66551ddf124/2c64df3177748da53321aae3f490d2accc674cfefbd0a7c5215673a80f6a4fa4.jpg)



Figure 17: LLaMA-200M (3-bit) pre-trained on 10 billion C4 tokens (batched over 40 thousand iterations). QuEST is pink and BBQ is gray.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/67335314-6215-4a10-a04e-e66551ddf124/fb296a1438e747aea958bf5fbc9cd107a5222883ff4128485952de8853e5b175.jpg)



Figure 18: LLaMA-200M (2-bit) pre-trained on 10 billion C4 tokens (batched over 40 thousand iterations). QuEST is orange and BBQ is pink.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/67335314-6215-4a10-a04e-e66551ddf124/3015e0953c48db38a4bc2526e161753ff61693e46e9bda3e56ab4ed953f052c2.jpg)



Figure 19: LLaMA-200M (1-bit) pre-trained on 10 billion C4 tokens (batched over 40 thousand iterations). QuEST is green and BBQ is brown.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/67335314-6215-4a10-a04e-e66551ddf124/3bff94852c9f77f5246163cdef07e28f6290f5debf476b767236d5b37ba4cb63.jpg)



Figure 20: LLaMA-300M (4-bit) pre-trained on 20 billion C4 tokens (batched over 80 thousand iterations). QuEST is green and BBQ is brown.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/67335314-6215-4a10-a04e-e66551ddf124/eeb319d3b6dcbf959fcf4ec3f229a22db504555b61398662ad185a0a78b0bb3a.jpg)



Figure 21: LLaMA-300M (3-bit) pre-trained on 20 billion C4 tokens (batched over 80 thousand iterations). QuEST is orange and BBQ is green.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/67335314-6215-4a10-a04e-e66551ddf124/66c3611d43a56c7ad6c08e90c278a14dd9ec0dabc7740f3c2a4cf30b9c994442.jpg)



Figure 22: LLaMA-300M (2-bit) pre-trained on 20 billion C4 tokens (batched over 80 thousand iterations). QuEST is green and BBQ is red.
