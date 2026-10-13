# BAYESIAN ATTENTION MECHANISM: A PROBABILISTIC FRAMEWORK FOR POSITIONAL ENCODING AND CONTEXT LENGTH EXTRAPOLATION

Arthur S. Bianchessi<sup>1</sup> Yasmin C. Aguirre<sup>1</sup> Rodrigo C. Barros<sup>1,2</sup> Lucas S. Kupssinskü<sup>1</sup> <sup>1</sup>MALTA – Machine Learning Theory and Applications Lab, PUCRS – Porto Alegre, Brazil <sup>2</sup>Kunumi Institute, Brazil 

{arthur.bianchessi, yasmin.cardozo}@edu.pucrs.br {rodrigo.barros, lucas.kupssinsku}@pucrs.br rodrigo.barros@kunumi.com 

## ABSTRACT

Transformer-based language models rely on positional encoding (PE) to handle token order and support context length extrapolation. However, existing PE methods lack theoretical clarity and rely on limited evaluation metrics to substantiate their extrapolation claims. We propose the Bayesian Attention Mechanism (BAM), a theoretical framework that formulates positional encoding as a prior within a probabilistic model. BAM unifies existing methods (e.g., NoPE and ALiBi) and motivates a new Generalized Gaussian positional prior that substantially improves long-context generalization. Empirically, BAM enables accurate information retrieval at 500× the training context length, outperforming previous state-of-the-art context length generalization in long context retrieval accuracy while maintaining comparable perplexity and introducing minimal additional parameters. 

## 1 INTRODUCTION

Transformer-based neural models currently dominate language modeling (LM) due to their superior ability to capture variable-length token dependencies via self-attention. Nevertheless, transformers inherently lack positional information, requiring the incorporation of Positional Encoding (PE). PE is vital, particularly for enabling LMs trained on shorter contexts to generalize to significantly longer sequences during inference—a desirable capability known as context length extrapolation. However, the precise impact of PE on extrapolation remains poorly understood (Kazemnejad et al., 2023). 

Several PE methods have been proposed to facilitate context length extrapolation, including Sinusoidal embeddings (Vaswani, 2017), RoPE (Su et al., 2024), ALiBi (Press et al., 2022), and even the omission of positional encoding entirely (NoPE) (Kazemnejad et al., 2023). Despite reported empirical successes in extrapolation, two critical issues persist: (i) many existing PE techniques are empirically motivated with limited theoretical foundations, and thus their behavior is not well-understood (Liu et al., 2024; Huang et al., 2023); (ii) evaluation methods rely heavily on perplexity, which may inadequately reflect true extrapolation capability, as LMs can achieve low perplexity simply through localized attention patterns, as demonstrated in sliding-window evaluations (Huang et al., 2023). 

To address these issues, we introduce the Bayesian Attention Mechanism (BAM), a theoretical framework that reframes self-attention as an expectation of values computed under a joint probabilistic model of content and position of tokens. Within BAM, PE naturally emerges as a prior distribution over token positions, clarifying the theoretical basis of existing techniques. Notably, we illustrate how NoPE and ALiBi correspond explicitly to Uniform and Laplace positional priors, respectively. 

Leveraging this robust theoretical foundation, we propose a new positional encoding strategy utilizing a Generalized Gaussian prior. Our approach<sup>1</sup> introduces fewer than 1,000 additional parameters yet delivers substantially improved extrapolation performance, demonstrated clearly in retrieval-based tasks and traditional perplexity evaluations. Thus, BAM serves both as a unified theoretical framework for analyzing PE schemes and as a practical method for enhancing long-context attention. 

## 2 BAYESIAN ATTENTION MECHANISM

In this section, we motivate the perspective of framing attention as a Bayesian mechanism, supporting both token content and position information. We show that PE strategies can be seen as priors of a Bayesian attention mechanism, hereby called BAM. 

## 2.1 BAYESIAN ATTENTION AND THE JOINT PROBABILITY DISTRIBUTION $p _ { i j }$

Definition 1. For a fixed query vector $\mathbf { q } _ { i } \in \mathcal { R } ^ { 1 \times d }$ and key-value matrices K, $\mathbf { V } \in \mathcal { R } ^ { L \times d } , \mathbf { a }$ Bayesian Attention Mechanism computes self-attention as an expectation over its values: 

$$
\text { self - attention } (\mathbf {q} _ {i}, \mathbf {K}, \mathbf {V}) = \frac {\exp \left(\operatorname{score} \left(\mathbf {q} _ {i} , \mathbf {k} _ {j}\right)\right)}{\Sigma_ {z} \exp \left(\operatorname{score} \left(\mathbf {q} _ {i} , \mathbf {k} _ {z}\right)\right)} \mathbf {V} = \Sigma_ {j} p _ {i j} \mathbf {v} _ {j} = \mathbf {E} _ {j | i} [ \mathbf {V} ]
$$

Definition 1 states that the attention mechanism can be expressed as an expectation over values of the $i ^ { t h }$ query, where $p _ { i j }$ is the probability of token $j \in [ 1 , L ]$ when attended by token i. This definition is consonant with the self-attention mechanism defined by Vaswani (2017) and with prior attempts to frame self-attention as an expectation over values (Singh & Buckley, 2023), where the scoring function is the scaled dot product and $\sigma \left( \operatorname { s c o r e } ( \mathbf { q } _ { i } , \mathbf { K } ) \right)$ computes $p _ { i j } \ \forall j$ . The term $p _ { i j }$ is usually called the attention weight, however we frame it as a joint probability over content and positions. 

Definition 2. In Bayesian Attention, $p _ { i j }$ is a joint probability over the content of token j $( f _ { \mathrm { c o n t } } )$ and its position relative to query ${ \bf q } _ { i } \left( g _ { \mathrm { p o s } } \right)$ 

$$
p _ {i j} = p (f _ {\text { cont }} (\mathbf {q} _ {i}, \mathbf {k} _ {j}) | g _ {\text { pos }} (i, j)) \times p (g _ {\text { pos }} (i, j))
$$

Definition 2 allows us to interpret $p _ { i j }$ as a joint probability distribution dependent on both token content and position. Together, Definitions 1 and 2 frame positional encoding as a probability distribution over the positions of the tokens within the context. Note that this definition, particularly in $p ( f _ { \mathrm { c o n t } } ( \mathbf { q } _ { i } , \mathbf { k } _ { j } ) | g _ { \mathrm { p o s } } ( i , j ) )$ , encompasses a dependency of content on position. This dependency is trivially modeled by a scalar Z when the scoring function is additive, as detailed below. 

When framing PE as that probability distribution over tokens in a context, we can derive parametrized probability distributions that explain positional encoding strategies such as NoPE (Kazemnejad et al., 2023) and ALiBi (Press et al., 2022), and propose novel PE strategies with known behaviors. 

Theorem 1. If the scoring function of the attention mechanism is additive, i.e., of the form $f _ { \mathrm { c o n t } } ( \mathbf { q } _ { i } , \mathbf { k } _ { j } ) + g _ { \mathrm { p o s } } ( i , j )$ , then $p _ { i j }$ is the product of the marginal probabilities over content and position, dependent on a normalizing scalar $Z { \mathrm { : r } }$ 

Proof. By Definition 1, we have that self-attention $( \mathbf { q } _ { i } , \mathbf { K } , \mathbf { V } ) = \sigma ( \operatorname { s c o r e } ( \mathbf { q } _ { i } , \mathbf { K } ) ) \mathbf { V } = \Sigma _ { j } \mathbf { v } _ { j } p _ { i j } =$ $\mathbf { E } _ { j \mid i } [ V ]$ ]. Following the assumption that the scoring function is $f _ { \mathrm { c o n t } } ( \mathbf { q } _ { i } , \mathbf { k } _ { j } ) + g _ { \mathrm { p o s } } ( i , j )$ , we have: 

$$
\begin{array}{l} p _ {i j} = \sigma (\text {score} (\mathbf {q} _ {i}, \mathbf {k} _ {j})) \\ = \frac {\exp \left(f _ {\text {cont}} (\mathbf {q} _ {i} , \mathbf {k} _ {j}) + g _ {\text {pos}} (i , j)\right)}{\Sigma_ {z} \left(\exp \left(f _ {\text {cont}} (\mathbf {q} _ {i} , \mathbf {k} _ {z}) + g _ {\text {pos}} (i , z)\right)\right)} \\ = \frac {\exp \left(f _ {\text {cont}} (\mathbf {q} _ {i} , \mathbf {k} _ {j})\right) \cdot \exp \left(g _ {\text {pos}} (i , j)\right)}{\Sigma_ {z} \left(\exp \left(f _ {\text {cont}} (\mathbf {q} _ {i} , \mathbf {k} _ {z}) + g _ {\text {pos}} (i , z)\right)\right)} \\ = \frac {\exp \left(f _ {\text {cont}} (\mathbf {q} _ {i} , \mathbf {\tilde {k}} _ {j})\right) \cdot \exp \left(g _ {\text {pos}} (i , j)\right)}{\Sigma_ {z} \left(\exp \left(f _ {\text {cont}} (\mathbf {q} _ {i} , \mathbf {\tilde {k}} _ {z}) + g _ {\text {pos}} (i , z)\right)\right)} \cdot \boxed {\frac {\Sigma_ {z} \left(\exp \left(f _ {\text {cont}} (\mathbf {q} _ {i} , \mathbf {\tilde {k}} _ {z})\right)\right) \cdot \Sigma_ {z} \left(\exp \left(g _ {\text {pos}} (i , z)\right)\right)}{\Sigma_ {z} \left(\exp \left(f _ {\text {cont}} (\mathbf {q} _ {i} , \mathbf {\tilde {k}} _ {z})\right)\right) \cdot \Sigma_ {z} \left(\exp \left(g _ {\text {pos}} (i , z)\right)\right)}} = 1 \end{array}
$$

$$
\begin{array}{l}= \boxed {\frac {\exp \left(f _ {\text {cont}} (\mathbf {q} _ {i} , \mathbf {k} _ {j})\right)}{\Sigma_ {z} \left(\exp \left(f _ {\text {cont}} (\mathbf {q} _ {i} , \mathbf {k} _ {z})\right)\right)}} \cdot \boxed {\frac {\exp \left(g _ {\text {pos}} (i , j)\right)}{\Sigma_ {z} \left(\exp \left(g _ {\text {pos}} (i , z)\right)\right)}} \cdot \boxed {\frac {\Sigma_ {z} \left( \right.\exp \left(f _ {\text {cont}} (\mathbf {q} _ {i} , \mathbf {k} _ {z})\right) \cdot \Sigma_ {z} \left(\exp \left(g _ {\text {pos}} (i , z)\right)\right)}{\Sigma_ {z} \left(\exp \left(f _ {\text {cont}} (\mathbf {q} _ {i} , \mathbf {k} _ {z}) + g _ {\text {pos}} (i , z)\right)\right)}}\\= \boxed {p (f _ {\text {cont}} (\mathbf {q} _ {i}, \mathbf {k} _ {j}))} \cdot \boxed {p (g _ {\text {pos}} (i, j))} \cdot \boxed {\frac {1}{\Sigma_ {z} \left(p (f _ {\text {cont}} (\mathbf {q} _ {i} , \mathbf {k} _ {z})) \cdot p (g _ {\text {pos}} (i , z))\right)}}\\= \boxed {p (f _ {\text {cont}} (\mathbf {q} _ {i}, \mathbf {k} _ {j}))} \cdot \boxed {p (g _ {\text {pos}} (i, j))}\\= \boxed {Z}\end{array}
$$

The distributions governing token content and position are thus dependent by a normalizing scalar factor Z. We can further interpret Z from different perspectives, shedding light on the relationship between token content and position in self-attention (see Appendix K). 

This derivation shows us that adding positional information in the scoring function of the self-attention mechanism leads to a product of probabilities over content and position. We now show that existing PE methods can actually be described by parametrized probability distributions, and that by defining particular distributions we can force the model to explicitly attend to long context. 

## 2.2 POSITIONAL ENCODING AS PRIORS TO BAM

With Theorem 1, we can frame positional encoding as priors to BAM. In particular, we present lemmas that derive NoPE (Kazemnejad et al., 2023) and ALiBi (Press et al., 2022) as specific prior distributions to Bayesian self-attention. 

Lemma 1. The causal mask in decoder models is a special case of BAM prior where Causal $\mathbf { M a s k } \Rightarrow p ( g _ { \mathrm { p o s } } ( i , j ) ) = \mathbf { U n i f o r m } ( 1 , i , j )$ over a given context $\mathbf { x } _ { 1 , \ldots , L }$ 

Lemma 2. ALiBi is a special case of BAM prior where the token position distribution comprises both Uniform and Laplace distributions. 

$$
\mathrm{ALiBi} \Rightarrow p (g _ {\text { pos }} (i, j)) = \text { Uniform } (1, i, j) \cdot \text { Laplace } \left(0, \frac {1}{m}, j - i\right) \text {   over   a   context   } \mathbf {x} _ {1, \dots , L}
$$

Lemma 3. ALiBi becomes local attention as the relative length $| j - i |$ increases. 

$$
\text {If} p _ {i j} = \operatorname{softmax} \left(\mathbf {q} _ {i} \mathbf {K} ^ {\top} + \mathbf {M} _ {i \bullet} + \mathbf {A} _ {i \bullet}\right) \text {then} \lim _ {| j - i | \to \infty} p _ {i j} = 0
$$

Proofs. See Appendix B.1, B.2, and B.3. 

## 2.3 A PE STRATEGY WITH A GENERALIZED GAUSSIAN AS PRIOR

Now we change the distribution over positions to be a Generalized Gaussian Distribution (GGD) and show its advantages over the existing PE methods. We call this new PE method GGD-BAM. 

Let $\mathbf { B } = [ b _ { i j } ] _ { L \times L }$ where $\begin{array} { r } { b _ { i j } = - \left| \frac { j - i - \mu } { \alpha } \right| ^ { \beta } } \end{array}$ , for $i = 1 , \dots , L$ and $j = 1 , \dots , L$ be a matrix of non-linear biases that are added in the scoring function of the self-attention mechanism. This makes $p ( g _ { \mathrm { p o s } } ( i , j ) ) = \mathbf { G G D } ( \mu , \alpha , \beta , j - i )$ , for $\beta > 0$ and $\alpha > 0$ . Self-attention is thus computed as: 

$$
\operatorname{softmax} \left(\mathbf {q} _ {i} \mathbf {K} ^ {\top} + \mathbf {M} _ {i \bullet} + \mathbf {B} _ {i \bullet}\right).
$$

When $\mu = 0 , \beta = 1$ , and $\begin{array} { r } { \alpha = \frac { 1 } { m } } \end{array}$ , we have an instance of ALiBi, i.e., a Laplace prior. 

Lemma 4. GGD-BAM becomes local attention as the relative length $|j - i|$ increases, for any $\beta > 0$ and $\alpha \geq 0$ . If $p_{ij} = \text{softmax}\left(\mathbf{q}_i\mathbf{K}^\top + \mathbf{M}_{i\bullet} + \mathbf{B}_{i\bullet}\right)$ then $\lim_{|j - i| \to \infty} p_{ij} = 0$ 

Proof. See Appendix B.4. 

Theorem 2. GGD-BAM is necessarily capable of seeing more context length than ALiBi: $\frac{\mathrm{GGD - BAM}}{\mathrm{ALiBi}} = \frac{\mathbf{B}_{ij}}{\mathbf{A}_{ij}} = \frac{-|\frac{j - i - \mu}{\alpha}|\beta}{-m|j - i|} < 1$ for some $\mu$ and $\alpha$ , and for $\beta < 1$ 

Proof. See Appendix B.5. 

## 2.4 RELAXING THE REQUIREMENT FOR $\beta > 0$

The requirement for $\beta > 0$ comes from the definition of a GGD. We already showed that by making $0 < \beta \overset { \cdot } { < } 1$ , GGD-BAM can see context beyond ALiBi. Now, we relax this requirement and allow $\beta < 0$ . This effectively increases the size of the tail of the distribution and makes GGD-BAM capable of ignoring local context and focusing on arbitrarily-long context. 

Theorem 3. GGD-BAM ignores local context for any $\beta < 0$ and $\alpha \geq 0$ . $\forall \beta < 0, \forall \alpha \geq 0$ , if $p_{ij} = \text{softmax}\left(\mathbf{q}_i\mathbf{K}^\top + \mathbf{M}_{i\bullet} + \mathbf{B}_{i\bullet}\right)$ then $\lim_{|j - i| \to 0} p_{ij} = 0$ , 

Theorem 4: GGD-BAM takes into account arbitrarily long context for any $\beta < 0$ and $\alpha \geq 0$ . $\forall \beta < 0, \forall \alpha \geq 0$ , If $p_{ij} = \text{softmax}\left(\mathbf{q}_i \mathbf{K}^\top + \mathbf{M}_{i\bullet} + \mathbf{B}_{i\bullet}\right)$ then $\lim_{|j - i| \to \infty} p_{ij} \neq 0$ , 

Proofs. See Appendix B.6 and B.7. 

## 2.5 INTUITIVE EXPLANATION OF BAM

To complement the formal derivations provided so far, we include here an intuitive visualization of the positional priors of BAM. Figure 1 illustrates the probability distribution $p ( g _ { \mathrm { p o s } } ( i , j ) )$ that modulates attention over token positions $j$ for a fixed query position i. These curves reflect the prior belief of the model over the relevance of each position j when computing the attention for token i, as per the decomposition $p _ { i j } \propto p ( f _ { \mathrm { c o n t } } ( \mathbf { q } _ { i } , \mathbf { k } _ { j } ) ) \times p ( g _ { \mathrm { p o s } } ( i , j ) )$ 

Uniform Prior. This distribution corresponds to the absence of any positional inductive bias beyond the causal mask (NoPE). All positions within the causal window $( \mathrm { i } . \mathrm { e } . , j \le i )$ are assigned equal probability, whereas all positions outside the causal window $( j > i )$ are assigned probability zero. 

Since Definition 2 states that $p _ { i j } \propto p ( f _ { \mathrm { c o n t } } ( \mathbf { q } _ { i } , \mathbf { k } _ { j } ) ) \times p ( g _ { \mathrm { p o s } } ( i , j ) )$ , the $p _ { i j }$ of each token outside the causal window is zero. As proved in Theorem 1, this case reduces BAM to the NoPE baseline. The gray-shaded region in Figure 1 represents the non-causal part of the sequence, i.e., positions $j > i ,$ which are masked out in auto-regressive settings. 

ALiBi as a Laplace Prior. The ALiBi mechanism injects a linearly increasing bias into the attention logits, which corresponds to a Laplace prior over relative position $| j - i |$ . The resulting prior has a sharp peak near the query token and rapidly decays over distance. This can be seen in the green curve of Figure 1, which emphasizes local context. As demonstrated in Theorem 2, this behavior limits the attention window, making ALiBi sensitive to short-term dependencies but less effective at capturing long-range interactions. As the relative distance between query and key increases, the probability tends to zero, making ALiBi a local attention mechanism. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/c04e7020-29b8-4cd8-9c96-522d050b6324/fb97eb19cb3abfbd4b83b1c1d407ec89bd7b70e08b6d5c00bd30363e2e44cb0b.jpg)



Figure 1: Visual comparison of different positional priors $p ( g _ { \mathrm { p o s } } ( i , j ) )$ in BAM. Each curve represents the distribution over past token positions for a fixed query $\mathbf { q } _ { i }$ in a fixed token position $i .$


GGD-BAM with $\beta \in ( 0 , 1 )$ The generalized Gaussian prior with fractional exponent produces a heavier-tailed distribution than Laplace. As shown in the pink curve, GGD-BAM maintains significant probability mass across distant tokens. This reflects the theoretical result in Theorem 2, where BAM with $\beta \in ( 0 , 1 )$ exhibits a slower decay rate and is able to attend to longer-range dependencies without diminishing the influence of distant content tokens as fast as ALiBi. The effect of the location parameter $\mu$ over the GGD is to move the peak along the distribution. If $\mu > 0 .$ , it moves the peak beyond the causal mask. Among the parameters of the GGD, $\mu$ offers a smaller impact on the generalization of BAM, and could be fixed in $\mu = 0$ with little harm to perplexity and retrieval. 

GGD-BAM with $\beta < 0 .$ Perhaps the most counterintuitive case is the use of an inverted generalized Gaussian prior. The black curve illustrates a scenario where the prior probability is effectively zero in the vicinity of the query position and sharply concentrated at the far end of the context window. Note that $\beta < 0$ is not a valid parametrization of the GGD in a strict probability sense. However, by relaxing the need for $\beta > 0 .$ , we reach an intriguing theoretical result: the attention mechanism stops looking to local context and shifts to faraway tokens, allowing for arbitrarily-large context scenarios. Even though having attention heads with only negative $\beta$ would make the model blind to local context, attention heads with a negative exponent can act as retrieval heads capable of attending to very long context windows. This is corroborated by our results (see Section 3) where the longest passkey retrieval was achieved by a model in which some attention heads had a negative $\beta .$ 

Relaxing the requirement for $\beta > 0$ is important to increase context length extrapolation, though it is not desirable that all attention heads have $\beta < 0$ . The parametrization $\bar { \beta } < 0$ renders the model to be unable to attend to local context, and this increases its perplexity in language modeling. Therefore GGD-BAM is capable of not only encoding locality but also to explicitly suppress local content in favor of distant information. This behavior is beneficial in settings where important information appears in long-range context, such as causal reasoning and information retrieval. 

Interpretability and Control. One of the central advantages of BAM is that these curves are not merely heuristics but correspond directly to explicit priors over token positions. This makes it possible to visualize, interpret, and even learn the attention pattern preferences of a model. Such a view also offers a principled mechanism for extrapolation beyond training context lengths, by selecting priors that maintain probability mass over long sequences. 

Scalable Softmax (SSMax). Standard Softmax-based attention mechanisms suffer from a phenomenon known as attentionfading, where the attention distribution becomes increasingly uniform as context length grows (Nakanishi, 2025). This occurs because the denominator in the Softmax computation increases with context size n, while the numerator for each token remains constant, leading to vanishing attention peaks. To address this, Nakanishi (Nakanishi, 2025) proposes Scalable Softmax (SSMax), which rescales the attention logits dynamically as a function of sequence length: $\begin{array} { r } { z _ { i } \mapsto \frac { n ^ { s z _ { i } } } { \sum _ { j = 1 } ^ { n } n ^ { s z _ { j } } } = \frac { e ^ { ( s \log n ) z _ { i } } } { \sum _ { i = 1 } ^ { n } e ^ { ( s \log n ) z _ { j } } } } \end{array}$ , where $s \in \mathbb { R }$ is a learnable scalar. Although SSMax is not a PE method, it can be used to address a distinct and complementary challenge in long-context attention, the fading attention. This modification of the softmax function preserves the sharpness of the distribution regardless of input size, mitigating the flattening effect observed in softmax and improving long-context generalization. 

While BAM models positional structure as an explicit prior over token positions, it becomes more susceptible tofading attention as it increases the context length that the LM can attend to. The two techniques are compatible and can be used together, enabling improved context length extrapolation both by a better normalizing factor via SSMax and in the distribution over positions using BAM. This learnable normalizing factor in the softmax function also serves the purpose of learning the normalizing dependence scalar Z that we introduced during the derivation of BAM. 

The BAM matrix B. To use GGD-BAM, the only modification we need to apply in the transformer is to add a relative position based matrix B to the attention score. Figure 2 shows how the attention score is computed for the entire query matrix Q using GGD-BAM. As usual, the causal mask M masks tokens beyond $\mathbf { q } _ { i }$ by adding −∞ to those respective positions. Each value in B is computed according to the relative position $j - i , \alpha , \beta$ and $\mu .$ We fix $\mu = 0$ for most of our results. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/c04e7020-29b8-4cd8-9c96-522d050b6324/ed9283ad94942a22ae36a23c0c20b7b02ecf35ad8a2cd95501960886feda9a55.jpg)


$$
+ \begin{array}{c c c c c} 0 & - \infty & - \infty & - \infty & - \infty \\ 0 & 0 & - \infty & - \infty & - \infty \\ 0 & 0 & 0 & - \infty & - \infty \\ 0 & 0 & 0 & 0 & - \infty \\ 0 & 0 & 0 & 0 & 0 \end{array}
$$

$$
\frac {1}{\alpha} \times \left[ \begin{array}{c c c c c} 0 & 1 & 2 & 3 & 4 \\ - 1 & 0 & 1 & 2 & 3 \\ - 2 & - 1 & 0 & 1 & 2 \\ - 3 & - 2 & - 1 & 0 & 1 \\ - 4 & - 3 & - 2 & - 1 & 0 \end{array} \right] \Bigg | _ {\mathbf {B}} ^ {\mu}
$$


Figure 2: Visual representation of the scoring function in GGD-BAM. The first matrix accounts for the content and the two others for the Uniform and GGD positional priors.


## 3 EMPIRICAL ANALYSIS

We perform an empirical analysis to evaluate the behavior of GGD-BAM in realistic long-context scenarios. We compare BAM and its variant coupled with Scaled Softmax (BAM SSMax) (Nakanishi, 2025) against several widely used PE methods: Sinusoidal (Vaswani, 2017), NoPE (Kazemnejad et al., 2023), RoPE (Su et al., 2024), Local RoPE (RoPE with sliding-window attention), and ALiBi (Press et al., 2022), as well as their versions coupled with Scaled Softmax. 

All models presented in this section contain approximately 120M parameters, including ~25M for input embeddings and ~95M for the transformer layers. Models were trained on the FineWeb 10B dataset (Penedo et al., 2024) using the Mistral-7B v0.3 tokenizer (Jiang et al., 2023), with a training context length of 512 tokens. We evaluate model performance on two tasks: (1) language modeling on long-context samples drawn from FineWeb 10B and Wikipedia (Foundation, 2023); and (2) the Passkey Retrieval task (Mohtashami & Jaggi, 2023), which measures a model’s ability to retrieve specific information from distant positions in the input sequence. 

BAM introduces three learnable parameters, $\theta _ { \alpha } , \theta _ { \beta }$ , and $\theta _ { \mu } { . }$ , for each attention head in each layer. This results in a total overhead of $3 \times$ Heads × Layers parameters. In all experiments reported in this section, we train only $\theta _ { \alpha }$ and $\theta _ { \beta }$ , fixing $\theta _ { \mu } = 0 $ , which amounts to just 384 additional parameters in a 120M parameter model. Further implementation and training details are provided in Appendix C. 

## 3.1 PASSKEY RETRIEVAL

To assess the capability of the LM to access and use long-range information, we evaluate the models on the Passkey Retrieval task (Mohtashami & Jaggi, 2023). This task measures whether a language model can recall a specific five digit number called the “passkey” embedded somewhere within a longer context window. In the passkey retrieval task, we generate 20 sequences, each containing a passkey inserted at uniformly spaced positions: ${ \frac { 0 L } { 1 9 } } , { \frac { 1 L } { 1 9 } } , \dots , L .$ . In the end of the sequence, we append the prompt <The passkey is:> and measure how accurate the model can predict the next five tokens. To avoid inaccuracies due to tokenization, each digit is considered a distinct token. 

As shown in Figure 3, only models trained with BAM retained high accuracy when extrapolating to sequences up to 32,000 tokens. BAM SSMax maintained perfect accuracy across all tested lengths, demonstrating robust access to information throughout the full context window. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/c04e7020-29b8-4cd8-9c96-522d050b6324/272c1a5043af7f8f401fc4462de5bba9dbfde20e0a0f98db43173f37c49522aa.jpg)



Figure 3: Passkey retrieval accuracy with distinct PE. BAM SSMax outperforms all PE methods maintaining perfect accuracy for a context beyond 64× the training context length.


In contrast, all other evaluated PE methods such as Sinusoidal, RoPE, and NoPE rapidly degrade to near-random accuracy beyond their training horizon. Even ALiBi, which showed good perplexity extrapolation (see Appendix D), struggles to maintain retrieval performance at very long context windows. Appendix H.4 shows a similar trend to the evaluated PE methods without SSMax. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/c04e7020-29b8-4cd8-9c96-522d050b6324/65b2997d6cb2b630d06c9587552d4f696eac69e5876cdd730789dc7c82a9d315.jpg)



Figure 4: Passkey retrieval accuracy across context lengths and depths. The horizontal axis represent context length and the vertical axis represents the position of passkey in the context. In the bottom row and last column, we see average accuracy across length and position, respectively.


In Figure 4 we show a heatmap plot with passkey retrieval accuracy considering all possible depths of the passkey. We see that BAM is able to score perfectly in most lengths and depths while being trained only on length 512. BAM only degrades to 0% accuracy in $3 / 5 5$ of the evaluated lengths and depths. Although accuracy is above 80% for 500× the training length, it appears that the model will degrade to zero eventually, however we did not have enough vram to test for longer context. 

The superior performance of BAM in this retrieval setting provides empirical support that its positional prior enables meaningful access to distant content, rather than merely preserving surface-level fluency using local context. In Appendix D, we assess the perplexity of GGD-BAM in comparison to the baselines in Wikitext (Foundation, 2023) and Fineweb (Penedo et al., 2024). In Appendix F, we assess GGD-BAM in the Needle in a Haystack (NIAH) subset of the RULER benchmark (Hsieh et al., 2024), as a means to provide a thorough empirical analysis on long-context extrapolation. 

## 3.2 ATTENTION WEIGHTS FOR $\beta \le 0 , \beta \in ( 0 , 1 )$ AND $\beta \simeq 1$

To provide empirical evidence for the claims presented in Theorems 3 and 4, we visualize attention weights from individual attention heads and their respective values of β in the Passkey Retrieval task. 

To create this visualization, we craft a passkey retrieval prompt of length 841 $( 8 2 5 + 1 6 )$ , where the first 32 tokens are the task prompt–instruction to the model to remember the passkey; the following 25 tokens are the passkey itself; the next 768 tokens arefiller text that the model should ignore; and the last 16 tokens are the retrieval prompt answer that the model should complete with the passkey. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/c04e7020-29b8-4cd8-9c96-522d050b6324/cc608691994ef5ad933c9d8d0363fe623cf2a5b4334b1b93088ac27f01a8e240.jpg)



(a) $\theta _ { \beta } \simeq - 0 . 2 0 5 ( { \bf e . g . } \ \beta \leq 0 )$


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/c04e7020-29b8-4cd8-9c96-522d050b6324/ff968b13b99411de9d5dbc8c174e6cc6243ea3cfd168f828e508f119ad37c91e.jpg)



(b) θ<sub>β</sub> ≃ 0.456 (e.g. β ∈ (0, 1))


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/c04e7020-29b8-4cd8-9c96-522d050b6324/490057003a7334bcb8bba4afdd87bd14f051dbade6d0198a924e1521b67e6c25.jpg)



(c) $\theta _ { \beta } \simeq 1 . 0 3 1$ (e.g. β ≃ 1)



Figure 5: Attention weights from GGD-BAM during the Passkey Retrieval task. When $\beta \leq 0$ attention concentrates on distant keys (e.g., the passkey tokens), suppressing nearby content.


In Figure 5 (a), with $\beta \leq 0$ , the attention head effectively ignores local context and sharply focuses on distant tokens—including the passkey—as predicted by our theoretical formulation. This behavior aligns with Theorems 3 and 4, which show that for $\beta \leq 0 ,$ , the GGD-BAM prior suppresses attention weights near the query token and sustains probability mass across long distances. As a result, these heads act as retrieval specialists, increasing the performance of the model for passkey retrieval. 

In Figure 5 (b), with $\beta \in ( 0 , 1 )$ , attention exhibits long-tail behavior, allocating attention mass more evenly across both nearby and distant positions. Compared to $\beta \simeq 1$ , this head decays more slowly, retaining mid- and long-range dependencies. This supports our claim in Theorem 2 that GGD-BAM with fractional β values can attend to longer contexts than ALiBi (β ≃ 1). 

In Figure 5 (c), with $\beta \simeq 1$ , the attention pattern is highly localized, closely resembling ALiBi’s linear bias. Attention focuses on immediate neighbors, with a rapid decay over distance. This configuration is suitable for capturing local dependencies but is inadequate for long-range retrieval, as evidenced by its poor performance on the passkey task at large context lengths. 

The visualizations in Figure 5 show that BAM is highly interpretable. The negative $\theta _ { \beta }$ distribution that was conjectured to improve long-context retrieval appeared after training, and during inference it effectively caused higher attention weights towards the passkey that was further in the context. 

## 3.3 $\theta _ { \beta }$ AND $\theta _ { \alpha }$ TRENDS AFTER TRAINING

We now analyze the trends of $\theta _ { \beta }$ and $\theta _ { \alpha }$ after training. Our first analysis focus on the 120M model trained with distinct context sizes. Figure 6 shows an interest trend where we identify three linearly separable clusters of parameters: the first cluster with $\theta _ { \beta } > 0 .$ , where each attention head works similarly to a Laplace distribution (ALiBi); the second cluster with $- 0 . 6 \leq \theta _ { \beta } \leq 0 ,$ , which works as a retrieval head; the third cluster has fewer instances than the other two, with $\theta _ { \beta } < - 0 . 6$ . We conjecture that this cluster works as a more aggressive retrieval head. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/c04e7020-29b8-4cd8-9c96-522d050b6324/26b290fef74226f0e9ab7ce29e4307dd32a681f6df9257069cbecbec7d758610.jpg)



Figure 6: Trend of $\theta _ { \beta } \times \theta _ { \alpha }$ regarding the 120M model trained on 512, 1024, and 2048 tokens.


When we analyze distinct model scales, the same three clusters emerge. Figure 7 shows the same clusters identified in Figure $^ { 6 , }$ providing evidence that these probability distribution over positions are stable and transferable across many tasks. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/c04e7020-29b8-4cd8-9c96-522d050b6324/e1ea785a1b3e6babe283dc0cb0628b181694f37e214415dd62cec0823d907c21.jpg)



Figure 7: Trend of $\theta _ { \beta } \times \theta _ { \alpha }$ on different model scales.


In Appendices D, F, and H, we present additional empirical results for perplexity, for long-context performance on the Ruler Benchmark, as well as several additional ablation studies. We found that BAM has similar perplexity to ALiBi and has no measurable impact on inference time, while outperforming every baseline in context extrapolation across all evaluated tasks of the Ruler benchmark. 

## 4 RELATED WORK

Several strategies for PE have been developed to allow Transformer-based LMs to encode token order. The most used PE methods are described below: 

Sinusoidal Positional Encoding. Introduced by Vaswani et al. (Vaswani, 2017), sinusoidal encodings inject fixed positional information into the model by adding position-dependent vectors to token embeddings. In the self-attention mechanism, each ${ \bf q } _ { i } , { \bf k } _ { j }$ and $\mathbf { v } _ { j }$ have both content and positional information. BAM, instead, explicitly models position as a separate probabilistic component and does not add positional information directly into $\mathbf { q } _ { i } , \mathbf { k } _ { j }$ , and $\mathbf { v } _ { j }$ 

Rotary Position Embedding (RoPE). RoPE (Su et al., 2024) encodes absolute positions through complex-valued rotations applied directly to Q and K. RoPE encodes relative position by phaseshifting the token representations before the dot product. While RoPE introduces position at the dot-product level and preserves relative distance structure, it does not decouple content and position semantically. Han & Ji (2025) show that RoPE asymptotically disentangles semantics and position information in additive components in self-attention logits. Thus, we can see that RoPE asymptotically approximates BAM as the context length increases, though it lacks the flexibility and interpretability of assigning positional priors as established in our formulation of BAM. 

T5 Relative Bias. T5 (Raffel et al., 2020) avoids absolute encodings entirely and instead learns a bias $b ( i - j )$ for each relative distance i − j that is added to the attention logits.This PE method could be viewed in BAM as an empirical non-parametric distribution over the positions $p ( g _ { \mathrm { p o s } } ( i , j ) )$ . However, both our theoretical grounding and experiments show that a parametric distribution with fewer trainable parameters can achieve superior context length extrapolation. 

No Positional Encoding (NoPE). Haviv et al. (2022) and Kazemnejad et al. (2023) examined transformers without explicit PE. Although the attention mechanism is purely content-driven, the authors were the first to highlight that the causal mask $\mathbf { M } _ { i j }$ (often not represented explicitly in the notation of other PE methods) in decoder-based transformers is sufficient to derive both absolute and relative PE.We showed NoPE to be a special case of BAM under a uniform positional prior. 

Attention with Linear Biases (ALiBi). ALiBi (Press et al., 2022) injects a linear positional penalty into the attention scores. This formulation is interesting because it allows context length extrapolation in language modeling without introducing learnable parameters to the LM. We show that ALiBi is a special case of BAM with a Laplacian prior and we test it as an initialization strategy in our ablation study (see AppendixC). We explain how ALiBi maintains low perplexity in longer context windows and why it fails to retrieve information as it becomes local attention as the context length increases. 

## 5 CONCLUSION

We introduced BAM, a principled probabilistic framework that reconceptualizes positional encoding as a prior over token positions within attention. By framing the attention mechanism as a factorized joint distribution over content and position, BAM not only offers a theoretical grounding for existing methods such as NoPE and ALiBi, but also motivates new families of positional priors. Our proposed Generalized Gaussian prior GGD-BAM significantly improves context length extrapolation in passkey retrieval task by more than 25× compared to other PEs while maintaining low perplexity. 

Despite its simplicity—increasing the parameter count of the model in negligible 0.00032% trainable parameters—GGD-BAM enables models to attend over significantly longer context windows without direct exposure during training. Experiments on FineWeb and Wikipedia show that BAM is uniquely able to recover distant information even at 250,000-token sequences, where other methods collapse. Moreover, our theoretical results demonstrate that BAM can express attention patterns that emphasize distant context or suppress locality, offering a new axis of inductive bias design for Transformers. 

Future work includes applying BAM to larger models, further exploring the interpretability of learned positional priors, and extending the BAM framework to multi-modal input settings. Additionally, it remains an open question whether the extrapolation capabilities induced by BAM are preserved, or potentially enhanced, after instruction and preference fine-tuning. Investigating BAM under these downstream adaptation regimes is crucial for understanding its robustness in real-world applications. We further discuss all limitations of this work in Appendix I. 

## ACKNOWLEDGMENTS

This study was financed in part by the Coordination for the Improvement of Higher Education Personnel (CAPES) — Finance Code 001; by Conselho Nacional de Desenvolvimento Científico e Tecnológico (CNPq)— Grant Number: 443072/2024-8; and by Fundação de Amparo à Pesquisa do Estado do Rio Grande do Sul (FAPERGS) — Grant Number: 25/2551-0000891-3. 

This work was supported by Kunumi Institute. The authors thank the institution for its financial support and commitment to advancing scientific research 

## REFERENCES



Yushi Bai, Shangqing Tu, Jiajie Zhang, Hao Peng, Xiaozhi Wang, Xin Lv, Shulin Cao, Jiazheng Xu, Lei Hou, Yuxiao Dong, et al. Longbench v2: Towards deeper understanding and reasoning on realistic long-context multitasks. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3639–3664, 2025. 





Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018. 





Wikimedia Foundation. Wikimedia downloads version 20231101.en, 2023. URL https://dumps. wikimedia.org. 





Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024. 





Chi Han and Heng Ji. Computation mechanism behind llm position generalization. arXiv preprint arXiv:2503.13305, 2025. 





Adi Haviv, Ori Ram, Ofir Press, Peter Izsak, and Omer Levy. Transformer Language Models without Positional Encodings Still Learn Positional Information. arXiv preprint arXiv:2203.16634, December 2022. doi: 10.48550/arXiv.2203.16634. URL http://arxiv.org/abs/2203. 16634. 





Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020. 





Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, and Boris Ginsburg. Ruler: What’s the real context size of your long-context language models? arXiv preprint arXiv:2404.06654, 2024. 





Yutong Hu, Quzhe Huang, Mingxu Tao, Chen Zhang, and Yansong Feng. Can perplexity reflect large language model’s ability in long text understanding? arXiv preprint arXiv:2405.06105, 2024. 





Yunpeng Huang, Jingwei Xu, Junyu Lai, Zixu Jiang, Taolue Chen, Zenan Li, Yuan Yao, Xiaoxing Ma, Lijuan Yang, Hao Chen, et al. Advancing transformer architecture in long-context large language models: A comprehensive survey. arXiv preprint arXiv:2311.12351, 2023. 





Albert Q Jiang, A Sablayrolles, A Mensch, C Bamford, D Singh Chaplot, Ddl Casas, F Bressand, G Lengyel, G Lample, L Saulnier, et al. Mistral 7b. arxiv. arXiv preprint arXiv:2310.06825, 10, 2023. 





Amirhossein Kazemnejad, Inkit Padhi, Karthikeyan Natesan Ramamurthy, Payel Das, and Siva Reddy. The impact of positional encoding on length generalization in transformers. Advances in Neural Information Processing Systems, 36:24892–24928, 2023. 





Sehoon Kim, Coleman Hooper, Thanakul Wattanawong, Minwoo Kang, Ruohan Yan, Hasan Genc, Grace Dinh, Qijing Huang, Kurt Keutzer, Michael W Mahoney, et al. Full stack optimization of transformer inference: a survey. arXiv preprint arXiv:2302.14017, 2023. 





Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the middle: How language models use long contexts. Transactions of the Association for Computational Linguistics, 12:157–173, 02 2024. ISSN 2307-387X. doi: 10.1162/tacl_a_00638. URL https://doi.org/10.1162/tacl_a_00638. 



Amirkeivan Mohtashami and Martin Jaggi. Random-access infinite context length for transformers. Advances in Neural Information Processing Systems, 36:54567–54585, 2023. 

Ken M Nakanishi. Scalable-softmax is superior for attention. arXiv preprint arXiv:2501.19399, 2025. 



Guilherme Penedo, Hynek Kydlícek, Anton Lozhkov, Margaret Mitchell, Colin A Raffel, Leandroˇ Von Werra, Thomas Wolf, et al. The fineweb datasets: Decanting the web for the finest text data at scale. Advances in Neural Information Processing Systems, 37:30811–30849, 2024. 





Ofir Press, Noah Smith, and Mike Lewis. Train short, test long: Attention with linear biases enables input length extrapolation. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=R8sQPpGCv0. 





Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal ofmachine learning research, 21(140):1–67, 2020. 



Noam Shazeer. Glu variants improve transformer. arXiv preprint arXiv:2002.05202, 2020. 

Ryan Singh and Christopher L Buckley. Attention: Marginal probability is all you need? arXiv preprint arXiv:2304.04556, 2023. 

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024. 

A Vaswani. Attention is all you need. Advances in Neural Information Processing Systems, 2017. 

Biao Zhang and Rico Sennrich. Root mean square layer normalization. Advances in Neural Information Processing Systems, 32, 2019. 

## A PRELIMINARIES AND NOTATION

We denote by $\mathbf { x } _ { i }$ the $i ^ { \mathrm { { t h } } }$ token in a sequence of length $L ,$ written as $\mathbf { x } _ { 1 , \ldots , L }$ . Each token is projected into query, key, and value via learned matrices: the query $\mathbf { q } _ { i } \in \mathbb { R } ^ { 1 \times d }$ is obtained from $\mathbf { x } _ { i } \mathbf { W _ { q } }$ , and the full key and value matrices are given by $\mathbf { K } = \mathbf { X } \mathbf { W _ { k } } \in \mathbb { R } ^ { L \times d }$ and $\mathbf { V } = \mathbf { X } \mathbf { W _ { v } } \in \mathbb { R } ^ { L \times d }$ , respectively. 

We define M as the standard causal mask used to enforce auto-regressive decoding constraints. We use • to denote a slice from a matrix. For instance, $\mathbf { M } _ { i } .$ is the causal mask line vector for a single q<sub>i</sub>. 

In our formulation, we interpret attention weights as a joint probability distribution $p _ { i j }$ over two dependent components: content and position. The random variable $f _ { \mathrm { c o n t } } ( \mathbf { q } _ { i } , \mathbf { k } _ { j } )$ denotes the content similarity between the $i ^ { \mathrm { { t h } } }$ query and the $j ^ { \mathrm { t h } }$ key, modeled by the conditional probability $p \big ( f _ { \mathrm { c o n t } } \big ( \mathbf { q } _ { i } , \mathbf { k } _ { j } \big ) \big )$ . Similarly, the positional relation is modeled by the random variable $g _ { \mathrm { p o s } } ( i , j )$ , which represents the relative position of token j with respect to token i, captured by the prior distribution $p ( g _ { \mathrm { p o s } } ( i , j ) )$ . 

It is a convention to denote the scoring function of the non-normalized logits in self-attention as score $\left( { \bf q } _ { i } , { \bf K } \right) = { \bf q } _ { i } { \bf K } ^ { \top } + { \bf M } _ { i } \mathbf { . }$ . To follow this convention, we adapt the notation used in ALiBi to present the linear biases in matrix form, so the scoring function becomes: score $( \mathbf { q } _ { i } , \mathbf { K } ) \ =$ $\mathbf { q } _ { i } \bar { \mathbf { K } } ^ { \top } + \mathbf { M } _ { i \bullet } + \mathbf { A } _ { i \bullet }$ . Finally, we introduce the BAM score $( \bar { \bf q } _ { i } , \bf K ) = q _ { i } K ^ { \top } + M _ { i \bullet } + B _ { i \bullet }$ by adding a matrix of non-linear biases to the scoring function. 

Another convention we adopt is $\begin{array} { r } { \sigma ( \mathbf { z } ) = \mathrm { s o f t m a x } ( \mathbf { z } ) = \frac { \exp ( \mathbf { z } _ { i } ) } { \Sigma _ { j } \exp ( \mathbf { z } _ { j } ) } } \end{array}$ 

Positional Priors The prior over positions, $p ( g _ { \mathrm { p o s } } ( i , j ) )$ , is modeled using parametric distributions over relative positions $| j - i |$ . We consider the following parametric distributions: 

• Uniform distribution: assigns equal probability mass to all valid positions before the current token: 

$$
p \left(g _ {\text { pos }} (i, j)\right) = \operatorname{Uniform} (a, b, x) \propto \mathbb {I} [ a \leq x \leq b ].
$$

• Laplace distribution: decays exponentially with distance, controlled by a scale parameter $b > 0 \colon$ 

$$
p \left(g _ {\text { pos }} (i, j)\right) = \operatorname{Laplace} (\mu = 0, \alpha , x) \propto \exp \left(- \frac {| x |}{\alpha}\right).
$$

• Generalized Gaussian distribution (GGD): introduces a shape parameter $\beta > 0$ and scale $\alpha > 0$ , allowing flexible control over decay behavior: 

$$
p (g _ {\text { pos }} (i, j)) = \text { GGD } (\mu , \alpha , \beta , x) \propto \exp \left(- \left| \frac {x - \mu}{\alpha} \right| ^ {\beta}\right).
$$

This formulation generalizes several priors: when $\beta = 1$ it recovers the Laplace distribution; when $\beta = 2$ it becomes Normal; larger values of $\beta$ yield sharper, more localized priors; lower values of $\beta$ yield large tailed distributions. 

## B PROOFS

## B.1 NOPE IS A UNIFORM PRIOR TO BAM

Lemma 1: The causal mask in decoder models is a special case of BAM prior where Causal Mask $\Rightarrow p(g_{\mathrm{pos}}(i,j)) = \mathrm{Uniform}(1,i,j)$ over a given context $\mathbf{x}_{1,\dots,L}$ 

Proof. The causal mask in a decoder model changes the scores of every token $\mathbf { x } _ { i + 1 , i + 2 , . . . , L } \ t _ { 0 } - \mathrm { i n f } ;$ 

$$
\text { causal - self - attention } = \text { softmax } (\mathbf {q} _ {i} \mathbf {K} ^ {\top} + \mathbf {M} _ {i \bullet}) \mathbf {V}, \text { where } \mathbf {M} _ {i \bullet} = \left[ \begin{array}{c c} 0; & j \leq i \\ - \infty ; & \text { otherwise } \end{array} \right]
$$

Since $\mathbf { q } _ { i } \mathbf { K } ^ { \top }$ has only content information and $\mathbf { M } _ { i \bullet }$ has only positional information, we can use Theorem 1 to rewrite it as 

$$
\frac {\exp (\mathbf {q} _ {i} \mathbf {k} _ {j} ^ {\top})}{\Sigma_ {z} (\exp (\mathbf {q} _ {i} \mathbf {k} _ {z} ^ {\top}))} \cdot \frac {\exp (\mathbf {M} _ {i \bullet})}{\Sigma_ {z} (\exp (\mathbf {M} _ {\mathbf {i z}}))} \cdot \frac {1}{Z}
$$

Only the softmax $( \mathbf { M } _ { i \bullet } )$ term depends on the token position, so it is equivalent to $p ( g _ { \mathrm { p o s } } ( i , j ) )$ 

$$
p (g _ {\mathrm{pos}} (i, j)) = \frac {\exp (\mathbf {M} _ {i \bullet})}{\Sigma_ {z} (\exp (\mathbf {M} _ {\mathbf {i z}}))} = \left\{ \begin{array}{l l} \frac {1}{i}, & \text { if } j \leq i \\ 0, & \text { otherwise } \end{array} \right.,
$$

which is a Uniform distribution over tokens $\mathbf { x } _ { 1 \ldots i } .$ 

## B.2 ALIBI IS A PRIOR COMPRISING BOTH UNIFORM AND LAPLACE DISTRIBUTIONS

Lemma 2: ALiBi is a special case of BAM prior where the token position distribution comprises both Uniform and Laplace distributions. $\text{ALiBi} \Rightarrow p(g_{\text{pos}}(i,j)) = \text{Uniform}(1,i,j) \cdot \text{Laplace}\left(0, \frac{1}{m}, j - i\right)$ over a context $x_{1,\ldots,L}$ 

Proof. Let $\mathbf { A } = [ a _ { i j } ] _ { L \times L }$ where $a _ { i j } = - m | j - i |$ , for $i = 1 , \ldots , L$ and $j = 1 , \dots , L$ be the matrix with linear biases defined in ALiBi PE. The self-attention mechanism with ALiBi as PE is computed as: 

$$
\operatorname{softmax} \left(\mathbf {q} _ {i} \mathbf {K} ^ {\top} + \mathbf {M} _ {i \bullet} + \mathbf {A} _ {i \bullet}\right) \mathbf {V}
$$

where $\mathbf { M } _ { i \bullet }$ is the causal mask and $- m | j - i |$ are the linear biases added by ALiBi. According to Theorem 1 and Lemma 1, we can rewrite the softmax as: 

$$
\frac {\exp (\mathbf {q} _ {i} \mathbf {k} _ {j} ^ {\top})}{\Sigma_ {z} (\exp (\mathbf {q} _ {i} \mathbf {k} _ {z} ^ {\top}))} \cdot \frac {\exp (\mathbf {M} _ {i j})}{\Sigma_ {z} (\exp (\mathbf {M} _ {\mathbf {i z}}))} \cdot \frac {\exp (\mathbf {A} _ {i j})}{\Sigma_ {z} (\exp (\mathbf {A} _ {i z}))} \frac {1}{Z _ {\mathrm{ALiBi}}}
$$

where $Z _ { \mathrm { A L i B i } }$ is a scaling (normalizing) factor. Lemma 1 further allows us to substitute the causal mask term with a uniform distribution: 

$$
\frac {\exp (\mathbf {q} _ {i} \mathbf {k} _ {j} ^ {\top})}{\Sigma_ {z} (\exp (\mathbf {q} _ {i} \mathbf {k} _ {z} ^ {\top}))} \cdot \text {Uniform} (1, i, j) \cdot \frac {\exp (\mathbf {A} _ {i j})}{\Sigma_ {z} (\exp (\mathbf {A} _ {i z}))} \frac {1}{Z _ {\mathrm{ALiBi}}}
$$

The softmax $\left( - m | j - i | \right)$ term has no content information, just positional, so we further work on it: 

$$
\begin{array}{l} \frac {\exp (\mathbf {A} _ {i j})}{\Sigma_ {z} (\exp (\mathbf {A} _ {i z}))} = \frac {\exp (- m | j - i |)}{\Sigma_ {z} \exp (- m | z - i |)} \\ = \frac {\frac {m}{2} \exp (- m | j - i |)}{\Sigma_ {z} \frac {m}{2} \exp (- m | z - i |)} \\ = \frac {\text { Laplace } (0 , \frac {1}{m} , j - i)}{\Sigma_ {z} \frac {m}{2} \exp (- m | z - i |)} \\ = \text { Laplace } (0, \frac {1}{m}, j - i) \end{array}
$$

We can drop the scalar normalizing denominator $\begin{array} { r } { \sum _ { z } \frac { m } { 2 } \exp \big ( - m | z - i | \big ) } \end{array}$ since $Z _ { \mathrm { A L i B i } }$ accounts for the normalization of the whole expression. Back to Definition 2, we have $p ( g _ { \mathrm { p o s } } ( i , j ) ) \ =$ Uniform $( 1 , i , j )$ · Laplace $\begin{array} { r } { \left( 0 , \frac { 1 } { m } , j - i \right) } \end{array}$ □ 

## B.3 ALIBI IS LOCAL ATTENTION FOR LARGE $| j - i |$ LENGTHS

Lemma 3: ALiBi becomes local attention as the relative length $| j - i |$ increases. 

$$
\text {   If   } p _ {i j} = \operatorname{softmax} \left(\mathbf {q} _ {i} \mathbf {K} ^ {\top} + \mathbf {M} _ {i \bullet} + \mathbf {A} _ {i \bullet}\right) \text {   then   } \lim _ {| j - i | \to \infty} p _ {i j} = 0
$$

Proof. We prove this lemma for a fixed query $\mathbf { q } _ { i }$ . The scoring function of ALiBi has three components: $\mathbf { q } _ { i } \mathbf { K } ^ { \top } , \mathbf { M } _ { i \bullet }$ , and $\mathbf { A } _ { i \bullet }$ . Let us take the limit of the scoring function: 

$$
\lim _ {| j - i | \to \infty} \left(\mathbf {q} _ {i} \mathbf {K} ^ {\top} + \mathbf {M} _ {i \bullet} + \mathbf {A} _ {i \bullet}\right) = \mathbf {q} _ {i} \mathbf {K} ^ {\top} + \mathbf {M} _ {i \bullet} + \lim _ {| j - i | \to \infty} (\mathbf {A} _ {i \bullet})
$$

We drop the limit in the causal mask $\mathbf { M } _ { i \bullet }$ as the only effect of increasing the context size and the distance between i and $j$ in the causal mask is making the mask bigger in size, but it stills follows the same formation law with 0 to the left of the query and −∞ elsewhere. 

$$
\lim _ {| j - i | \to \infty} (\mathbf {A} _ {i \bullet}) = \lim _ {| j - i | \to \infty} (- m | j - i |) = - \infty
$$

When we plug $- \infty$ back into the scoring function we see that it becomes $- \infty$ , and consequently the softmax becomes 0. 

$$
\lim _ {| j - i | \rightarrow \infty} p _ {i j} = \lim _ {| j - i | \rightarrow \infty} \operatorname{softmax} \left(\mathbf {q} _ {i} \mathbf {K} ^ {\top} + \mathbf {M} _ {i \bullet} + \mathbf {A} _ {i \bullet}\right) = 0
$$

## B.4 GGD-BAM IS LOCAL ATTENTION FOR LARGE $| j - i |$ LENGTHS

Lemma 4: GGD-BAM becomes local attention as the relative length $|j - i|$ increases, for any $\beta > 0$ and $\alpha > 0$ . If $p_{ij} = \mathrm{softmax}\left(\mathbf{q}_i\mathbf{K}^\top + \mathbf{M}_{i\bullet} + \mathbf{B}_{i\bullet}\right)$ then $\lim_{|j - i| \to \infty} p_{ij} = 0$ 

Proof. This proof is similar to Lemma 3. We prove this lemma for a fixed query $\mathbf { q } _ { i }$ . Let $\mu = 0$ . The scoring function of GGD-BAM has three components $\mathbf { q } _ { i } \mathbf { K } ^ { \top } , \mathbf { M } _ { i \bullet }$ , and $\mathbf { B } _ { i \bullet }$ , lets take the limit of the scoring function and see how it behaves: 

$$
\lim _ {| j - i | \to \infty} \left(\mathbf {q} _ {i} \mathbf {K} ^ {\top} + \mathbf {M} _ {i \bullet} + \mathbf {B} _ {i \bullet}\right) = \mathbf {q} _ {i} \mathbf {K} ^ {\top} + \mathbf {M} _ {i \bullet} + \lim _ {| j - i | \to \infty} (\mathbf {B} _ {i \bullet})
$$

We drop the limit in the causal mask $\mathbf { M } _ { i \bullet }$ as the only effect of increasing the context size and the distance between i and $j$ in the causal mask is making the mask bigger in size, but it stills follows the same formation law with 0 to the left of the query and $- \infty$ elsewhere. 

$$
\lim _ {| j - i | \rightarrow \infty} (\mathbf {B} _ {i \bullet}) = \lim _ {| j - i | \rightarrow \infty} \left(- \left| \frac {j - i}{\alpha} \right| ^ {\beta}\right) = - \infty
$$

When we plug $\infty$ back into the scoring function we see that it becomes $- \infty .$ , and consequently the softmax becomes 0. 

$$
\lim _ {| j - i | \rightarrow \infty} p _ {i j} = \lim _ {| j - i | \rightarrow \infty} \operatorname{softmax} \left(\mathbf {q} _ {i} \mathbf {K} ^ {\top} + \mathbf {M} _ {i \bullet} + \mathbf {B} _ {i \bullet}\right) = 0
$$

□ 

## B.5 GGD-BAM SEES MORE CONTEXT THAN ALIBI

$$
\frac {\mathrm{GGD-BAM}}{\mathrm{ALiBi}} = \frac {\mathbf {B} _ {i j}}{\mathbf {A} _ {i j}} = \frac {- \left| \frac {j - i - \mu}{\alpha} \right| ^ {\beta}}{- m | j - i |} <   1 \text {for some} \mu \text {and} \alpha , \text {and for} \beta <   1
$$

Proof. For GGD-BAM to see more context than $\mathbf { A L i B i } ,$ , it must be the case that the ratio between $\mathbf { B } _ { i j }$ and $\mathbf { A } _ { i j } \ { \frac { - \left| { \frac { j - i - \mu } { \alpha } } \right| ^ { \beta } } { - m \left| j - i \right| } }$ is less than 1 for some $\beta < 1$ and for some α and $\mu .$ Let $\begin{array} { r } { \mu = 0 , \alpha = { \frac { 1 } { \sqrt [ \beta ] { m } } } } \end{array}$ and $\beta \in ( 0 , 1 )$ , then we have: 

$$
\begin{array}{r} \frac {\mathbf {B} _ {i j}}{\mathbf {A} _ {i j}} = \frac {- \left| \frac {j - i}{\alpha} \right| ^ {\beta}}{- m | j - i |} \\ = \frac {- m | j - i | ^ {\beta}}{- m | j - i |} \\ = | j - i | ^ {\beta - 1} \\ <   1 \end{array}
$$

Since the ratio between BAM and ALiBi can be less then 1, BAM shrinks at a slower rate than ALiBi, effectively capturing longer contexts as $| j - i |$ increases. □ 

## B.6 GGD-BAM IGNORES LOCAL CONTEXT FOR $\beta < 0 \mathrm { A N D } \alpha > 0$

Theorem 3: GGD-BAM ignores local context for any $\beta < 0$ and $\alpha > 0$ 

$$
\forall \beta <   0, \forall \alpha > 0, \text {   If   } p _ {i j} = \operatorname{softmax} \left(\mathbf {q} _ {i} \mathbf {K} ^ {\top} + \mathbf {M} _ {i \bullet} + \mathbf {B} _ {i \bullet}\right) \text {   then   } \lim _ {| j - i | \to 0} p _ {i j} = 0,
$$

Proof. This proof is similar to Lemmas 3 and 4. We prove this lemma for a fixed query $\mathbf { q } _ { i }$ . Since i is fixed for a query, $| j - i | \to 0$ implies that the token $j$ and i are the same token. The scoring function of GGD-BAM has three components $\mathbf { q } _ { i } \mathbf { K } ^ { \top } , \mathbf { M } _ { i \bullet }$ , and $\mathbf { B } _ { i \bullet }$ , lets take the limit of the scoring function and see how it behaves: 

$$
\lim _ {| j - i | \to 0} \left(\mathbf {q} _ {i} \mathbf {K} ^ {\top} + \mathbf {M} _ {i \bullet} + \mathbf {B} _ {i \bullet}\right) = \mathbf {q} _ {i} \mathbf {K} ^ {\top} + \mathbf {M} _ {i \bullet} + \lim _ {| j - i | \to 0} (\mathbf {B} _ {i \bullet})
$$

We drop the limit in the causal mask $\mathbf { M } _ { i \bullet }$ as the only effect of i and $j$ being the same token is that ${ \bf { M } } _ { i j } = { \bf { \bar { 0 } } }$ 

When the shape parameter $\beta$ is negative, then we can perform the following manipulation. 

$$
\lim _ {| j - i | \rightarrow 0} (\mathbf {B} _ {i \bullet}) = \lim _ {| j - i | \rightarrow 0} \left(- \left| \frac {j - i}{\alpha} \right| ^ {\beta}\right) = \lim _ {| j - i | \rightarrow 0} \left(- \left| \frac {\alpha}{j - i} \right| ^ {| \beta |}\right) = - \infty
$$

When we plug $- \infty$ back into the scoring function we see that it becomes $- \infty$ , and consequently the softmax becomes 0. 

$$
\lim _ {| j - i | \rightarrow 0} p _ {i j} = \lim _ {| j - i | \rightarrow 0} \operatorname{softmax} \left(\mathbf {q} _ {i} \mathbf {K} ^ {\top} + \mathbf {M} _ {i \bullet} + \mathbf {B} _ {i \bullet}\right) = 0
$$

Since the result of the softmax is zero, GGD-BAM does not attend to local context when $\beta < 0$ and $\alpha > 0$ □ 

## B.7 GGD-BAM CAN HAVE ARBITRARILY LONG CONTEXT FOR $\beta < 0 \mathrm { A N D } \alpha > 0$

$$
\begin{array}{l} \text {Theorem 4: GGD - BAM takes into account arbitrarily long context for any \beta <   0 and \alpha > 0 .} \\ \forall \beta <   0, \forall \alpha > 0, \text {If p_{ij} = softmax (\mathbf {q}_ {i} \mathbf {K} ^ {\top} + \mathbf {M} _ {i\bullet} + \mathbf {B} _ {i\bullet}) then \lim _ {|j - i|\to\infty} p_{ij} \neq 0 ,} \end{array}
$$

Proof. This proof is similar to Lemmas 3 and 4 and Theorem 3. We prove this lemma for a fixed query $\mathbf { q } _ { i }$ . The scoring function of GGD-BAM has three components $\mathbf { q } _ { i } \mathbf { \bar { K } } ^ { \top } , \mathbf { M } _ { i \bullet }$ , and $\mathbf { B } _ { i \bullet }$ , lets take the limit of the scoring function and see how it behaves: 

$$
\lim _ {| j - i | \to \infty} \left(\mathbf {q} _ {i} \mathbf {K} ^ {\top} + \mathbf {M} _ {i \bullet} + \mathbf {B} _ {i \bullet}\right) = \mathbf {q} _ {i} \mathbf {K} ^ {\top} + \mathbf {M} _ {i \bullet} + \lim _ {| j - i | \to \infty} (\mathbf {B} _ {i \bullet})
$$

We drop the limit in the causal mask $\mathbf { M } _ { i \bullet }$ as the only effect of increasing the context size and the distance between i and $j$ in the causal mask is making the mask bigger in size, but it stills follows the same formation law with 0 to the left of the query and $- \infty$ elsewhere. 

When the shape parameter $\beta$ is negative, then we can perform the following manipulation. 

$$
\lim _ {| j - i | \to \infty} (\mathbf {B} _ {i \bullet}) = \lim _ {| j - i | \to \infty} \left(- \left| \frac {j - i}{\alpha} \right| ^ {\beta}\right) = \lim _ {| j - i | \to \infty} \left(- \left| \frac {\alpha}{j - i} \right| ^ {| \beta |}\right) = 0
$$

When we plug 0 back into the scoring function we see that it becomes −∞, and consequently the softmax becomes 0. 

$$
\begin{array}{c} \lim _ {| j - i | \to \infty} p _ {i j} = \lim _ {| j - i | \to \infty} \operatorname{softmax} \left(\mathbf {q} _ {i} \mathbf {K} ^ {\top} + \mathbf {M} _ {i \bullet} + \mathbf {B} _ {i \bullet}\right) \\ = \operatorname{softmax} \left(\mathbf {q} _ {i} \mathbf {K} ^ {\top} + \mathbf {M} _ {i \bullet}\right) \end{array}
$$

Since the result of the softmax is not necessarily zero, $p _ { i j }$ also is not necessarily 0, thus GGD-BAM can attend to arbitrarily long context when $\beta < 0$ and $\alpha > 0$ □ 

## C EXPERIMENTAL SETUP

Here we detail the experimental setup necessary to replicate our results. 

Our models were based on Llama3 (Grattafiori et al., 2024). We used RMSNorm (Zhang & Sennrich, 2019) and, on feedforward blocks, we used swiglu activation function (Shazeer, 2020). We trained all our models on the Fineweb 10B dataset (Penedo et al., 2024) utilizing the Mistral-7B v0.3 tokenizer (Jiang et al., 2023) for text processing. Training was performed using context lengths of 512, 1024, and 2048 tokens to evaluate performance under different sequence lengths. 

To maintain document separation within packed sequences, we employed an attention mask preventing self-attention between distinct documents (Grattafiori et al., 2024). The models were optimized using RAdam with decoupled weight decay set to 0.1 and an initial learning rate of $1 \times 1 0 ^ { - 3 }$ . We applied a cosine learning rate decay schedule, reducing the learning rate to a minimum of $0 . 1 \times$ its initial value. 

We trained LMs up to 1.1 billion learnable parameters. Table 1 details the configuration of each trained model, including embedding, trainable parameters, attention heads, and hidden layer size. 


Table 1: Architecture details of the LM we evaluated in our study.


<table><tr><td>Attribute</td><td>120M</td><td>432M</td><td>1.1B</td></tr><tr><td>Parameters embedding</td><td>~25M</td><td>~50M</td><td>~67M</td></tr><tr><td>Parameters transformer</td><td>~95M</td><td>~380M</td><td>~1B</td></tr><tr><td>Parameters BAM (<eq>\theta_{\alpha}, \theta_{\beta}</eq> and <eq>\theta_{\mu}</eq>)</td><td>576</td><td>1008</td><td>1440</td></tr><tr><td>Attention heads</td><td>16</td><td>24</td><td>32</td></tr><tr><td>Layers</td><td>12</td><td>14</td><td>15</td></tr><tr><td>Hidden size</td><td>768</td><td>1536</td><td>2048</td></tr><tr><td>ff hidden size</td><td><eq>2 \times 768</eq></td><td><eq>2 \times 1536</eq></td><td><eq>4 \times 2048</eq></td></tr><tr><td>Learning Rate</td><td><eq>1 \times 10^{-3}</eq></td><td><eq>5 \times 10^{-4}</eq></td><td><eq>3 \times 10^{-4}</eq></td></tr></table>

Training was executed for approximately 19, 251 steps with a global batch size of 589, 824 tokens, leveraging up to 6 NVIDIA A6000 GPUs. For the 1.1B parameter models, we perform additional 256 training steps with a context length of 1024. This was necessary because bigger models tends to overfit in the trained context length (see Appendix H.5 for further details). The resulting models from these specific training configurations formed the basis for our subsequent performance evaluation. 

In our implementation we define three learnable parameters, $\theta _ { \mu } , \theta _ { \alpha }$ and $\theta _ { \beta }$ for each attention head. So $\mathbf { B } = [ \bar { b } _ { i j } ] _ { L \times L }$ where: 

$$
b _ {i j} = - e ^ {\theta_ {\alpha}} \left(\left| (j - i) - \left(e ^ {\theta_ {\mu}} - e ^ {- \theta_ {\mu}}\right) \right| + \epsilon\right) ^ {\theta_ {\beta}},
$$

for $i = 1 , \ldots , L$ and $j = 1 , \dots , L$ . So the GGD is parametrized in the following way: $\mu = e ^ { \theta _ { \mu } } - e ^ { - \theta _ { \mu } }$ $\beta = \theta _ { \beta }$ and $\alpha = e ^ { - \frac { \theta _ { \alpha } } { \theta _ { \beta } } }$ , that ensures $\alpha \ge 0 . \epsilon = 1 0 ^ { - 5 }$ avoids division by zero when $\beta < 0$ 

To evaluate our models in contexts longer than 10,000 tokens we implemented key-value caching (Kim et al., 2023). 

## D PERPLEXITY EVALUATION

## D.1 PERPLEXITY ON SHORT CONTEXT

We evaluate generalization for models with distinct PE methods by computing the perplexity for models trained on a context length of 512 in a hold-out validation set of Fineweb with sentences up to 512 tokens and on Wikipedia with sentences up to 512 tokens. Results are shown in Table 2. 


Table 2: Perplexity of 120M models in Wikipedia and Fineweb data.


<table><tr><td></td><td>Wikipedia</td><td>Fineweb</td></tr><tr><td>Sinusoidal</td><td>18.9310</td><td>22.3573</td></tr><tr><td>ALiBi</td><td>18.5452</td><td>22.1771</td></tr><tr><td>NoPE</td><td>19.8609</td><td>23.7898</td></tr><tr><td>BAM</td><td>18.9281</td><td>22.2640</td></tr><tr><td>RoPE</td><td>18.3599</td><td>22.4428</td></tr><tr><td>Sinusoidal SSMax</td><td>19.1467</td><td>22.3150</td></tr><tr><td>NoPE SSMax</td><td>20.3899</td><td>23.7581</td></tr><tr><td>ALiBi SSMax</td><td>18.4967</td><td>22.2125</td></tr><tr><td>BAM SSMax</td><td>18.6897</td><td>22.1363</td></tr><tr><td>RoPE SSMax</td><td>20.1854</td><td>24.4507</td></tr></table>

We see that in context length seen during training, BAM outperforms almost all the compared baselines. The only exception is ALiBi, which outperforms BAM by less than 0.1 points in perplexity. 

ALiBi outperforming BAM in this case is expected from our probabilistic interpretation of PE. ALiBi is a Laplace distribution only focused on the local context whereas BAM has also the ability to use information in long context. We notice in experiments focused on long context (see Appendices F and H) that this small gap in perplexity to ALiBi translates into huge long-context retrieval gain. 

## D.2 PERPLEXITY ON LONG CONTEXT

We evaluate context length generalization for models with distinct PE methods by computing the perplexity for models that were trained on a context length of 512 in a hold-out validation set of sentences longer than 512. This is a similar evaluation procedure as performed by Press et al. (2022). 

Results shown in Figure 8 are consistent with those reported in the ALiBi paper (Press et al., 2022), confirming that Sinusoidal, RoPE, and NoPE fail to extrapolate to sequence lengths beyond those seen during training. For these models, perplexity increases sharply as the context grows, indicating that the models are unable to maintain coherent predictions over long sequences. In contrast, only ALiBi, BAM, BAM combined with Scalable Softmax (SSMax), and RoPE local maintain a stable perplexity profile under context extrapolation. These models exhibit sub-linear or nearly flat perplexity growth as the input length increases to 32,000 tokens—despite being trained only on context length of 512 tokens. This confirms our theoretical results where priors introduced by BAM provide robust generalization to unseen context lengths, on par with ALiBi’s linear bias-based extrapolation. 

We evaluate all the PE context length extrapolation regarding perplexity on Wikipedia dataset. Figure 9 shows the log-scaled perplexity and we see a similar trend to Fineweb 10B. BAM, RoPE and ALiBi are able to maintain low perplexity across all the evaluated context lengths. We see that SSMax has more impact in lowering BAMs perplexity when compared to other PEs. 

However, this type of perplexity-based evaluation has limitations (Hu et al., 2024). It does not measure whether the model attends to the full sequence or only relies on the most recent tokens. For instance, the RoPE Local variant implemented here applies local attention restricted to a sliding window and still achieves competitive results with ALiBi. This indicates that models can make accurate next-token predictions without integrating information from earlier parts of the sequence. Hence, we claim that perplexity should not be taken as the sole measure of effective context length extrapolation. Indeed, information retrieval evaluation seems to be more suitable to assess extrapolation of trained lengths. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/c04e7020-29b8-4cd8-9c96-522d050b6324/4586df1e521da8442ae5acab070e558b37a89c838baa7529c055893884cb5f9f.jpg)



(a) PE without SSMax


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/c04e7020-29b8-4cd8-9c96-522d050b6324/c8f776c4b577636733601bbd882709fbb676f16cfa7bd7b31df8245dc44f183e.jpg)



(b) PE with SSMax



Figure 8: Log-scaled perplexity computed up to 64× the training context length of 512 tokens. BAM, RoPE Local and ALiBi are able to maintain the lowest perplexity on longer contexts.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/c04e7020-29b8-4cd8-9c96-522d050b6324/7b4958a37b5b163395fa474c284109f03f352795e15a436f430feee5cc5d7b27.jpg)



(a) PE without SSMax


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/c04e7020-29b8-4cd8-9c96-522d050b6324/40975edba9bebf2d07370e68e423913b77cbc706bc7245e66a2d7983a5d9b3a7.jpg)



(b) PE with SSMax



Figure 9: Log-scaled perplexity on Wikipedia dataset. BAM, RoPE Local and ALiBi are able to maintain the lowest perplexity on longer contexts.


## E DOWNSTREAM EVALUATION

While perplexity provides a general notion regarding model capability in language modeling, it does not necessarily correlate with capability on downstream tasks. Here we evaluate the large scale 1B parameter models performance on downstream tasks from MMLU Hendrycks et al. (2020), ARC-easy, and ARC-challenge benchmarks Clark et al. (2018). Table 3 shows that BAM SSMax is superior to RoPE in all the evaluated benchmarks. 


Table 3: GGD-BAM vs RoPE Large-Scale 1B parameter models on MMLU and ARC Benchmarks


<table><tr><td></td><td>MMLU</td><td>ARC-Easy</td><td>ARC-Challenge</td></tr><tr><td>BAM SSMax</td><td>0.3716</td><td>0.5770</td><td>0.4132</td></tr><tr><td>RoPE SSMax</td><td>0.3573</td><td>0.5715</td><td>0.4123</td></tr></table>

## F RULER BENCHMARK

To assess GGD-BAM capability to attend to long-context information, we assessed its performance in the NIAH subset of the Ruler benchmark (Hsieh et al., 2024). We deliberately chose only the NIAH subset because it is designed to isolate the PE’s ability to access information at a specific distance while ensuring the model maintains stable next-token prediction performance. 

Performance on other tasks, such as Question Answering and Variable Tracking, is known to correlate more with model size than with context extrapolation, and thus fall out of the scope of this paper. 

## F.1 GGD-BAM VS BASELINES (120M)

As seen in Table 4, GGD-BAM outperforms all the baselines on the three variations of the NIAH task. Specially in NIAH Single 1, our PE method outperforms all other PE methods by a substantial margin, being the only one able to perform retrieval above 6k tokens. 

The second best performing PE is RoPE, followed by ALiBi. In the Single 1 version of Ruler NIAH, RoPE maintains 0.64 accuracy while ALiBi scores a mere 0.16. 

Regardless of the PE, in the Single 3 version of NIAH, where the Passkey appears in the form of a UUID, only BAM achieves an accuracy above 0.8. 


Table 4: Accuracy of 120M models on the NIAH subset of the Ruler Benchmark.


<table><tr><td>Task</td><td>PE</td><td>1K</td><td>1.5K</td><td>2K</td><td>3K</td><td>4K</td><td>6K</td><td>8K</td><td>10K</td><td>12K</td></tr><tr><td rowspan="6">Single 1</td><td>NoPE SSMax</td><td>0.04</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>RoPE SSMax</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>0.64</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>ALiBi SSMax</td><td>0.96</td><td>0.26</td><td>0.22</td><td>0.18</td><td>0.16</td><td>0.10</td><td>0.02</td><td>0.04</td><td>0.02</td></tr><tr><td>RoPE Local</td><td>0.40</td><td>0.22</td><td>0.24</td><td>0.20</td><td>0.12</td><td>0.08</td><td>0.00</td><td>0.02</td><td>0.00</td></tr><tr><td>Sinusoidal SSMax</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>BAM SSMax (ours)</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>0.98</td><td>0.92</td><td>0.88</td></tr><tr><td rowspan="6">Single 2</td><td>NoPE SSMax</td><td>0.70</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>RoPE SSMax</td><td>0.98</td><td>0.96</td><td>0.94</td><td>0.84</td><td>0.52</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>ALiBi SSMax</td><td>1.00</td><td>0.46</td><td>0.12</td><td>0.10</td><td>0.06</td><td>0.02</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>RoPE Local</td><td>0.84</td><td>0.36</td><td>0.10</td><td>0.14</td><td>0.10</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>Sinusoidal SSMax</td><td>0.70</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>BAM SSMax (ours)</td><td>1.00</td><td>1.00</td><td>1.00</td><td>0.88</td><td>0.24</td><td>0.06</td><td>0.02</td><td>0.00</td><td>0.00</td></tr><tr><td rowspan="6">Single 3</td><td>NoPE SSMax</td><td>0.04</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>RoPE SSMax</td><td>0.28</td><td>0.30</td><td>0.10</td><td>0.06</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>ALiBi SSMax</td><td>0.24</td><td>0.04</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>RoPE Local</td><td>0.10</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>Sinusoidal SSMax</td><td>0.08</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>BAM SSMax (ours)</td><td>0.84</td><td>0.68</td><td>0.42</td><td>0.08</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr></table>

## F.2 LARGE SCALE GGD-BAM VS ROPE

Here we perform a large-scale experiment directly comparing RoPE and BAM with 1B parameters in the Ruler benchmark. Results are show in Table 5. BAM achieves superior performance in comparison to RoPE in every evaluated task in the Ruler benchmark, with a highlight of achieving almost perfect accuracy across tasks on the Single 1 subset. 

Other tasks such as Multikey 2 and Multikey 3 are harder for both models. This shows that only our pre-training may not be enough for models to perform such tasks. However, since our goal here is to assess how distinct PE behave on exactly the same training regime, we see that GGD-BAM clearly achieves better longer context in Single 1, 2 and 3, MultiKey 1, MultiQuery and MultiValue. 

## G LONGBENCHV2

In Table 6, we present the results of 1B parameter models on the complete LongBenchV2 benchmark Bai et al. (2025) limited in 131k tokens. We chose to report just the large scale 1B parameter models because smaller models perform close to random guessing in these tasks. BAM outperforms 


Table 5: GGD-BAM vs RoPE Ruler Benchmark Large-Scale 1B parameter models.


<table><tr><td>Task</td><td>PE</td><td>1024</td><td>1536</td><td>2048</td><td>3072</td><td>4096</td><td>6144</td><td>8192</td><td>12288</td><td>16384</td><td>24576</td><td>32768</td></tr><tr><td rowspan="2">Single 1</td><td>BAM SSMax</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td></tr><tr><td>RoPE SSMax</td><td>1.00</td><td>0.88</td><td>0.68</td><td>0.40</td><td>0.30</td><td>0.10</td><td>0.02</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td rowspan="2">Single 2</td><td>BAM SSMax</td><td>1.00</td><td>1.00</td><td>1.00</td><td>0.98</td><td>1.00</td><td>0.88</td><td>0.82</td><td>0.46</td><td>0.18</td><td>0.06</td><td>0.02</td></tr><tr><td>RoPE SSMax</td><td>1.00</td><td>0.82</td><td>0.62</td><td>0.32</td><td>0.18</td><td>0.04</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td rowspan="2">Single 3</td><td>BAM SSMax</td><td>0.88</td><td>0.92</td><td>0.88</td><td>0.86</td><td>0.80</td><td>0.62</td><td>0.30</td><td>0.10</td><td>0.02</td><td>0.00</td><td>0.00</td></tr><tr><td>RoPE SSMax</td><td>0.76</td><td>0.30</td><td>0.16</td><td>0.04</td><td>0.02</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td rowspan="2">MultiKey 1</td><td>BAM SSMax</td><td>0.84</td><td>0.86</td><td>0.94</td><td>0.92</td><td>0.86</td><td>0.76</td><td>0.66</td><td>0.56</td><td>0.24</td><td>0.10</td><td>0.06</td></tr><tr><td>RoPE SSMax</td><td>0.80</td><td>0.86</td><td>0.68</td><td>0.38</td><td>0.32</td><td>0.08</td><td>0.06</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td rowspan="2">MultiKey 2</td><td>BAM SSMax</td><td>0.22</td><td>0.16</td><td>0.12</td><td>0.04</td><td>0.04</td><td>0.00</td><td>0.02</td><td>0.02</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>RoPE SSMax</td><td>0.26</td><td>0.14</td><td>0.04</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td rowspan="2">MultiKey 3</td><td>BAM SSMax</td><td>0.12</td><td>0.04</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>RoPE SSMax</td><td>0.18</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td rowspan="2">MultiQuery</td><td>BAM SSMax</td><td>0.88</td><td>0.88</td><td>0.85</td><td>0.82</td><td>0.71</td><td>0.68</td><td>0.62</td><td>0.44</td><td>0.23</td><td>0.10</td><td>0.02</td></tr><tr><td>RoPE SSMax</td><td>0.85</td><td>0.76</td><td>0.42</td><td>0.17</td><td>0.11</td><td>0.06</td><td>0.03</td><td>0.03</td><td>0.01</td><td>0.01</td><td>0.01</td></tr><tr><td rowspan="2">MultiValue</td><td>BAM SSMax</td><td>0.96</td><td>0.94</td><td>0.92</td><td>0.92</td><td>0.82</td><td>0.84</td><td>0.76</td><td>0.70</td><td>0.38</td><td>0.14</td><td>0.00</td></tr><tr><td>RoPE SSMax</td><td>0.96</td><td>1.00</td><td>1.00</td><td>0.78</td><td>0.70</td><td>0.42</td><td>0.32</td><td>0.10</td><td>0.04</td><td>0.08</td><td>0.00</td></tr></table>

RoPE in all evaluated tasks, with an overall score 5 points above RoPE. We opted not to show the Long Structured Data Understanding task because it has only four instances under 131k tokens. 


Table 6: LongBenchv2 Benchmark: GGD-BAM vs RoPE, 1B parameter models.


<table><tr><td></td><td>BAM SSMax</td><td>RoPE SSMax</td></tr><tr><td>Code Repository Understanding</td><td>41.7</td><td>25.0</td></tr><tr><td>Long In-context Learning</td><td>36.4</td><td>30.3</td></tr><tr><td>Long-dialogue History Understanding</td><td>35.0</td><td>35.0</td></tr><tr><td>Multi-Document QA</td><td>26.5</td><td>25.3</td></tr><tr><td>Single-Document QA</td><td>26.5</td><td>18.8</td></tr><tr><td>Overall</td><td>28.6</td><td>24.2</td></tr></table>

## H ABLATION STUDY

## H.1 INITIALIZATION

In this section we study two different initialization strategies for the shape $\theta _ { \beta } ,$ , scale $\theta _ { \alpha }$ and location $\theta _ { \mu }$ parameters of GGD-BAM. The first initialization is a Laplacian that replicates ALiBi, setting $\theta _ { \beta } = 1$ different $\theta _ { \alpha }$ for each layer and $\theta _ { \mu } = 0$ . The second initialization start from Uniform distribution prior $\theta _ { \beta } = 0$ , which is a middle ground between ALiBis Laplacian and $\theta _ { \beta } < 0$ , with $\theta _ { \alpha } = 0$ 

Figure 10 shows us that ALiBi initialization provides least extrapolation lengths. Our best results were achieved by using the initialization scheme of $\theta _ { \beta } = 0$ and $\theta _ { \alpha } = 0$ , this initialization is equivalent to assigning a Uniform prior to all tokens in the context. 

## H.2 LEARNABLE PARAMETERS

The GGD prior in BAM is parameterized by a shape parameter $\theta _ { \beta }$ , a scale parameter $\theta _ { \alpha }$ , and a location parameter $\theta _ { \mu } .$ . In this section, we evaluate how training different subsets of these parameters affects the performance of GGD-BAM. Each configuration introduces a different number of additional trainable parameters: training only $\theta _ { \beta }$ adds 192 parameters; training both $\theta _ { \beta }$ and $\theta _ { \alpha }$ adds 384; and training all three parameters $( \theta _ { \beta } , \theta _ { \alpha }$ , and $\theta _ { \mu } )$ adds 576 parameters to the model. 

In Figure 11 we can see that allowing all parameters $\theta _ { \beta } , \theta _ { \alpha }$ and $\theta _ { \mu }$ to be learned during training lowers model capacity to extrapolate. The best result was achieved when both $\theta _ { \beta }$ and $\theta _ { \alpha }$ are learn during training, showing that both parameters are important for context length extrapolation. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/c04e7020-29b8-4cd8-9c96-522d050b6324/cec6d7bdf79011d41b13024320b8b25f75d7d5f05e5f40cb72bc98e258bb3b3b.jpg)



Figure 10: Comparison of passkey retrieval accuracy of models trained on context length 512 training just $\theta _ { \beta }$ and $\theta _ { \alpha }$ with three distinct parameter initialization schemes.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/c04e7020-29b8-4cd8-9c96-522d050b6324/eadc56a846d478a94f53cf255accfbc84b406da5bce0e4c16323e20a47e02430.jpg)



Figure 11: Passkey retrieval accuracy of models training just: $\theta _ { \beta } ; \theta _ { \beta }$ and $\theta _ { \alpha } ;$ and $\theta _ { \beta } , \theta _ { \alpha }$ and $\theta _ { \mu } .$ . All variations were trained on context length 512. Training $\theta _ { \beta }$ and $\theta _ { \alpha }$ while fixing ${ \theta } _ { \beta } = 0$ yields more extrapolation length.


If we compare results on Figure 11 to other PE in Figure 3 we see that even our worst combination of training all parameters is superior to all other PE in long context passkey retrieval. 

## H.3 TRAINING CONTEXT LENGTH

Here we repeat the experiments for models trained on context length of 1,024 and 2,048, double and quadruple the original context length of 512. We show detailed results for training context length of 1,024 both on perplexity and passkey retrieval when compared to other PE methods. And we also compare passkey retrieval accuracy between BAM SSMax on those three distinct context lengths. 

Essentially, the trend of ALiBi, RoPE Local and BAM SSMax being the only PE methods that are able to maintain low perplexity on longer context is maintained, this is possible to identify in Figure 12. As expected, NoPE and Sinusoidal PE are the first models to exponentially increase in perplexity. 

RoPE SSMax improved it extrapolation performance and was able to maintain perplexity on par with ALiBi until around 7,500 tokens. It is worth noting that to achieve such results with RoPE, we expanded the $\mathbf { R } _ { \theta }$ manipulation post-training that was performed by Nakanishi (2025) from 50× to 100×. Without this RoPE SSMax would perform on par with standard RoPE. 

Regarding passkey retrieval accuracy, again all PE methods except BAM SSMax struggles to access long context information and maintain high accuracy, Figure 13 shows this trend. Although the Figure 13 shows context lengths up to 32,000, we evaluated BAM SSMax until 512,000 context length and it maintained accuracy above 80% until 300,000 and did not drop to zero throughout the evaluation. Beyond 512,000 context length, we did not have enough vram to perform evaluation. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/c04e7020-29b8-4cd8-9c96-522d050b6324/04be2db046f12f62b27c4dbd462843e9199e15a25fb4793c0a6f9091388c9d2f.jpg)



(a) Perplexity on Wikipedia


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/c04e7020-29b8-4cd8-9c96-522d050b6324/8593c8f2a2be28afa1056eaf646aa716da2503010d9117b4777ab6695ce206f8.jpg)



(b) Perplexity on Fineweb 10B



Figure 12: Log-scaled perplexity computed up to 32× the training context length of 1024 tokens. BAM SSMax, RoPE Local and ALiBi are able to maintain the lowest perplexity on longer contexts.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/c04e7020-29b8-4cd8-9c96-522d050b6324/9bf7c4e239887dfe23648370abfe3abaa60216135bef27d9deaac9fb58e10d3b.jpg)



Figure 13: Passkey retrieval accuracy of models with distinct PE in the Passkey Retrieval task. BAM and BAM SSMax outperform all PE methods. Specifically BAM SSMax is capable of maintaining perfect accuracy for a context 32× the training context length.


In Figure 14 we can see how training in longer context lengths affects GGD-BAM. Generally, training for longer contexts appears to make the model more robust to long context generalization as the accuracy tend to drop slower. The model trained with context length of 2, 048 tokens generalizes all the way to 512k tokens while maintaining accuracy above 90%. The model trained with context length of 512 achieves 40% accuracy on 512k, showing a correlation between trained context length and retrieval accuracy at 512k tokens. Nevertheless, our model trained on context length of 512 is competitive with all the others until 300, 000 where others achieve higher accuracy. 

## H.4 SCALABLE SOFTMAX

Here we test all PE methods without SSMax. Figure 15 shows the same trend of models with SSMax, BAM outperforms every other PE method. It is worth noting that SSMax improves context length generalization in almost all the assessed PE methods. This shows that fading attention is indeed one of the problems in long context extrapolation, however BAM is superior to other PE on both cases. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/c04e7020-29b8-4cd8-9c96-522d050b6324/f709bdb96bd992146e34aa605e05078f2874010d360777e4ccd1f1bffb72f0dc.jpg)



Figure 14: Comparison of passkey retrieval accuracy of GGD-BAM models trained on context length 512, 1,024 and 2,048.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/c04e7020-29b8-4cd8-9c96-522d050b6324/18dcc4b5381408835218685fab1c9d8651ec4d44e52bf7bf3d651839b9166a47.jpg)



Figure 15: Passkey retrieval accuracy with distinct PE. BAM outperform all PE methods.


By comparing the results from BAM and BAM SSMax, we note that SSMax plays a role in maintaining the PE ability to extrapolate to longer lengths. This trend is also maintained when comparing RoPE and its SSMax version. The fact that Scalable Softmax improves context length generalization both for BAM and for RoPE shows that good PE is necessary but not sufficient for context length extrapolation. The softmax function tends to zero for longer context windows, which is a problem both for language modeling and for retrieval. To counterbalance this effect, scalable softmax applies a rescaling factor to the logits (Nakanishi, 2025), fixing that limitation. 

Scalable Softmax introduces a rescaling factor $s \times \ln ( n )$ , where n is the size of the input vector and s is a learnable parameter. Note that s has a similar effect to the normalizing scaler Z obtained when framing PE as a Bayesian mechanism. The only effective difference is that Z should be a function of both query and keys whereas s is a learnable parameter. 

To understand the effect of the normalizing scalar s in our models across distinct scales, we show in Figure 16 how this learnable parameter is distributed after training. We see that although the distribution appears to have a heavier tail in smaller models, the shape of the distribution is similar across model scales. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/c04e7020-29b8-4cd8-9c96-522d050b6324/cb5386d073ba9965f6108028ce6d71046b1663f8ec83733948283d35df1aad4f.jpg)



Figure 16: Scaling factor s in scalable softmax in three distinct model scales.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/c04e7020-29b8-4cd8-9c96-522d050b6324/8c08a1f29465ff52f792f32862a50ac01045790465c22986de8a18d4252ecdbe.jpg)



Figure 17: Loss of GGD-BAM before and after lightweight fine-tuning for context extension.


## H.5 CONTEXT EXTENSION

We observed during our first experiments that smaller models generalized better to longer context. We found out that bigger models are prone to overfitting the trained context length. 

We devised a lightweight fine-tuning with context length 1024 for 256 steps (in comparison to 512 during the beginning of the training) to make bigger models generalize once again. In Figure 17, we show that after lightweight fine-tuning the loss of our models become more stable for context beyond the training length of 512 tokens. This shows that, even if bigger models can overfit the training context length, a lightweight fine-tuning procedure can make them generalize to extended contexts. 

After context extension, we performed the PassKey Retrieval analysis and noticed that bigger models benefit more from the context extension than smaller ones. The results obtained by this lightweight fine-tuning procedure are shown in Table 7. 

In Table 7 we see that the 120M model generalizes up to 512× the trained context length with accuracy above 0.8. However, bigger models struggle in much shorter sequences. When analyzing each model to their context-extended counterparts, we see that all model-scales benefit from this procedure. The 1.1B parameter model, however, has most improvement of context extension. 


Table 7: Context extension effect across different model sizes of BAM SSMax on PassKey Retrieval.


<table><tr><td>Model</td><td>&lt;1K</td><td>51K</td><td>102K</td><td>153K</td><td>204K</td><td>256K</td><td>307K</td><td>358K</td><td>409K</td><td>460K</td><td>512K</td></tr><tr><td>120M</td><td>1.0</td><td>1.0</td><td>0.9</td><td>0.9</td><td>0.9</td><td>0.8</td><td>0.7</td><td>0.7</td><td>0.8</td><td>0.6</td><td>0.4</td></tr><tr><td>120M Ext.</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td>431M</td><td>1.0</td><td>0.5</td><td>0.1</td><td>0.3</td><td>0.1</td><td>0.1</td><td>0.1</td><td>0.2</td><td>0.1</td><td>0.0</td><td>0.2</td></tr><tr><td>431M Ext.</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td>1.1B</td><td>0.9</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.1</td><td>0.0</td><td>0.0</td><td>0.1</td><td>0.0</td><td>0.0</td></tr><tr><td>1.1B Ext.</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>0.9</td><td>1.0</td><td>0.8</td><td>0.9</td><td>0.7</td><td>0.7</td></tr></table>

## H.6 MODEL SIZE

Here we compare how BAM SSMax compares to RoPE SSMax on a large scale model size (1.1B) on the Passkey Retrieval setting. Models were trained with context length 512 and prompted to perform Passkey Retrieval with up to 512, 000 tokens. Figure 18 shows that BAM SSMax also dominates RoPE SSMax across all evaluated lengths, performing accurate Passkey retrieval in all contexts that we were able to assess using our available compute. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/c04e7020-29b8-4cd8-9c96-522d050b6324/674d15103afd471a1dd94e99255b9311701a331e6e66c32cdc7c2f70159a3255.jpg)



(a) BAM SSMax


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/c04e7020-29b8-4cd8-9c96-522d050b6324/78d8a1ec0a4067474bfe6d21a38a3b482e1b03364eed13ae1c36181e66838d29.jpg)



(b) RoPE SSMax



Figure 18: Passkey Retrieval accuracy on 1.1B models. In the bottom row and on the last column, we see average accuracy across length and position, respectively.


This ablation corroborates that BAM is capable of using information across longer contexts than RoPE, and that such a conclusion generalizes across bigger models (and not just in 120M settings). 

## H.7 PE IMPACT ON INFERENCE PERFORMANCE

Here we access how distinct PE strategies impact model throughput during inference. To perform this experiment, we initialize BAM and all the baselines trainable weights of four distinct model sizes and run 100 samples with batch size 1 and sequence length 512. 

In Table 8, we see that for smaller models the cost of performing Scalable Softmax dominates the results. This is imperceptible in bigger models, since the only impact of Scalable Softmax appears to be in the standard deviation. BAM does not affect model inference time in comparison to other PEs. When we account for the standard deviation, every model has equivalent inference time. 


Table 8: Inference time (ms) and vram (GB) during the backward pass for distinct model sizes.


<table><tr><td rowspan="2"></td><td colspan="2">120M</td><td colspan="2">430M</td><td colspan="2">1.1B</td></tr><tr><td>Time</td><td>VRAM</td><td>Time</td><td>VRAM</td><td>Time</td><td>VRAM</td></tr><tr><td>Sinusoidal</td><td><eq>32.84 \pm 0.40</eq></td><td>1.734</td><td><eq>50.98 \pm 0.37</eq></td><td>4.933</td><td><eq>117.26 \pm 0.42</eq></td><td>12.329</td></tr><tr><td>Sinusoidal SSMax</td><td><eq>32.82 \pm 1.13</eq></td><td>1.736</td><td><eq>53.39 \pm 0.28</eq></td><td>4.933</td><td><eq>120.91 \pm 0.55</eq></td><td>12.324</td></tr><tr><td>RoPE</td><td><eq>35.77 \pm 1.98</eq></td><td>1.754</td><td><eq>53.42 \pm 0.38</eq></td><td>4.936</td><td><eq>120.50 \pm 0.36</eq></td><td>12.330</td></tr><tr><td>RoPE SSMax</td><td><eq>36.04 \pm 1.39</eq></td><td>1.742</td><td><eq>55.65 \pm 0.17</eq></td><td>4.934</td><td><eq>123.34 \pm 0.42</eq></td><td>12.325</td></tr><tr><td>ALiBi</td><td><eq>31.43 \pm 0.73</eq></td><td>1.738</td><td><eq>52.24 \pm 0.27</eq></td><td>4.933</td><td><eq>119.34 \pm 0.39</eq></td><td>12.329</td></tr><tr><td>ALiBi SSMax</td><td><eq>32.21 \pm 0.09</eq></td><td>1.736</td><td><eq>54.22 \pm 0.21</eq></td><td>4.934</td><td><eq>122.09 \pm 0.33</eq></td><td>12.325</td></tr><tr><td>BAM</td><td><eq>33.13 \pm 0.53</eq></td><td>1.739</td><td><eq>52.85 \pm 0.24</eq></td><td>4.936</td><td><eq>120.31 \pm 0.38</eq></td><td>12.329</td></tr><tr><td>BAM SSMax</td><td><eq>33.01 \pm 0.81</eq></td><td>1.737</td><td><eq>59.50 \pm 0.18</eq></td><td>4.936</td><td><eq>133.14 \pm 0.41</eq></td><td>12.319</td></tr><tr><td>NoPE</td><td><eq>38.15 \pm 1.87</eq></td><td>1.738</td><td><eq>56.10 \pm 0.42</eq></td><td>4.933</td><td><eq>121.34 \pm 0.80</eq></td><td>12.329</td></tr><tr><td>NoPE SSMax</td><td><eq>38.49 \pm 1.30</eq></td><td>1.736</td><td><eq>58.36 \pm 0.70</eq></td><td>4.933</td><td><eq>124.95 \pm 0.60</eq></td><td>12.324</td></tr></table>

## I LIMITATIONS

Despite the theoretical and empirical strengths of BAM and its instantiation with GGD, our study is subject to several limitations, which we acknowledge and discuss below. 

Model Scale and Generalization to Larger LMs. Our experiments were conducted on Transformer models with up to 1.1 billion parameters due to limited compute availability. While our results show improvements in context length extrapolation at this scale, it remains an open question whether these gains persist or even amplify in very large language models. However, we note that this evaluation regime is consistent with prior work in the PE literature, including those introducing ALiBi (Press et al., 2022) and NoPE (Kazemnejad et al., 2023), which also validated their approaches using similar-scale models. 

Dataset Scope and Representativeness. Our empirical evaluation of perplexity is currently restricted to two datasets: FineWeb 10B (Penedo et al., 2024), and Wikipedia (Foundation, 2023). While these datasets provide coverage of both large-scale pretraining and structured text, this coverage is not exhaustive. Broader evaluations across additional domains—such as code, long-form scientific documents—would be valuable for assessing the robustness and generality of BAM-based priors. 

Coverage of Positional Encoding Methods. Due to computational constraints and the complexity of reimplementing certain positional encoding strategies, our experiments do not encompass all meth ods proposed in the literature. We did not evaluate the T5 relative position bias approach (Raffel et al., 2020), which requires bucketing mechanisms and distinct architectural modifications. Nonetheless, we believe that the set of baselines considered—covering absolute positional encodings (Sinusoidal), rotary encodings (RoPE), relative linear biases (ALiBi), and content-only baselines (NoPE)—provides a representative and diverse comparison to assess the context extrapolation capabilities of BAM. 

Generalization to Instruction/Preference Tuned LMs. We did not evaluate BAM in the context of instruction-tuned or preference-tuned models. LMs often undergo additional fine-tuning stage, such as supervised instruction following, reinforcement learning from human feedback, or direct preference optimization, which can significantly alter attention dynamics and generalization behavior. It remains an open question whether the context extrapolation benefits introduced by BAM are preserved, attenuated, or potentially enhanced in such settings. Assessing the how models with BAM as PE perform after instruction-tuned architectures is an important direction for future work. 

## J BROADER IMPACTS

Improving context length extrapolation in Transformers has the potential to reduce the computational and environmental costs associated with pretraining large language models. Because attention scales quadratically with sequence length, training with long contexts is prohibitively expensive. GGD-BAM enables models to generalize to longer sequences without requiring direct exposure during training, potentially lowering the need for long-context pre-training. 

This efficiency gain could contribute to more sustainable and accessible language model development, particularly for institutions with limited compute resources. Furthermore, better long-range generalization supports important applications such as legal and medical document processing, educational content understanding, and scientific analysis. However, these capabilities must be accompanied by careful evaluation to ensure reliability and safety in high-stakes domains. 

$$
Z
$$

## K.1 STATISTICAL-PHYSICS INTERPRETATION OF $Z$

Let $f _ { j } = f _ { \mathrm { c o n t } } ( \mathbf { q } _ { i } , \mathbf { k } _ { j } ) , \quad g _ { j } = g _ { \mathrm { p o s } } ( i , j ) .$ 

We introduce three normalization constants: 

$$
Z _ {\text {cont}} = \sum_ {j} e ^ {f _ {j}}, \quad Z _ {\text {pos}} = \sum_ {j} e ^ {g _ {j}}, \quad Z _ {\text {joint}} = \sum_ {j} e ^ {f _ {j} + g _ {j}}.
$$

The usual Gibbs–Boltzmann partitionfunction is 

$$
Z _ {\text { joint }} = \sum_ {j} \exp (f _ {j} + g _ {j}),
$$

and the corresponding free energy is $F _ { \mathrm { j o i n t } } = - \ln Z _ { \mathrm { j o i n t } }$ 

The factorization normalizer $Z$ that restores $\begin{array} { r } { \sum _ { j } \mathrm { s o f t m a x } ( f _ { j } + g _ { j } ) = 1 } \end{array}$ after factorizing $\exp ( f _ { j } +$ $g _ { j } ) = \exp ( f _ { j } ) \exp ( g _ { j } )$ is 

$$
Z = \frac {Z _ {\text { cont }} Z _ {\text { pos }}}{Z _ {\text { joint }}} = \sum_ {j} \left[ \operatorname{softmax} (f _ {j}) \times \operatorname{softmax} (g _ {j}) \right].
$$

Its log, ln $Z = \ln Z _ { \mathrm { c o n t } } + \ln Z _ { \mathrm { p o s } } - \ln Z _ { \mathrm { j o i n t } }$ , is precisely the interaction free-energy between the content and position potentials. Finally, introducing an inverse temperature $\gamma \ : \mathrm { y }$ ields 

$$
p _ {i j} \propto \exp \bigl (\gamma (f _ {j} + g _ {j}) \bigr),
$$

so γ could control how sharply the attention distribution peaks. 

## K.2 GEOMETRIC INTERPRETATION OF $Z$

Recall 

$$
p _ {\text { cont }} (j) = \operatorname{softmax} \bigl (f _ {\text { cont }} (q _ {i}, k _ {j}) \bigr), \quad p (g _ {\text { pos }} (i, j)) = \operatorname{softmax} \bigl (g _ {\text { pos }} (i, j) \bigr),
$$

and let 

$$
p _ {\text { cont }} = \left(p (f _ {\text { cont }} (\mathbf {q} _ {i}, \mathbf {k} _ {j}))\right) _ {j = 1} ^ {n}, \quad p _ {\text { pos }} = \left(p (g _ {\text { pos }} (i, j))\right) _ {j = 1} ^ {n}.
$$

Both vectors lie in the probability simplex $\begin{array} { r } { \Delta ^ { n - 1 } = \left\{ x \in \mathbb { R } ^ { n } : x _ { j } \geq 0 , \sum _ { i } x _ { j } = 1 \right\} } \end{array}$ . Then 

$$
Z = \sum_ {j = 1} ^ {n} p (f _ {\text { cont }} (\mathbf {q} _ {i}, \mathbf {k} _ {j})), p (g _ {\text { pos }} (i, j)) = \langle p _ {\text { cont }}, p _ {\text { pos }} \rangle ,
$$

where $\begin{array} { r } { \langle p _ { \mathrm { c o n t } } , p _ { \mathrm { p o s } } \rangle = \sum _ { j } p ( f _ { \mathrm { c o n t } } ( \mathbf { q } _ { i } , \mathbf { k } _ { j } ) ) , p ( g _ { \mathrm { p o s } } ( i , j ) ) . } \end{array}$ 

• Dot-product as overlap. $\langle p _ { \mathrm { c o n t } } , p _ { \mathrm { p o s } } \rangle \in [ 0 , 1 ]$ measures how much the two distributions “agree”—it is maximal when they coincide and minimal when they are disjoint. 

• Norms of probability vectors. Since $\| p \| _ { 2 } \le \| p \| _ { 1 } = 1$ for any $p \in \Delta ^ { n - 1 }$ , the raw dot-product is not a true cosine similarity unless one divides by $\| p _ { \mathrm { c o n t } } \| _ { 2 } \| p _ { \mathrm { p o s } } \| _ { 2 }$ . We omit that division because we need $\begin{array} { r l } { \sum _ { j } p ( \bar { f } _ { \mathrm { c o n t } } ( \mathbf { q } _ { i } , \mathbf { k } _ { j } ) ) , p ( g _ { \mathrm { p o s } } ( i , \dot { j } ) ) } \end{array}$ exactly to quantify the normalization gap of the product of two softmaxes. 

• Re-normalization identity. The product distribution $p _ { \mathrm { c o n t } } \odot p _ { \mathrm { p o s } }$ sums to $\langle p _ { \mathrm { c o n t } } , p _ { \mathrm { p o s } } \rangle \neq 1$ Inverting that sum, 

$$
Z = \sum_ {j} p (f _ {\text { cont }} (\mathbf {q} _ {i}, \mathbf {k} _ {j})), p (g _ {\text { pos }} (i, j)),
$$

precisely restores $\begin{array} { r } { \sum _ { j } \big [ p ( f _ { \mathrm { c o n t } } ( \mathbf { q } _ { i } , \mathbf { k } _ { j } ) ) , p ( g _ { \mathrm { p o s } } ( i , j ) ) \big ] Z = 1 . } \end{array}$ 

• Cosine-similarity caveat. If one instead defined cos $\begin{array} { r } { \theta = \frac { \left. { p _ { \mathrm { c o n t } } , p _ { \mathrm { p o s } } } \right. } { \| p _ { \mathrm { c o n t } } \| _ { 2 } \| p _ { \mathrm { p o s } } \| _ { 2 } } } \end{array}$ , that extra normal ization would destroy the simple re-normalization identity needed for attention. 

Thus $Z \ = \ 1 / \langle p _ { \mathrm { c o n t } } , p _ { \mathrm { p o s } } \rangle$ has a clear geometric meaning: it compensates for the overlap (or misalignment) between the content-based and position-based probability vectors, ensuring their elementwise product yields a valid distribution. 

## K.3 INFORMATION-THEORETIC INTERPRETATION OF Z

Let 

$$
p _ {\text { joint }} (j) = p _ {i j} = \operatorname{softmax} \left(f _ {\text { cont }} \left(q _ {i}, k _ {j}\right) + g _ {\text { pos }} (i, j)\right),
$$

and recall the marginals 

$$
p _ {\text { cont }} (j) = \operatorname{softmax} \bigl (f _ {\text { cont }} (q _ {i}, k _ {j}) \bigr), \quad p _ {\text { pos }} (j) = \operatorname{softmax} \bigl (g _ {\text { pos }} (i, j) \bigr).
$$

The mutual information between content and position under the joint distribution is 

$$
I (\text { cont;pos }) = \sum_ {j} p _ {\text { joint }} (j) \ln \frac {p _ {\text { joint }} (j)}{p (f _ {\text { cont }} (\mathbf {q} _ {i} , \mathbf {k} _ {j})) , p _ {\text { pos }} (j)}.
$$

Since 

$$
p _ {\text { joint }} (j) = \frac {e ^ {f _ {j} + g _ {j}}}{Z _ {\text { joint }}} \quad \text { and } \quad p (f _ {\text { cont }} (\mathbf {q} _ {i}, \mathbf {k} _ {j})), p _ {\text { pos }} (j) = \frac {e ^ {f _ {j}}}{Z _ {\text { cont }}} \frac {e ^ {g _ {j}}}{Z _ {\text { pos }}},
$$

one shows directly that 

$$
I (\mathrm{cont;pos}) = \ln {\frac {Z _ {\mathrm{cont}} Z _ {\mathrm{pos}}}{Z _ {\mathrm{joint}}}} = \ln Z.
$$

Hence 

$$
Z = \exp \bigl (I (\text { cont; pos }) \bigr),
$$

which implies: 

$Z = 1$ if and only if content and position are statistically independent $( I = 0 )$ 

• Z grows exponentially with the amount of shared information between content and position. 

Thus $Z$ can be seen as the “information-coupling” multiplier that re–normalizes the product of the two marginals into the true joint attention distribution. 