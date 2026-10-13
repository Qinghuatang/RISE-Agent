# CAUTIOUS OPTIMIZERS: IMPROVING TRAINING WITH ONE LINE OF CODE

Kaizhao Liang<sup>∗,1,†</sup>, Lizhang Chen<sup>∗,1,†</sup>, Bo Liu<sup>1</sup>, Qiang Liu<sup>1</sup> <sup>1</sup>The University of Texas at Austin 

## ABSTRACT

AdamW has been the default optimizer for transformer pretraining. For many years, our community searched for faster and more stable optimizers with only constrained positive outcomes. In this work, we propose a one-line modification in Pytorch to any momentum-based optimizer, which we rename cautious optimizer, e.g. C AdamW and C-Lion. Our theoretical result shows that this modification preserves Adam’s Hamiltonian function and it does not break the convergence guarantee under the Lyapunov analysis. In addition, a whole new family of optimizers is revealed by our theoretical insight. Among them, we pick the simplest one for empirical experiments, showing not only consistent speed-up on LLM pretraining, but also image classification, with minimum extra tuning on hyperparameters. 

## 1 INTRODUCTION

Algorithm 1 Caution an Optimizer (OPT) in PyTorch 

# param p, update u from OP T, grad g m = (u <sub>*</sub> g > 0).to(g.dtype) p.add (u <sub>*</sub> m/(m.mean()+eps), alpha=-lr) 

Optimization is an important and constantly evolving field in modern machine learning. Undoubtedly, Adam (Kingma, 2014) and AdamW (Loshchilov, 2017) are the most consequential optimizers proposed almost a decade ago. Since then, many efforts (Zhang et al., 2021; Loshchilov et al., 2017) have been made to discover better and faster optimizers beyond these two. However, until now, AdamW remains the dominant workhorse for applications, from pre-training Large Language Models (LLMs) (Touvron et al., 2023) to fine-tuning text to image diffusion (Rombach et al., 2022), with no real challenges to their ruling status. 

In the dawn of the era of LLMs, the arms race of model scaling intensifies (Achiam et al., 2023). A faster optimizer means more training tokens can be consumed within the same amount of time. Ultimately, this leads to more capable models (Kaplan et al., 2020). Hence, the interest in searching for an optimizer beyond AdamW is re-kindled. Recent progress in new AdamW alternatives such as Lion (Chen et al., 2024; 2023a), SHAMPOO (Gupta et al., 2018), SOAP (Vyas et al., 2024), ADOPT (Taniguchi et al., 2024), and Schedule-Free (Defazio et al., 2024), all claim substantial improvement over AdamW. 

However, these methods normally require non-trivial efforts to obtain optimal results, especially hyperparameter tuning, which greatly limits their potential and wide adoption. In light of this dilemma, we propose cautious optimizers, an exceptionally simple performance booster of any momentum-based optimizer that only requires one line of modification (see Algorithm 1). The change is simple: do not update unless the proposed update direction and the current gradients are aligned. With this minor change, we obtain consistent improvement over the base optimizer without modification of the original optimal hyperparameters. 

To provide an overview of the idea, let us consider a general optimizer for minimizing the loss L(w): 

$$
\boldsymbol {w} _ {t + 1} \leftarrow \boldsymbol {w} _ {t} - \epsilon_ {t} \boldsymbol {u} _ {t},
$$

where $\mathbf { \Delta } \mathbf { u } _ { t }$ is the negative update direction of the parameter ${ \pmb w } _ { t }$ at iteration t, and $\epsilon _ { t } > 0$ is the step size. We will assume that the update represents a generic momentum-based optimizer, including, for example, Polyak and Nesterov momentum, Adam, and Lion. In all these momentum-based optimizers, $\mathbf { \pmb { u } } _ { t }$ does not necessarily align with the gradient direction $\mathbf y _ { t } = \nabla \mathcal L ( \boldsymbol w _ { t } )$ , which can result in a temporary increase in the loss function and slow down convergence. 

Cautious optimizers avoid this issue by adding a simple mask function based on the sign consistency of $\mathbf { \pmb { u } } _ { t }$ and $\mathbf { \nabla } _ { \mathbf { \boldsymbol { g } } _ { t } }$ : 

$$
\pmb {w} _ {t + 1} \leftarrow \pmb {w} _ {t} - \epsilon_ {t} \pmb {u} _ {t} \circ \phi (\pmb {u} _ {t} \circ \pmb {g} _ {t}),
$$

where $\circ$ denotes an element-wise product, and ϕ is a map that reweights the update based on the product ${ \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf } { } { \mathbf } { } \mathbf { } { \mathbf } { } \mathbf { } \mathbf { } { } \mathbf { } \mathbf { }  { \mathbf { } \mathbf { } } { \mathbf } { \mathbf { } \mathbf { } } { \mathbf } { \mathbf { } \mathbf { } } { \mathbf } { \mathbf { } \mathbf { } \mathbf { } } { \mathbf } { \mathbf } { \mathbf } { \mathbf } { \mathbf } { \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf } { \mathbf } { \mathbf } { \mathbf } { \mathbf } { \mathbf \mathbf { } \mathbf } { \mathbf } { \mathbf \mathbf { } \mathbf } { \mathbf } { \mathbf \mathbf } { \mathbf } { \mathbf \mathbf }  $ . We simply take it as 

$$
\phi (\boldsymbol {v} x) = \alpha (\boldsymbol {v} x) \mathbb {I} (\boldsymbol {v} x > 0),
$$

so that the update is zeroed out for coordinates on which the sign of $\mathbf { \Delta } \mathbf { u } _ { t }$ and $\mathbf { \nabla } _ { \mathbf { \boldsymbol { g } } _ { t } }$ are inconsistent. Here $\alpha ( { \pmb x } )$ is a positive scaling factor, introduced to compensate the decrease of update magnitude due to masking. A simple choice of $\alpha$ is 

$$
\alpha (\boldsymbol {x}) = \frac {\dim (\boldsymbol {v} x)}{\operatorname{nnz} (\boldsymbol {v} x > 0) + \xi},\tag{1}
$$

where $\dim ( \cdot )$ and $\mathtt { n n z ( \cdot ) }$ represent the total number of elements and the number of non-zero elements of the input vector, respectively. Here, $\xi > 0$ is a positive constant, which we set to $\xi = 1$ by default. See Algorithms 1 and 2 for more details. 

This modification ensures the new negative update to have a non-negative inner product with the gradient, and hence decreases the loss monotonically when the step size is sufficiently small. Specifically, Taylor approximation shows 

$$
\mathcal {L} (\pmb {w} _ {t + 1}) - \mathcal {L} (\pmb {w} _ {t}) \approx - \epsilon_ {t} (\pmb {u} _ {t} \circ \pmb {g} _ {t}) ^ {\top} \phi (\pmb {u} _ {t} \circ \pmb {g} _ {t}) \leq 0.
$$

This ensures decrease of the loss, i.e., ${ \mathcal { L } } ( w _ { t + 1 } ) \leq { \mathcal { L } } ( w _ { t } )$ , when the step size is sufficiently small. In comparison, typical momentum-based optimizers do not always guarantee a monotonic decrease in loss even with infinitesimal step sizes. The nature of momentum dynamics introduces oscillations due to inertia-like effects. 

Our theoretical analysis shows that the modified algorithm converges to local optima under mild conditions on the base optimizers. An interesting aspect of this algorithm is that it does not get stuck at non-stationary points of the loss, even if $\mathbf { \pmb { u } } _ { t }$ can be temporarily completely conflicting with $\mathbf { \nabla } _ { \mathbf { \boldsymbol { g } } _ { t } }$ and is therefore entirely masked out. This is because, in typical momentum methods, the update direction $\mathbf { \pmb { u } } _ { t }$ continues to accumulate the gradients and will eventually be updated to have a positive inner product with $\mathbf { \nabla } _ { \mathbf { \boldsymbol { g } } _ { t } }$ if it is stuck at a non-stationary point. 

Our theoretical analysis encompasses both continuous-time and discrete-time cases. In the continuoustime setting, our theory shows that the modified algorithm guarantees convergence to local optima for optimizers that admit a Hamiltonian+Descent structure (Chen et al., 2023a; Liang et al., 2024), which broadly includes almost all existing popular algorithms, such as Adam, Lion, heavy ball, and Nesterov momentum. For these algorithms, we demonstrate that the cautious optimizers, with ϕ satisfying $x \phi ( x ) \ge \operatorname* { m a x } ( x , 0 )$ , retains the monotonic decreasing properties of the original Lyapunov (or Hamiltonian) functions of these algorithms while additionally minimizing the loss function. In the discrete-time setting, we analyze the behavior of various variants of mask functions and establish general conditions under which updates from cautious optimizers yield larger local descents compared to their base optimizers. 

To summarize our contributions, we present the following: 

• We propose Cautious Optimizers, a simple performance booster for any momentum-based optimizer, implemented with just a single line of code. 

• We theoretically demonstrate that cautious optimizers preserve the convergence guarantees of the base optimizer while also accelerating the decrease of the loss function. 

• We show consistent improvements across various tasks, from pretraining LLM to image classification. 

Algorithm 2 Cautious AdamW (C-AdamW)

Require: parameter w, step sizes $\{\epsilon_{t}\}$ , dampening factors $\beta_{1}, \beta_{2} \in [0,1)$ , e > 0, weight decay $\gamma \geq 0$ .

1: Initialize t = 0, and $m_{0}, v_{0}$ .

2: while $w_{t}$ not converged do

3: $t \leftarrow t + 1$ 4: $g_{t} \leftarrow \nabla_{w} L_{t}(w_{t-1})$ 5: $m_{t} \leftarrow \beta_{1} m_{t-1} + (1 - \beta_{1}) g_{t}$ 6: $v_{t} \leftarrow \beta_{2} v_{t-1} + (1 - \beta_{2}) g_{t}^{2}$ 7: $\hat{m}_{t} \leftarrow m_{t}/(1 - \beta_{1}^{t})$ 8: $\hat{v}_{t} \leftarrow v_{t}/(1 - \beta_{2}^{t})$ 9: $u_{t} \leftarrow \hat{m}_{t}/(\sqrt{\hat{v}_{t}} + e)$ 10: $\phi_{t} \leftarrow \mathbb{I}(u_{t} \circ g_{t} > 0)$ // Compute alignment mask

11: $\bar{\epsilon}_{t} = \epsilon_{t} \frac{d}{\|\phi_{t}\|_{0} + 1}$ //Scale lr, d is dimension of $\phi_{t}$ 12: $w_{t} \leftarrow w_{t-1} - \bar{\epsilon}_{t} \phi_{t} \circ u_{t}$ //Masked update

13: $w_{t} \leftarrow w_{t} - \bar{\epsilon}_{t} \gamma w_{t}$ //Add weight decay

14: end while 

## 2 THEORY

We start with introducing a general Hamiltonian descent framework for the continuous-time forms of general momentum algorithms (Section 2.1). We then introduce the cautious optimizers in the continuous time form and discuss its theoretical properties (Section 2.2). Finally, we discuss in Section 2.3 theoretical properties of cautious optimizers in discrete time forms. 

## 2.1 HAMILTONIAN+DESCENT

In the continuous-time form, most momentum-based algorithms can be viewed as variants of the damped Hamiltonian system, which admit a Lyapunov (or Hamiltonian) function that certifies their convergence towards the stationary points. 

The Lyapunov function is typically an augmented loss function $\mathcal { H } ( \boldsymbol { w } , \boldsymbol { s } )$ defined over both the weights w and an optimization state vector s (which includes the momentum). It must satisfy min<sub>s</sub> $\mathcal { H } ( \pmb { w } , \pmb { s } ) = \mathcal { L } ( \pmb { w } )$ , so that minimizing $\mathcal { L } ( w )$ is equivalent to minimizing $\mathcal { H } ( \boldsymbol { w } , \boldsymbol { s } )$ . This is typically achieved using a separable Hamiltonian of the form 

$$
\mathcal {H} (\boldsymbol {w}, \boldsymbol {s}) = \mathcal {L} (\boldsymbol {w}) + \mathcal {K} (\boldsymbol {s}),
$$

where $\kappa ( \cdot )$ is any lower-bounded function. Consulting physical intuitions, we can think of H as the total energy of a system parameterized by $( w , s )$ , with $\dot { \mathcal { L } }$ and K representing the potential energy and kinetic energy, respectively. 

The continuous-time form of common momentum-based algorithms can be unified into: 

$$
\begin{array}{l} \frac {\mathrm{d}}{\mathrm{d} t} \boldsymbol {w} _ {t} = - \nabla \mathcal {K} (\boldsymbol {s} _ {t}) - \Phi_ {t} (\nabla \mathcal {L} (\boldsymbol {w} _ {t})) \\ \frac {\mathrm{d}}{\mathrm{d} t} \boldsymbol {s} _ {t} = \nabla \mathcal {L} (\boldsymbol {w} _ {t}) - \Psi_ {t} (\nabla \mathcal {K} (\boldsymbol {s} _ {t})), \end{array}\tag{2}
$$

where $\Phi _ { t } ( \cdot )$ and $\Psi _ { t } ( \cdot )$ are monotonic mappings satisfying 

$$
\| \boldsymbol {x} \| _ {\Phi_ {t}} ^ {2} := \langle \boldsymbol {x}, \Phi_ {t} (\boldsymbol {x}) \rangle \geq 0, \quad \| \boldsymbol {x} \| _ {\Psi_ {t}} ^ {2} := \langle \boldsymbol {x}, \Psi_ {t} (\boldsymbol {x}) \rangle \geq 0,
$$

for any x. With $\Phi ( { \pmb x } ) = \Psi ( { \pmb x } ) = 0 .$ , the system in (2) reduces to the standard Hamiltonian system, which preserves $\mathcal { H } ( \pmb { w } _ { t } , \pmb { s } _ { t } ) =$ const along the trajectory. When the descending components $\Phi _ { t }$ and $\Psi _ { t }$ are added, the system ensures that $\mathcal { H } ( \boldsymbol { w } , \boldsymbol { s } )$ becomes monotonically non-decreasing: 

$$
\begin{array}{l l} & \frac {\mathrm{d}}{\mathrm{d} t} \mathcal {H} (\boldsymbol {w} _ {t}, \boldsymbol {s} _ {t}) = - \Delta_ {\mathcal {H}} (\boldsymbol {v} w _ {t}, \boldsymbol {v} s _ {t}) \leq 0, \\ \text {where} & \Delta_ {\mathcal {H}} (\boldsymbol {v} w _ {t}, \boldsymbol {v} s _ {t}) := \| \nabla \mathcal {L} (\boldsymbol {w} _ {t}) \| _ {\Phi_ {t}} ^ {2} + \| \nabla \mathcal {K} (\boldsymbol {s} _ {t}) \| _ {\Psi_ {t}} ^ {2}. \end{array}\tag{3}
$$

On the other hand, $\mathcal { L } ( w )$ , which is the true objective, is not necessarily decreasing monotonically. There can be cases where $\mathcal { L } ( w )$ increases temporarily in exchange for a large decrease in the kinetic energy $\mathcal { K } ( s )$ , while still ensuring a decrease in the total energy $\bar { \mathcal { H } } = \mathcal { L } + \bar { \mathcal { K } }$ . Specifically, 

$$
\begin{array}{c} \frac {\mathrm{d}}{\mathrm{d} t} \mathcal {L} (\boldsymbol {w} _ {t}) = - \Delta_ {\mathcal {L}} (\boldsymbol {w} _ {t}, \boldsymbol {s} _ {t}), \\ \Delta_ {\mathcal {L}} (\boldsymbol {w} _ {t}, \boldsymbol {s} _ {t}) := \nabla \mathcal {L} (\boldsymbol {w} _ {t}) ^ {\top} \nabla \mathcal {K} (\boldsymbol {s} _ {t}) + \| \nabla \mathcal {L} (\boldsymbol {w} _ {t}) \| _ {\Phi_ {t}} ^ {2}. \end{array}\tag{4}
$$

Here, $\Delta _ { \mathcal { L } } ( \boldsymbol { w } _ { t } , \boldsymbol { s } _ { t } )$ may be negative due to the cross term. 

See Appendix for the Hamiltonian of common optimizers including Adam (Kingma, 2014) and Lion-K (Chen et al., 2023b;a). 

## 2.2 CAUTIOUS DYNAMICS

Our idea is to change the dynamics to make it simultaneously decrease both $\mathcal { H } ( \boldsymbol { w } , \boldsymbol { s } )$ and $\mathcal { L } ( w )$ . We do this with a modified system: 

$$
\begin{array}{r l} & {\overline {{\boldsymbol {x}}} _ {t} = \nabla \mathcal {L} (\overline {{\boldsymbol {w}}} _ {t}) \circ \nabla \mathcal {K} (\overline {{\boldsymbol {s}}} _ {t})} \\ & {\frac {\mathrm{d}}{\mathrm{d} t} \overline {{\boldsymbol {w}}} _ {t} = - \phi (\overline {{\boldsymbol {x}}} _ {t}) \circ \nabla \mathcal {K} (\overline {{\boldsymbol {s}}} _ {t}) - \Phi_ {t} (\nabla \mathcal {L} (\overline {{\boldsymbol {w}}} _ {t}))} \\ & {\frac {\mathrm{d}}{\mathrm{d} t} \overline {{\boldsymbol {s}}} _ {t} = \nabla \mathcal {L} (\overline {{\boldsymbol {w}}} _ {t}) - \Psi_ {t} (\nabla \mathcal {K} (\overline {{\boldsymbol {s}}} _ {t})),} \end{array}\tag{5}
$$

where ◦ denotes the element-wise product and $\phi$ is a vector to vector mapping. Here we weigh each element of the update direction $\nabla \bar { \boldsymbol { K } } ( \overline { { \boldsymbol { s } } } _ { t } )$ based on the product of $\nabla K ( s )$ with the gradient $\bar { \nabla } \mathcal { L } ( \boldsymbol { w } )$ Note that we do not need to apply a mask on the $\Phi _ { t } ( \nabla \mathcal { L } ( \overline { { \mathbf { \mathscr { w } } } } _ { t } )$ term since it is always non-increasing by definition of $\Phi _ { t }$ 

The following conditions on the choice of function $\phi$ ensure that the system decreases both H and $\mathcal { L }$ simultaneously. 

Theorem 2.1. Following the dynamics in $( 5 ) i n \mathbb { R } ^ { d } ;$ , we have 

$$
\begin{array}{r l} & {\frac {\mathrm{d}}{\mathrm{d} t} \mathcal {H} (\overline {{\boldsymbol {w}}} _ {t}, \overline {{\boldsymbol {s}}} _ {t}) = (\overline {{\boldsymbol {x}}} _ {t} ^ {\top} (\mathbf {1} - \phi (\overline {{\boldsymbol {x}}} _ {t})) - \Delta_ {\mathcal {H} _ {t}} (\overline {{\boldsymbol {w}}} _ {t}, \overline {{\boldsymbol {s}}} _ {t}),} \\ & {\quad \frac {\mathrm{d}}{\mathrm{d} t} \mathcal {L} (\overline {{\boldsymbol {w}}} _ {t}) = - \overline {{\boldsymbol {x}}} _ {t} ^ {\top} \phi (\overline {{\boldsymbol {x}}} _ {t}) - \| \nabla \mathcal {L} (\overline {{\boldsymbol {w}}} _ {t}) \| _ {\Phi_ {t}} ^ {2}} \\ & {\qquad = (\overline {{\boldsymbol {x}}} _ {t} ^ {\top} (\mathbf {1} - \phi (\overline {{\boldsymbol {x}}} _ {t})) - \Delta_ {\mathcal {L} _ {t}} (\overline {{\boldsymbol {w}}} _ {t}, \overline {{\boldsymbol {s}}} _ {t}),} \end{array}
$$

Here, $\Delta _ { \mathcal { H } _ { t } } ( \overline { { \boldsymbol { w } } } _ { t } , \overline { { \boldsymbol { s } } } _ { t } )$ and $\Delta _ { \mathcal { L } _ { t } } ( \overline { { \boldsymbol { w } } } _ { t } , \overline { { \boldsymbol { s } } } _ { t } )$ , as defined in (3) and (4), respectively, represent the decreasing rates ofH and L in accordance with the original system (2). Hence: 

$\bullet I f { \pmb x } ^ { \top } ( \mathbf { 1 } - \phi ( { \pmb x } ) ) \leq 0 f o r a n y { \pmb x } \in \mathbb { R } ^ { d }$ , then both H and $\mathcal { L }$ decreases faster than the original system: 

$$
\begin{array}{l} \frac {\mathrm{d}}{\mathrm{d} t} \mathcal {H} (\overline {{\boldsymbol {w}}} _ {t}, \overline {{\boldsymbol {s}}} _ {t}) \leq - \Delta_ {\mathcal {H} _ {t}} (\overline {{\boldsymbol {w}}} _ {t}, \overline {{\boldsymbol {s}}} _ {t}) \leq 0, \\ \frac {\mathrm{d}}{\mathrm{d} t} \mathcal {L} (\overline {{\boldsymbol {w}}} _ {t}) \leq - \Delta_ {\mathcal {L} _ {t}} (\overline {{\boldsymbol {w}}} _ {t}, \overline {{\boldsymbol {s}}} _ {t}). \end{array}
$$

$I f x ^ { \top } \phi ( { \pmb x } ) \geq 0$ for any $\pmb { x } \in \mathbb { R } ^ { d }$ , then L decreases monotonically, $\begin{array} { r } { \frac { \mathrm { d } } { \mathrm { d } t } \mathcal { L } ( \overline { { \boldsymbol { w } } } _ { t } ) \leq 0 } \end{array}$ 

One sufficient condition for $\phi$ to satisfy both conditions in Theorem 2.1 is to enforce the following element-wise constraint: 

$$
\phi (\boldsymbol {v} x) _ {i} \geq 1 \text {   if   } x _ {i} > 0,
$$

and 

$$
\phi (\boldsymbol {v} x) _ {i} \leq 0 \text {   if   } x _ {i} <   0,\tag{6}
$$

where $\phi ( v x )$ denotes the i-th element of $\phi ( \pmb { v } x )$ . Under this condition, both H and $\mathcal { L }$ decrease monotonically following the cautious dynamics, at a rate faster than the original systems. In particular, the default choice $\phi ( \pmb { v } \bar { x } ) = \alpha ( \pmb { v } x ) \mathbb { I } ( \pmb { \dot { v } } \bar { x } \geq 0 )$ with $\alpha ( v x ) \geq 1$ satisfies the conditions above. 

Convergence to Stationary Points In addition to monotonically decreasing the loss function, we want to ensure that the algorithm does not get stuck unless the solution reaches a stationary point, which in practice is typically a local optimum, of the loss function. The following result demonstrates that this property holds under the same conditions as the original Hamiltonian descent system. 

Corollary 2.2. Assume that the norm $\| \cdot \| _ { \Psi } ^ { 2 }$ is positive definite, $\Psi ( 0 ) = 0 ,$ , and that $\mathcal { H } ( \boldsymbol { w } , \boldsymbol { s } ) =$ $\mathcal { L } ( \pmb { w } ) + \mathcal { K } ( \pmb { s } )$ is differentiable. Then, the bounded solutions of the original system (2) converge to a stationary point $o f \mathcal { H } ( \boldsymbol { w } , s )$ . Similarly, the bounded solutions of (5) also converge to a stationary point $o f \mathcal { H } ( \boldsymbol { w } , s )$ 

## 2.3 DISCRETE-TIME ANALYSIS

We analyze the discrete time case, demonstrating that each step of cautious optimizers are at least as good as the step of the original optimizers under mild conditions. 

We will consider a generic update of form 

$$
\begin{array}{l} \boldsymbol {w} _ {k + 1} = \boldsymbol {w} _ {k} - \epsilon_ {k} \boldsymbol {u} _ {k} (\boldsymbol {w} _ {k}, \boldsymbol {s} _ {k}), \\ \boldsymbol {s} _ {k + 1} = \boldsymbol {s} _ {k} + \boldsymbol {v} _ {k} (\boldsymbol {w} _ {k}, \boldsymbol {s} _ {k}), \end{array}\tag{7}
$$

where ${ \mathbf { } } u _ { k } , v _ { k }$ are vector fields that define the updates, and $\epsilon _ { k }$ is the step size. We write the cautious variants as 

$$
\begin{array}{l} \overline {{\boldsymbol {w}}} _ {k + 1} = \overline {{\boldsymbol {w}}} _ {k} - \epsilon_ {k} \overline {{\boldsymbol {u}}} _ {k} \circ \overline {{\boldsymbol {v}}} \phi_ {k}, \quad \overline {{\boldsymbol {u}}} _ {k} = \boldsymbol {u} _ {k} (\overline {{\boldsymbol {w}}} _ {k}, \overline {{\boldsymbol {s}}} _ {k}) \\ \overline {{\boldsymbol {s}}} _ {k + 1} = \overline {{\boldsymbol {s}}} _ {k} + \boldsymbol {v} _ {k} (\overline {{\boldsymbol {w}}} _ {k}, \overline {{\boldsymbol {s}}} _ {k}), \end{array}\tag{8}
$$

where $\overline { { \boldsymbol { v } \phi } } _ { k }$ is a mask vector determined by the algorithm, $\overline { { \pmb { v } \phi } } _ { k } = \alpha ( \overline { { \pmb { u } } } _ { k } \circ \overline { { \pmb { v } g } } _ { k } ) \mathbb { I } ( \overline { { \pmb { u } } } _ { k } \circ \overline { { \pmb { v } g } } _ { k } > 0 )$ where $\overline { { \boldsymbol { v } g } } _ { k } = \nabla \mathcal { L } ( \overline { { \boldsymbol { w } } } _ { k } )$ ). 

The following is a comparison result showing that each step of the cautious optimizer yields larger loss decrease than the original optimizer under mild conditions. 

Theorem 2.3. Consider (7) and (8) with a µ-smooth loss function $\mathcal { L } ( \cdot )$ . Assume the element-wise operator ϕ satisfies 

$$
\Delta (\boldsymbol {v} x) := - \boldsymbol {v} x ^ {\top} (1 - \phi (\boldsymbol {v} x)) \geq 0.
$$

Starting from $( { \pmb w } _ { t } , { \pmb s } _ { t } ) = ( \overline { { { \pmb w } } } _ { t } , \overline { { { \pmb s } } } _ { t } )$ , we have 

$$
\mathcal {L} (\overline {{\boldsymbol {w}}} _ {t + 1}) \leq \mathcal {L} (\boldsymbol {w} _ {t + 1}),
$$

which holdsfor step size $\begin{array} { r } { \epsilon _ { t } \leq \frac { 2 \Delta ( \overline { { \pmb { v } u } } _ { t } \circ \overline { { \pmb { v } g _ { t } } } ) } { \mu \| \overline { { \pmb { v } r } } _ { t } \| ( 2 \cdot \| \overline { { \pmb { u } } } _ { t } \| + \| \overline { { \pmb { v } r } } _ { t } \| ) } } \end{array}$ , where vr<sub>t</sub> = u<sub>t</sub>◦(1−vϕ<sub>t</sub>) and vg<sub>t</sub> = ∇L(w<sub>t</sub>). 

This result works only for a range of step sizes due to the need of Taylor approximation. In Appendix A.5, we show a case when the comparison holds for all step sizes when using a different mask function based on inner products and when L is convex. The following is another result showing that when imposing more restrictive conditions on the mask to the step size that ensure that the cautious optimizer is guaranteed to decrease loss at each step. 

Theorem 2.4. Consider updates (7) and (8). Assuming $\mathcal { L } ( \cdot )$ is µ-smooth, and consider the following maskfunction: 

$$
\overline {{\boldsymbol {v} \phi}} _ {k} = \alpha_ {k} \mathbb {I} \left(\nabla \mathcal {L} (\overline {{\boldsymbol {w}}} _ {k}) \circ \overline {{\boldsymbol {v} u}} _ {k} \geq \frac {\mu \sigma}{2} \overline {{\boldsymbol {v} u}} _ {k} \circ \overline {{\boldsymbol {v} u}} _ {k}\right),
$$

where $\{ \alpha _ { k } \}$ is any sequence and $\sigma \geq \epsilon _ { k } \alpha _ { k }$ . Starting from $( { \pmb w } _ { t } , { \pmb s } _ { t } ) = ( \overline { { { \pmb w } } } _ { t } , \overline { { { \pmb s } } } _ { t } )$ , we have 

$$
\mathcal {L} (\overline {{{\boldsymbol {w}}}} _ {k + 1}) \leq \mathcal {L} (\overline {{{\boldsymbol {w}}}} _ {k}).
$$

The results above demonstrate that cautious optimizers reduce the loss more efficiently than the original optimizers at each single step. A natural question is whether these comparison results can be extended to multiple steps. This, however, becomes challenging because, after the first step, the two optimizers explore different regions of the loss landscape, making it easy to construct counterexamples where one method outperforms the other. This is consistent with the nofree lunch theorems, which state that no single optimizer can dominate another across all possible loss functions (Wolpert & Macready, 1997). Nevertheless, it is reasonable to expect that the advantage observed in a single step would naturally extend to multiple steps for the practical loss functions encountered in deep learning, as evidenced by the experiments presented. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/cf6503dd-a7d4-4104-9ea0-a4d743d42f4b/4a8168fb52d3fa4a42157c2b120e7d60c57e3809e1bfafd3b75febe648ce0fd8.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/cf6503dd-a7d4-4104-9ea0-a4d743d42f4b/5014df0b681f4ba4e14feb49f5bf37b4c6e6cbfcd3d2851c9554360a4ddaa9e4.jpg)


$$
w ^ {0}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/cf6503dd-a7d4-4104-9ea0-a4d743d42f4b/488272db34c0dd176519745a3254565f83f415f308def9c03127ae2080340e22.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/cf6503dd-a7d4-4104-9ea0-a4d743d42f4b/2bbc6bb13a93a296db82ad226036724a1a59dee2835b2f5b9ef70c0306fd05dd.jpg)



Figure 1: We compare gradient descent with Polyak momentum (GDM) and its element-wise cautious variant (C-GDM), using gradient descent (GD) as a baseline. The step size for GD and the hyperparameters of GDM (including step size and momentum coefficients) are chosen to achieve the optimal convergence rates, which can be analytically derived (see, e.g., Goh (2017)). For cautious optimizers, step sizes ϵ and momentum coefficients $\beta$ are empirically tuned, as shown in Figure 2. Detailed experimental settings are described in Section 3.1. In Plot (a), we visualize the optimization trajectories of the three methods, starting from the initial point (1, 1) with zero-initialized momentum. Notably, C-GDM converges to the optimum with significantly reduced overshooting and oscillation, Plot (b) zooms in on the trajectories from Plot (a), focusing on a smaller region $( 0 . 0 2 \times 0 . 0 2 )$ for enhanced clarity. Furthermore, Plots (c) and (d) show that C-GDM consistently and monotonically decreases both the objective and the Hamiltonian associated with the original GDM, highlighting its superior performance in minimizing these metrics compared to GDM.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/cf6503dd-a7d4-4104-9ea0-a4d743d42f4b/6268c12c6c953191af59502bbab8c68649b233d583cea1753f473ec24a59eb39.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/cf6503dd-a7d4-4104-9ea0-a4d743d42f4b/f5319f5f126ce9aa957dd0e351d29c6d84276bf46106df5c3aa26220cdc5baab.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/cf6503dd-a7d4-4104-9ea0-a4d743d42f4b/7b47aaa67f16c1724cb0cbada09af60d23a80b71fc90ff02c3a28ef5bf4cc3e6.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/cf6503dd-a7d4-4104-9ea0-a4d743d42f4b/87439cc7038f813a4eb877111015da6856d5283595db18001843eca405c156c5.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/cf6503dd-a7d4-4104-9ea0-a4d743d42f4b/e473a1771d84d1eeb0128966268c9823f0cec0ef5ec1b8c95c82acc66109c11c.jpg)



Figure 2: Convergence rate heatmaps for the objective function with condition number $\kappa = 4$ (same setup as in 1). The heatmaps illustrate convergence rates, where values greater than 1 indicate divergence for the corresponding $( \epsilon , \beta )$ configuration. Smaller convergence rates correspond to faster convergence. From left to right, we show heatmaps for $\phi _ { c }$ with $c = 0 . 0 1 , 0 . { \overset { \cdot } { 1 } }$ and $\phi _ { \mathrm { i n n e r } }$ as shown in (13). Note that $\phi _ { 0 }$ represents a constant function, reducing to gradient descent with momentum (GDM), whose optimal convergence rate is given in closed form (Goh, 2017). From the heatmaps, it is clear that all cautious momentum variantsϕ<sub>c</sub> with $c = 0 . 0 1 , 0 . 1$ and $\phi _ { \mathrm { i n n e r } }$ demonstrate superior convergence rates compared to GDM.


## 3 EXPERIMENTS

In this section, we evaluate the performance of cautious optimizers compared to their standard counterparts, highlighting the benefits introduced by the cautious masking mechanism. We begin with a 2D toy experiment to provide a visual demonstration of how cautious masking improves optimization. Subsequently, we extend the evaluation to large-scale pretraining tasks for both language and vision models, comparing the performance of standard optimizers and their cautious variants. 

## 3.1 2D OPTIMIZATION TOY

We consider a 2D optimization problem with objective ${ \mathcal L } ( { \pmb w } ) = \kappa ( { \pmb w } _ { 1 } ) ^ { 2 } + ( { \pmb w } _ { 2 } ) ^ { 2 }$ , where $\textbf { \em w } =$ $( w _ { 1 } , w _ { 2 } ) \in \mathbb { R } ^ { 2 }$ is the parameter. Obviously, the optimum is at $\pmb { w } ^ { * } = ( 0 , 0 )$ . We set $\kappa = 4$ in our experiments. We apply gradient descent (GD), gradient descent with Polyak momentum (GDM), and cautious gradient descent with momentum (C-GDM) on this toy example, starting from $\pmb { w } _ { 0 } = ( 1 , 1 )$ Specifically, for GDM, we adopt the conventional momentum update: 

$$
\boldsymbol {s} _ {t} \leftarrow \beta \boldsymbol {s} _ {t - 1} + \nabla \mathcal {L} (\boldsymbol {w} _ {t}),
$$

$$
\pmb {w} _ {t} \leftarrow \pmb {w} _ {t - 1} - \epsilon \pmb {s} _ {t},
$$

where $\beta \in [ 0 , 1 )$ and ϵ is the learning rate. 

Observation: On the right of Figure 1, we compare GDM and C-GDM with the same hyperparameters $( \epsilon , \beta )$ . We ablate over different combinations of $( \beta , \epsilon ) \in \{ ( 0 . 0 1 , 0 . 5 ) , ( 0 . 0 1 , 0 . 9 ) , ( \bar { 0 . 0 1 , 0 . 9 9 } )$ (0.01, 0.999), (0.1, 0.99), (0.001, 0.99)}. Across all settings, C-GDM outperforms GDM, confirming the importance of cautious masking. Given the same $( \epsilon , \beta )$ , cautious is always not worse than momentum, and often it gives significant improvement, especially when the choice $( \epsilon , \beta )$ is suboptimal for momentum, meaning that cautious masking makes it more robust. From the left of Figure 1, one can see that GDM, due to the momentum, has fluctuating $\mathcal { L } ( { \boldsymbol w } _ { t } )$ , while C-GDM ensures that $\mathcal { L } ( { \boldsymbol w } _ { t } )$ monotonically decreases. In addition, C-GDM achieves a faster drop in terms of GDM’s Hamiltonian. 

In Figure 1, we compare GDM and C-GDM, each using their optimal $( \epsilon , \beta )$ . For GDM, the optimal values are derived theoretically (e.g., (Goh, 2017)), while for C-GDM, they are obtained through a grid search. Despite using the optimal configuration, GDM exhibits significant overshooting, oscillations, and slower loss convergence. In contrast, C-GDM achieves smoother trajectories, reduced overshooting, and faster convergence, demonstrating its superior stability and efficiency. 

We estimate an algorithm’s convergence rate as the slope of log $\mathcal { L } ( { \boldsymbol w } _ { t } )$ over time via linear regression and plot heatmaps over (ϵ, β)-space, where ϵ and β are the learning rate and momentum. Figure 2 shows that cautious methods achieve lower optimal convergence rates compared to the momentum method $( \frac { \sqrt { \kappa } - 1 } { \sqrt { \kappa } + 1 }$ , red dot). The heatmaps highlight that all cautious momentum variants outperform GDM in convergence rates. 

## 3.2 PRETRAINING LARGE LANGUAGE MODELS (LLMS)


We begin by investigating the language modeling task using a 100M LLaMA (Touvron et al., 2023) model as the foundational architecture. The models are trained on the C4 (Colossal Clean Crawled Corpus) dataset (Raffel et al., 2020), a large-scale web-crawled text corpus containing billions of tokens. We provide results from the following settings: we take a 100M model and train it with batch size up to 2 million tokens for 50 billion tokens (25× Chinchilla Optimal). For optimization, we employ AdamW (Loshchilov, 2017) and Lion (Chen et al., 2023c), two popular optimizers in modern language modeling, as baselines. These are compared with their cautious counterparts, which we term Cautious AdamW (C-AdamW) and Cautious Lion (C-Lion).


<table><tr><td>lr</td><td>1e-4</td><td>3e-4</td><td>1e-3</td><td>3e-3</td><td>1e-2</td><td>2e-2</td><td>3e-2</td><td>1e-1</td></tr><tr><td>AdamW</td><td>85.050</td><td>24.384</td><td>19.249</td><td>19.007</td><td>18.965</td><td>19.609</td><td>**</td><td>**</td></tr><tr><td>C-AdamW</td><td>-</td><td>-</td><td>19.065</td><td>18.771</td><td>18.684</td><td>18.821</td><td>-</td><td>-</td></tr><tr><td rowspan="3"></td><td>lr</td><td>3e-5</td><td>1e-4</td><td>3e-4</td><td>6e-4</td><td>1e-3</td><td>1e-2</td><td></td></tr><tr><td>Lion</td><td>**</td><td>28.250</td><td>21.401</td><td>21.937</td><td>**</td><td>**</td><td></td></tr><tr><td>C-Lion</td><td>-</td><td>21.354</td><td>19.795</td><td>20.403</td><td>20.977</td><td>**</td><td></td></tr></table>


Table 1: We report final evaluation perplex, the lower the better. ”**” are runs that did not converge due to either too large or too small learning rates. $^ { , , , , , }$ stands for runs we skip due to lack of baseline comparison. Each model is trained with batch size up to 2 million tokens for 50 billion tokens in total (25 × Chinchilla Optimal(Hoffmann et al., 2022)). We use $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 5$ and weight decay 0.1 on AdamW; Sequence lengths of all models are 1024. For Lion experiments, we follow the recommendation from the (Chen et al., $2 0 2 3 \mathrm { a } )$ and use $\beta _ { 1 } = 0 . 9 5$ and $\beta _ { 2 } = 0 . 9 8$ . For scheduler, we use CosineAnnealing with warmup and the learning rate is decayed to 10% of the initial learning rate. Gradient accumulation is set to 8 to increase the global batch size.


Observation: As shown in Table 1, Cautious Optimizers demonstrate consistent improvements in both evaluation perplexity and sample efficiency. Table 1 shows the cautious modification is robust across learning rates and Cautious doesn’t change the optimality of hyperparameter search done on the base optimizers. Surprisingly, Cautious can also tolerate a higher learning rate in the Lion experiments and achieve stable training even when the baseline diverges. 

To further confirm our finding, we also include a scaling experiment with C-AdamW on FineWeb-Edu (Penedo et al., 2024), a more recent and higher quality web-scale text dataset. After rigorous and thorough hyperparameter search, we found that C-AdamW is consistently outperforming baseline AdamW. In table 2, we report optimal results for each scale. 


Table 2: Perplexity comparison between AdamW and C-AdamW across different model scales at 1× Chinchilla (Hoffmann et al., 2022), hyperparameters are extensively searched with coordinate descent over a discrete grid. Details can be found in appendix C


<table><tr><td>Scale</td><td>AdamW</td><td>C-AdamW</td><td>Improvement (%)</td></tr><tr><td>130M</td><td>27.39</td><td>27.30</td><td>0.33</td></tr><tr><td>300M</td><td>18.30</td><td>18.28</td><td>0.10</td></tr><tr><td>520M</td><td>15.07</td><td>14.92</td><td>1.00</td></tr><tr><td>1.2B</td><td>11.36</td><td>11.32</td><td>0.32</td></tr></table>


Furthermore, we perform downstream evaluation on the produced 1.2B checkpoints with 1× Chinchilla (20× tpp) on 7 downstream tasks, where the checkpoint trained by cautious optimizer wins in 5 of them (MMLU, OpenBookQA, Arc Easy as well as HellaSwag and Arc Challenge).


<table><tr><td>Task / Group</td><td>Metric</td><td>AdamW</td><td>C-AdamW</td></tr><tr><td>Arc Easy (Clark et al., 2018)</td><td>acc</td><td><eq>0.6082 \pm 0.0100</eq></td><td><eq>0.6090 \pm 0.0100</eq></td></tr><tr><td>Arc Challenge (Clark et al., 2018)</td><td>acc_norm</td><td><eq>0.2875 \pm 0.0132</eq></td><td><eq>0.2978 \pm 0.0134</eq></td></tr><tr><td>Hellaswag (Zellers et al., 2019)</td><td>acc_norm</td><td><eq>0.4169 \pm 0.0049</eq></td><td><eq>0.4193 \pm 0.0049</eq></td></tr><tr><td>Lambada OpenAI (Radford et al., 2019)</td><td>acc</td><td><eq>0.3311 \pm 0.0066</eq></td><td><eq>0.3229 \pm 0.0065</eq></td></tr><tr><td>OpenBookQA (Mihaylov et al., 2018)</td><td>acc</td><td><eq>0.2340 \pm 0.0190</eq></td><td><eq>0.2360 \pm 0.0190</eq></td></tr><tr><td>PIQA (Bisk et al., 2020)</td><td>acc_norm</td><td><eq>0.6774 \pm 0.0109</eq></td><td><eq>0.6768 \pm 0.0109</eq></td></tr><tr><td>MMLU (Hendrycks et al., 2021)</td><td>acc</td><td><eq>0.2529 \pm 0.0037</eq></td><td><eq>0.2535 \pm 0.0037</eq></td></tr></table>


Table 3: Comparison of benchmark results between C-AdamW and AdamW across multiple tasks and MMLU groups. All evaluations are done with LM Eval-Harness(Gao et al., 2024). Bold indicates the better score.


## 3.3 IMAGE CLASSIFICATION

We also include a classic classification task on Mini-ImageNet on ViT (Dosovitskiy, 2020) with two additional optimizers MARS (Yuan et al., 2024) and LaProp (Ziyin et al., 2020), both are momentum-based optimizers and their cautious variants perform consistently better as shown in the table 4. 

<table><tr><td>Method</td><td>Eval_Top1</td></tr><tr><td>C-AdamW</td><td>73.52</td></tr><tr><td>AdamW</td><td>72.11</td></tr><tr><td>C-LaProp</td><td>73.92</td></tr><tr><td>LaProp (Ziyin et al., 2020)</td><td>71.73</td></tr><tr><td>C-MARS</td><td>74.91</td></tr><tr><td>MARS (Yuan et al., 2024)</td><td>74.06</td></tr></table>


Table 4: Top-1 evaluation accuracy on Mini-ImageNet, the higher the better. We can see that the cautious variant is better across base optimizer options. Hyperparameters can be found in appendix D


## 4 RELATED WORK

We provide a brief overview of existing efforts on designing Adam-like optimizers, and the related works on Hamiltonian dynamics. 

Adam and Its Variants A plethora of Adam variants have been developed to address different aspects of optimization challenges (Kingma, 2014; Loshchilov & Hutter, 2017). AdamW (Loshchilov & Hutter, 2017) introduced a key improvement by decoupling weight decay from optimization steps, restoring the original formulation of weight decay regularization. NAdam (Dozat, 2016) incorporated Nesterov updates into Adam, while AdaBelief (Zhuang et al., 2020) refined the second momentum $v _ { t }$ to track the EMA of $( g _ { t } - m _ { t } ) ^ { 2 }$ , improving generalization. Adan (Xie et al., 2024) added an extra momentum term for better training performance, albeit at the cost of additional memory usage. More recently, ADOPT (Taniguchi et al., 2024) innovated by folding normalized updates into firstorder momentum updates. From an efficiency perspective, several approaches target memory cost reduction. AdaFactor (Shazeer & Stern, 2018) factorizes second-order statistics into a row-column outer product, enabling sub-linear memory usage. K-Fac (Martens & Grosse, 2015) approximates the Fisher information matrix with a Kronecker-factored representation, supporting sublinear natural gradient updates. Techniques like fused gradient computation (Lv et al., 2023) further minimize memory costs.(Wang et al., 2024) is a concurrent work that focus on continuous learning setting. In contrast to these approaches, our proposed C-AdamW introduces a single-line modification to the widely used AdamW optimizer. This modification not only retains the simplicity and efficiency of AdamW but also eliminates the need for hyperparameter tuning, as the default parameters of AdamW suffice. Furthermore, while many of the aforementioned methods focus on optimizing or extending specific aspects of the Adam algorithm, C-AdamW is more generalit seamlessly integrates with all momentum-based optimizers, offering a general solution with minimal implementation effort. 

Hamiltonian Dynamics Hamiltonian dynamics, rooted in classical mechanics, provides a powerful mathematical framework for analyzing the motion of systems in continuous spaces. This perspective has gained traction in optimization, where the introduction of Hamiltonian principles sheds light on momentum-based algorithms (Sutskever et al., 2013; Nesterov, 1983; Nguyen et al., 2024; Anonymous, 2024). Unlike Gradient Descent (GD), which ensures a monotonic decrease in the objective function, momentum-based methods often follow non-monotonic trajectories, posing unique analytical challenges (Jin et al., 2018). To address this, researchers have developed multiple Lyapunov functions for convex settings (Krichene et al., 2015; Wilson et al., 2016), providing a structured approach to analyze convergence. (Sutskever et al., 2013) offered a physical interpretation of momentum in optimization, linking it to the dynamics described by Hamiltonian mechanics, and demonstrated how these principles underpin classical methods like those of Nesterov and Polyak (Nes terov, 1983). Furthermore, Hamiltonian dynamics have been instrumental in deriving convergence rates for accelerated methods (Jin et al., 2018) and, more recently, for advanced optimizers like Lion (Chen et al., 2023a) and its distributed variant (Liu et al., 2024). In a related vein, (Maddison et al., 2018) explored optimization methods from the perspective of continuous-time ODEs, emphasizing their Hamiltonian structure. Mirror Descent, a related framework, has been shown to maintain efficiency estimates with a mild dependence on the dimensionality of decision variables, making it particularly suitable for large-scale optimization problems (Tzen et al., 2023; Krichene et al., 2015). These advancements highlight the versatility and depth of Hamiltonian formalism in bridging the gap between optimization theory and practical algorithm design. 

## 5 CONCLUSION

In summary, we introduce Cautious Optimizers, an enhancement for momentum-based optimizers that can be implemented with a single line of code. Our theoretical analysis demonstrates that Cautious Optimizers not only preserve the convergence guarantees of the base optimizers but also accelerate the reduction of the loss function. Empirically, it delivers faster LLM pretraining and better accuracy on image classification. Finally, we suggest a few promising future directions: (1) Apply cautious optimizers to more settings such as reinforcement learning and continuous learning; (2) masking in the eigenspace rather than the parameter space; (3) rigorous analyses of how cautious optimizers strictly improve the convergence rate (empirically shown in 1). 

## 6 ACKNOWLEDGEMENT

This work was supported by the Amazon AI PhD Fellowship, and in part by the Institute for Foundations of Machine Learning (IFML) and the Office of Naval Research (ONR) under Grant No. N00014-25-1-2354. 

## REFERENCES



Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 





Anonymous. Improving adaptive moment optimization via preconditioner diagonalization. In Submitted to The Thirteenth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=NdNuKMEv9y. under review. 





Shane Bergsma, Nolan Dey, Gurpreet Gosal, Gavia Gray, Daria Soboleva, and Joel Hestness. Straight to zero: Why linearly decaying the learning rate to zero works best for llms. arXiv preprint arXiv:2502.15938, 2025. 





Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. Piqa: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pp. 7432–7439, 2020. 





Lizhang Chen, Bo Liu, Kaizhao Liang, and Qiang Liu. Lion secretly solves constrained optimization: As lyapunov predicts. arXiv preprint arXiv:2310.05898, 2023a. 





Xiangning Chen, Chen Liang, Da Huang, Esteban Real, Kaiyuan Wang, Yao Liu, Hieu Pham, Xuanyi Dong, Thang Luong, Cho-Jui Hsieh, et al. Symbolic discovery of optimization algorithms. arXiv preprint arXiv:2302.06675, 2023b. 





Xiangning Chen, Chen Liang, Da Huang, Esteban Real, Kaiyuan Wang, Yao Liu, Hieu Pham, Xuanyi Dong, Thang Luong, Cho-Jui Hsieh, et al. Symbolic discovery of optimization algorithms. arXiv preprint arXiv:2302.06675, 2023c. 





Xiangning Chen, Chen Liang, Da Huang, Esteban Real, Kaiyuan Wang, Hieu Pham, Xuanyi Dong, Thang Luong, Cho-Jui Hsieh, Yifeng Lu, et al. Symbolic discovery of optimization algorithms. Advances in neural information processing systems, 36, 2024. 





Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv:1803.05457v1, 2018. 





Aaron Defazio, Xingyu Alice Yang, Harsh Mehta, Konstantin Mishchenko, Ahmed Khaled, and Ashok Cutkosky. The road less scheduled. arXiv preprint arXiv:2405.15682, 2024. 





Alexey Dosovitskiy. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 





Timothy Dozat. Incorporating nesterov momentum into adam. 2016. 





Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. The language model evaluation harness, 07 2024. URL https://zenodo.org/records/12608602. 





Gabriel Goh. Why momentum really works. Distill, 2017. doi: 10.23915/distill.00006. URL http://distill.pub/2017/momentum. 





Vineet Gupta, Tomer Koren, and Yoram Singer. Shampoo: Preconditioned stochastic tensor optimiza tion. In International Conference on Machine Learning, pp. 1842–1850. PMLR, 2018. 





Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollar, and Ross Girshick. Masked´ autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 16000–16009, 2022. 





Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. Proceedings ofthe International Conference on Learning Representations (ICLR), 2021. 





Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022. 





Shengyi Huang, Michael Noukhovitch, Arian Hosseini, Kashif Rasul, Weixun Wang, and Lewis Tunstall. The n+ implementation details of rlhf with ppo: A case study on tl; dr summarization. arXiv preprint arXiv:2403.17031, 2024. 





Chi Jin, Praneeth Netrapalli, and Michael I Jordan. Accelerated gradient descent escapes saddle points faster than gradient descent. In Conference On Learning Theory, pp. 1042–1085. PMLR, 2018. 





Keller Jordan, Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse, and Jeremy Bernstein. Muon: An optimizer for hidden layers in neural networks, 2024. URL https: //kellerjordan.github.io/posts/muon/. 





Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020. 





Diederik P Kingma. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014. 





Walid Krichene, Alexandre Bayen, and Peter L Bartlett. Accelerated mirror descent in continuous and discrete time. Advances in neural information processing systems, 28, 2015. 





Kaizhao Liang, Bo Liu, Lizhang Chen, and Qiang Liu. Memory-efficient llm training with online subspace descent. arXiv preprint arXiv:2408.12857, 2024. 





Bo Liu, Lemeng Wu, Lizhang Chen, Kaizhao Liang, Jiaxu Zhu, Chen Liang, Raghuraman Krishnamoorthi, and Qiang Liu. Communication efficient distributed training with distributed lion. arXiv preprint arXiv:2404.00438, 2024. 





I Loshchilov. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 





Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 





Ilya Loshchilov, Frank Hutter, et al. Fixing weight decay regularization in adam. arXiv preprint arXiv:1711.05101, 5, 2017. 





Kai Lv, Yuqing Yang, Tengxiao Liu, Qi jie Gao, Qipeng Guo, and Xipeng Qiu. Full parameter fine-tuning for large language models with limited resources. In Annual Meeting ofthe Associationfor Computational Linguistics, 2023. URL https://api.semanticscholar.org/ CorpusID:259187846. 





Chris J Maddison, Daniel Paulin, Yee Whye Teh, Brendan O’Donoghue, and Arnaud Doucet. Hamiltonian descent methods. arXiv preprint arXiv:1809.05042, 2018. 





James Martens and Roger Baker Grosse. Optimizing neural networks with kronecker-factored approximate curvature. In International Conference on Machine Learning, 2015. URL https: //api.semanticscholar.org/CorpusID:11480464. 





Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering. In EMNLP, 2018. 





Yurii Evgenevich Nesterov. A method for solving the convex programming problem with convergence rate o (1/κˆ 2). In Dokl. akad. nauk Sssr, volume 269, pp. 543–547, 1983. 





Son Nguyen, Lizhang Chen, Bo Liu, and Qiang Liu. H-fac: Memory-efficient optimization with factorized hamiltonian descent. arXiv preprint arXiv:2406.09958, 2024. 





Guilherme Penedo, Hynek Kydlek, Anton Lozhkov, Margaret Mitchell, Colin A Raffel, Leandro Von Werra, Thomas Wolf, et al. The fineweb datasets: Decanting the web for the finest text data at scale. Advances in Neural Information Processing Systems, 37:30811–30849, 2024. 





PowerInfer. Powerinfer/qwq-longcot-500k, 2024. 





Qwen. Qwq: Reflect deeply on the boundaries of the unknown, November 2024. URL https: //qwenlm.github.io/blog/qwq-32b-preview/. 





Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. 2019. 





Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal ofmachine learning research, 21(140):1–67, 2020. 





Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-¨ resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695, 2022. 





Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. International journal ofcomputer vision, 115:211–252, 2015. 





John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. 





Noam Shazeer and Mitchell Stern. Adafactor: Adaptive learning rates with sublinear memory cost. In International Conference on Machine Learning, pp. 4596–4604. PMLR, 2018. 





Ilya Sutskever, James Martens, George Dahl, and Geoffrey Hinton. On the importance of initialization and momentum in deep learning. In International conference on machine learning, pp. 1139–1147. PMLR, 2013. 





Shohei Taniguchi, Keno Harada, Gouki Minegishi, Yuta Oshima, Seong Cheol Jeong, Go Nagahara, Tomoshi Iiyama, Masahiro Suzuki, Yusuke Iwasawa, and Yutaka Matsuo. Adopt: Modified adam can converge with any \beta 2 with the optimal rate. arXiv preprint arXiv:2411.02853, 2024. 





Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothee´ Lacroix, Baptiste Roziere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and` efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 





Belinda Tzen, Anant Raj, Maxim Raginsky, and Francis Bach. Variational principles for mirror descent and mirror langevin dynamics. IEEE Control Systems Letters, 2023. 





Nikhil Vyas, Depen Morwani, Rosie Zhao, Itai Shapira, David Brandfonbrener, Lucas Janson, and Sham Kakade. Soap: Improving and stabilizing shampoo using adam. arXiv preprint arXiv:2409.11321, 2024. 





Shaowen Wang, Anan Liu, Jian Xiao, Huan Liu, Yuekui Yang, Cong Xu, Qianqian Pu, Suncong Zheng, Wei Zhang, and Jian Li. Cadam: Confidence-based optimization for online learning. arXiv preprint arXiv:2411.19647, 2024. 





Ashia C Wilson, Benjamin Recht, and Michael I Jordan. A lyapunov analysis of momentum methods in optimization. arXiv preprint arXiv:1611.02635, 2016. 





David H Wolpert and William G Macready. No free lunch theorems for optimization. IEEE transactions on evolutionary computation, 1(1):67–82, 1997. 





Xingyu Xie, Pan Zhou, Huan Li, Zhouchen Lin, and Shuicheng Yan. Adan: Adaptive nesterov momentum algorithm for faster optimizing deep models. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024. 





An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115, 2024. 





Huizhuo Yuan, Yifeng Liu, Shuang Wu, Xun Zhou, and Quanquan Gu. Mars: Unleashing the power of variance reduction for training large models. arXiv preprint arXiv:2411.10438, 2024. 





Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 2019. 





Guoqiang Zhang, Niwa Kenta, and W Bastiaan Kleijn. Extending adamw by leveraging its second moment and magnitude. arXiv preprint arXiv:2112.06125, 2021. 



Juntang Zhuang, Tommy Tang, Yifan Ding, Sekhar C Tatikonda, Nicha Dvornek, Xenophon Papademetris, and James Duncan. Adabelief optimizer: Adapting stepsizes by the belief in observed gradients. Advances in neural information processing systems, 33:18795–18806, 2020. 

Liu Ziyin, Zhikang T Wang, and Masahito Ueda. Laprop: Separating momentum and adaptivity in adam. arXiv preprint arXiv:2002.04839, 2020. 

## A APPENDIX

The appendix is organized into the following key components: 

• Cautious Hamiltonian Descent: We provide conditions 2.1 on the choice of function ϕ ensure that the system decreases both H and L simultaneously and Corollary 2.2 asserts that with a positive definite norm and differentiable Hamiltonian, bounded solutions of the discussed systems converge to a stationary point of the Hamiltonian. 

• Applications to Optimizers: In section A.3, the cautious framework is applied to derive cautious variants of popular optimizers, including Adam, Signed Momentum, and Lion, highlighting their theoretical properties and practical advantages. 

• Discrete-Time Analysis: A rigorous analysis connects the continuous-time dynamics to discrete-time updates, providing convergence guarantees and bounds on loss reduction for cautious optimizers. 

• Pseudocode: We present the implementation details for cautious optimizers, focusing on the cautious Lion optimizer as a representative example. 

• Experimental Details: Comprehensive details on the experimental setup, hyperparameters, and hardware configurations are provided, demonstrating the effectiveness of cautious optimizers in accelerating convergence and improving performance on large-scale tasks such as language modeling and masked autoencoder pretraining. 

## A.1 HAMILTONIAN OF COMMON OPTIMIZERS

We introduce the Hamiltonian functions of the common optimizers. 

Example A.1. Adam (Kingma, 2014) yields the following continuous-time form and Hamiltonian, 

$$
\frac {\mathrm{d}}{\mathrm{d} t} \pmb {v} w _ {t} = - \frac {\pmb {v} m _ {t}}{\sqrt {\pmb {v} v _ {t}} + e},
$$

$$
\frac {\mathrm{d}}{\mathrm{d} t} \boldsymbol {v} m _ {t} = a (\nabla \mathcal {L} (\boldsymbol {v} w _ {t}) - \boldsymbol {v} m _ {t}),
$$

$$
\frac {\mathrm{d}}{\mathrm{d} t} \pmb {v} v _ {t} = b (\nabla \mathcal {L} (\pmb {v} w _ {t}) ^ {\odot 2} - \pmb {v} v _ {t}),
$$

$$
w i t h \quad \mathcal {H} (\boldsymbol {v} w, \boldsymbol {v} m, \boldsymbol {v} v) = \mathcal {L} (\boldsymbol {v} w) + \frac {1}{2 a} \langle \frac {\boldsymbol {v} m}{\sqrt {\boldsymbol {v} v} + e}, \boldsymbol {v} m a.
$$

We can show that $\begin{array} { r } { \frac { \mathrm { d } } { \mathrm { d } t } \mathcal { H } ( v w _ { t } , v m _ { t } , v v _ { t } ) \leq 0 } \end{array}$ when $a \ge b / 4$ 

Example A.2. The Lion-K optimizer (Chen et al., 2023b;a) (without weight decay) can be written into 

$$
\frac {\mathrm{d}}{\mathrm{d} t} \pmb {v} w _ {t} = \nabla \mathcal {K} ((1 - b) \pmb {v} m _ {t} - b \nabla \mathcal {L} (\pmb {v} w _ {t})),
$$

$$
\frac {\mathrm{d}}{\mathrm{d} t} \pmb {v} m _ {t} = - a (\nabla \mathcal {L} (\pmb {v} w _ {t}) + \pmb {v} m _ {t})
$$

where $a \ge 0 , b \in [ 0 , 1 ]$ and $\kappa ( \boldsymbol { v } x )$ is any convex function that attains the minimum at $v x = 0 .$ . One ofits Hamiltonians that yields the Hamiltonian+descent structure $( E q \left( l \boldsymbol { 3 } \right)$ in Chen et al. (Chen et al., 2023a)) is 

$$
\mathcal {H} (\pmb {v} w, \pmb {v} m) = a \mathcal {L} (\pmb {v} w) + \frac {1}{1 - b} \mathcal {K} ((1 - b) \pmb {v} m).
$$

See Chen et al. (2023a) for other Hamiltonian functions. Lion-K includes a large family algorithms as special cases, including Polyka momentum, Nesterov momentum, signed momentum, mirror descent, Frank-Wolfe, etc. 

## A.2 CAUTIOUS DYNAMICS

We establish conditions in Theorem 2.1 for the function $\phi$ that ensure simultaneous decreases in both H and $\mathcal { L } .$ Corollary 2.2 further asserts that, with a positive definite norm and differentiable Hamiltonian, bounded solutions of the systems converge to a stationary point of the Hamiltonian. 

Theorem 2.1. Following the dynamics in (5) in $\mathbb { R } ^ { d }$ , we have 

$$
\frac {\mathrm{d}}{\mathrm{d} t} \mathcal {H} (\overline {{\boldsymbol {w}}} _ {t}, \overline {{\boldsymbol {s}}} _ {t}) = (\overline {{\boldsymbol {x}}} _ {t} ^ {\top} (\mathbf {1} - \phi (\overline {{\boldsymbol {x}}} _ {t})) - \Delta_ {\mathcal {H} _ {t}} (\overline {{\boldsymbol {w}}} _ {t}, \overline {{\boldsymbol {s}}} _ {t}),
$$

and 

$$
\begin{array}{c} \frac {\mathrm{d}}{\mathrm{d} t} \mathcal {L} (\overline {{\boldsymbol {w}}} _ {t}) = - \overline {{\boldsymbol {x}}} _ {t} ^ {\top} \phi (\overline {{\boldsymbol {x}}} _ {t}) - \| \nabla \mathcal {L} (\overline {{\boldsymbol {w}}} _ {t}) \| _ {\Phi_ {t}} ^ {2} \\ = (\overline {{\boldsymbol {x}}} _ {t} ^ {\top} (\mathbf {1} - \phi (\overline {{\boldsymbol {x}}} _ {t})) - \Delta_ {\mathcal {L} _ {t}} (\overline {{\boldsymbol {w}}} _ {t}, \overline {{\boldsymbol {s}}} _ {t}), \end{array}
$$

Here, $\Delta _ { \mathcal { H } _ { t } } ( \overline { { \mathbf { w } } } _ { t } , \overline { { \mathbf { s } } } _ { t } )$ and $\Delta _ { \mathcal { L } _ { t } } ( \overline { { \boldsymbol { w } } } _ { t } , \overline { { \boldsymbol { s } } } _ { t } )$ , as defined in (3) and (4), respectively, represent the decreasing rates of H and L in accordance with the original system (2). Hence: 

• If $\pmb { x } ^ { \top } ( \mathbf { 1 } - \phi ( \pmb { x } ) ) \leq 0$ for any $\pmb { x } \in \mathbb { R } ^ { d }$ , then both H and L decreases faster than the original system: 

$$
\begin{array}{l} \frac {\mathrm{d}}{\mathrm{d} t} \mathcal {H} (\overline {{\boldsymbol {w}}} _ {t}, \overline {{\boldsymbol {s}}} _ {t}) \leq - \Delta_ {\mathcal {H} _ {t}} (\overline {{\boldsymbol {w}}} _ {t}, \overline {{\boldsymbol {s}}} _ {t}) \leq 0, \\ \frac {\mathrm{d}}{\mathrm{d} t} \mathcal {L} (\overline {{\boldsymbol {w}}} _ {t}) \leq - \Delta_ {\mathcal {L} _ {t}} (\overline {{\boldsymbol {w}}} _ {t}, \overline {{\boldsymbol {s}}} _ {t}). \end{array}
$$

$\mathrm { I f } { \pmb x } ^ { \top } \phi ( { \pmb x } ) \ge 0$ for any $\pmb { x } \in \mathbb { R } ^ { d }$ , then $\mathcal { L }$ decreases monotonically, 

$$
\frac {\mathrm{d}}{\mathrm{d} t} \mathcal {L} (\overline {{\boldsymbol {w}}} _ {t}) \leq 0.
$$

Proof. For simplicity, we write $\overline { { \pmb { x } } } _ { t } = \nabla \mathcal { L } ( \overline { { \pmb { w } } } _ { t } ) \circ \nabla \mathcal { K } ( \overline { { \pmb { s } } } _ { t } )$ . 

Recall the definition of $\Delta _ { \mathcal { H } _ { t } } ( \overline { { \mathbf { w } } } _ { t } , \overline { { \mathbf { s } } } _ { t } )$ and $\Delta _ { \mathcal { L } _ { t } } ( \overline { { \boldsymbol { w } } } _ { t } , \overline { { \boldsymbol { s } } } _ { t } )$ 

$$
\begin{array}{l} \Delta_ {\mathcal {H}} (\overline {{\boldsymbol {w}}} _ {t}, \overline {{\boldsymbol {s}}} _ {t}) := \| \nabla \mathcal {L} (\overline {{\boldsymbol {w}}} _ {t}) \| _ {\Phi_ {t}} ^ {2} + \| \nabla \mathcal {K} (\overline {{\boldsymbol {s}}} _ {t}) \| _ {\Psi_ {t}} ^ {2} \\ \Delta_ {\mathcal {L}} (\overline {{\boldsymbol {w}}} _ {t}, \overline {{\boldsymbol {s}}} _ {t}) := \nabla \mathcal {L} (\overline {{\boldsymbol {w}}} _ {t}) ^ {\top} \nabla \mathcal {K} (\overline {{\boldsymbol {s}}} _ {t}) + \| \nabla \mathcal {L} (\overline {{\boldsymbol {w}}} _ {t}) \| _ {\Phi_ {t}} ^ {2}. \end{array}
$$

Following the dynamics in (5), let us see the derivation of H w.r.t. t: 

$$
\begin{array}{r l} & {\frac {\mathrm{d}}{\mathrm{d} t} \mathcal {H} (\overline {{\boldsymbol {w}}} _ {t}, \overline {{\boldsymbol {s}}} _ {t})} \\ & {= \nabla \mathcal {L} (\overline {{\boldsymbol {w}}} _ {t}) ^ {\top} \dot {\overline {{\boldsymbol {w}}}} _ {t} + \nabla \mathcal {K} (\overline {{\boldsymbol {s}}} _ {t}) ^ {\top} \dot {\overline {{\boldsymbol {s}}}} _ {t}} \\ & {= \nabla \mathcal {L} (\overline {{\boldsymbol {w}}} _ {t}) ^ {\top} (- \nabla \mathcal {K} (\overline {{\boldsymbol {s}}} _ {t}) \circ \phi (\overline {{\boldsymbol {x}}} _ {t}) - \Phi_ {t} (\nabla \mathcal {L} (\overline {{\boldsymbol {w}}} _ {t}))) + \nabla \mathcal {K} (\overline {{\boldsymbol {s}}} _ {t}) ^ {\top} (\nabla \mathcal {L} (\overline {{\boldsymbol {w}}} _ {t}) - \Psi_ {t} (\nabla \mathcal {K} (\overline {{\boldsymbol {s}}} _ {t})))} \\ & {= \nabla \mathcal {L} (\overline {{\boldsymbol {w}}} _ {t}) ^ {\top} (\nabla \mathcal {K} (\overline {{\boldsymbol {s}}} _ {t}) \circ (\mathbf {1} - \phi (\overline {{\boldsymbol {x}}} _ {t}))) - \nabla \mathcal {L} (\overline {{\boldsymbol {w}}} _ {t}) ^ {\top} \Phi_ {t} (\nabla \mathcal {L} (\overline {{\boldsymbol {w}}} _ {t})) - \nabla \mathcal {K} (\overline {{\boldsymbol {s}}} _ {t}) ^ {\top} \Psi_ {t} (\nabla \mathcal {K} (\overline {{\boldsymbol {s}}} _ {t}))} \\ & {= \mathbf {1} ^ {\top} ((\nabla \mathcal {L} (\overline {{\boldsymbol {w}}} _ {t}) \circ \nabla \mathcal {K} (\overline {{\boldsymbol {s}}} _ {t})) \circ (\mathbf {1} - \phi (\nabla \mathcal {L} (\overline {{\boldsymbol {w}}} _ {t}) \circ \nabla \mathcal {K} (\overline {{\boldsymbol {s}}} _ {t})))) - \| \nabla \mathcal {L} (\overline {{\boldsymbol {w}}} _ {t}) \| _ {\Phi} ^ {2} - \| \nabla \mathcal {K} (\overline {{\boldsymbol {s}}} _ {t}) \| _ {\Psi_ {t}} ^ {2}} \\ & {= \overline {{\boldsymbol {x}}} _ {t} ^ {\top} (\mathbf {1} - \phi (\overline {{\boldsymbol {x}}} _ {t})) - \Delta_ {\mathcal {H}} (\overline {{\boldsymbol {w}}} _ {t}, \overline {{\boldsymbol {s}}} _ {t}).} \end{array}\tag{9}
$$

Given the fact that ϕ is an element-wise operator, it is noteworthy that if $x \cdot ( 1 - \phi ( x ) ) \leq 0$ then the first term in $( 9 ) \mathbf { 1 } ^ { \top } \left( \left( \nabla \mathcal { L } ( \overline { { \boldsymbol w } } ) \circ \nabla \bar { K } ( \overline { { \boldsymbol s } } ) \right) \circ \left( \mathbf { 1 } - \phi ( \nabla \mathcal { L } ( \overline { { \boldsymbol w } } ) \circ \nabla K ( \overline { { \boldsymbol s } } ) ) \right) \right) \boldsymbol \leq 0$ since we see $\nabla \mathcal { L } ( \overline { { \boldsymbol { w } } } ) \circ \nabla \mathcal { K } ( \overline { { \boldsymbol { s } } } )$ as x. 

Next, let us look into the derivative of $\mathcal { L } ( \overline { { { \boldsymbol { w } } } } _ { t } )$ w.r.t. t: 

$$
= - \nabla \mathcal {L} (\overline {{\boldsymbol {w}}} _ {t}) ^ {\top} \left(\nabla \mathcal {K} (\overline {{\boldsymbol {s}}} _ {t}) \circ \phi (\nabla \mathcal {L} (\overline {{\boldsymbol {w}}} _ {t}) \circ \nabla \mathcal {K} (\overline {{\boldsymbol {s}}} _ {t}))\right) - \nabla \mathcal {L} (\overline {{\boldsymbol {w}}} _ {t}) ^ {\top} \Phi_ {t} (\nabla \mathcal {L} (\overline {{\boldsymbol {w}}} _ {t}))
$$

$$
= - \overline {{\boldsymbol {x}}} _ {t} ^ {\top} \phi (\overline {{\boldsymbol {x}}} _ {t}) - \| \nabla \mathcal {L} (\overline {{\boldsymbol {w}}} _ {t}) \| _ {\Phi_ {t}} ^ {2}\tag{10}
$$

$$
= (\overline {{\boldsymbol {x}}} _ {t} ^ {\top} (\mathbf {1} - \phi (\overline {{\boldsymbol {x}}} _ {t})) - \Delta_ {\mathcal {L} _ {t}} (\overline {{\boldsymbol {w}}} _ {t}, \overline {{\boldsymbol {s}}} _ {t}).\tag{11}
$$

It is evident that if ${ \pmb x } ^ { \top } \phi ( { \pmb x } ) \ge 0$ for any $\pmb { x } \in \mathbb { R } ^ { d }$ , then the first term in $( 1 0 ) , \overline { { \mathbf { x } } } _ { t } ^ { \top } \phi ( \overline { { \mathbf { x } } } _ { t } )$ , satisfies $\overline { { \pmb { x } } } _ { t } ^ { \top } \phi ( \overline { { \pmb { x } } } _ { t } ) \ge 0$ . Consequently, $\begin{array} { r } { \frac { d } { d t } \mathcal { L } ( \overline { { \pmb { w } } } _ { t } ) \leq 0 } \end{array}$ . This condition holds when ${ \pmb x } ^ { \top } \phi ( { \pmb x } ) \ge 0$ for all $\pmb { x } \in \mathbb { R } ^ { d }$ 

It is noteworthy that if $\pmb { x } ^ { \top } ( \mathbf { 1 } - \phi ( \pmb { x } ) ) \leq 0$ for any $\pmb { x } \in \mathbb { R } ^ { d }$ , then the first term in (11), $\overline { { { \pmb x } } } _ { t } ^ { \top } ( { \bf 1 } - \phi ( \overline { { { \pmb x } } } _ { t } ) ) \leq$ 0, holds. Consequently, we have $\begin{array} { r } { \frac { d } { d t } \mathcal { L } ( \overline { { \pmb w } } _ { t } ) \leq - { \Delta _ { \mathcal { L } _ { t } } } ( \overline { { \pmb w } } _ { t } , \overline { { \pmb s } } _ { t } ) } \end{array}$ 

Corollary 2.2. Assume that the norm $\| \cdot \| _ { \Psi } ^ { 2 }$ is positive definite, $\Psi ( 0 ) = 0$ , and that $\mathcal { H } ( \boldsymbol { w } , \boldsymbol { s } ) =$ $\mathcal { L } ( \pmb { w } ) + \mathcal { K } ( \pmb { s } )$ is differentiable. Then, the bounded solutions of the original system (2) converge to a stationary point of $\mathcal { H } ( \boldsymbol { w } , \boldsymbol { s } )$ . Similarly, the bounded solutions of (5) also converge to a stationary point of $\mathcal { H } ( \boldsymbol { w } , \boldsymbol { s } )$ 

Proof. First, Let us look at system (2), we use LaSalle’s invariance principle to find the conditions that the accumulation points (positive limit points) $\boldsymbol { \mathbf { \rho } } _ { | } ( \boldsymbol { \mathbf { \mathit { w } } } ^ { * } , \boldsymbol { \mathbf { s } } ^ { * } )$ satisfy: 

$$
\| \nabla \mathcal {L} (\boldsymbol {w} ^ {*}) \| _ {\Phi_ {t}} ^ {2} = \| \nabla \mathcal {K} (\boldsymbol {s} ^ {*}) \| _ {\Psi_ {t}} ^ {2} = 0.
$$

By the assumption that $\| \cdot \| _ { \Psi _ { t } } ^ { 2 }$ is positive definite and $\Psi _ { t } ( 0 ) = 0$ , we have $\nabla \mathcal { K } ( s ^ { * } ) = 0$ 

For positive limit points of systems (2), if $\nabla \mathcal { L } ( \pmb { w } ^ { * } ) \neq 0$ , then the point $( w ^ { * } , s ^ { * } )$ is not a positive limit point since 

$$
\dot {\boldsymbol {s}} ^ {*} = \nabla \mathcal {L} (\boldsymbol {w} ^ {*}) - \Psi_ {t} (\nabla \mathcal {K} (\boldsymbol {s} ^ {*})) = \nabla \mathcal {L} (\boldsymbol {w} ^ {*}) \neq 0.
$$

Thus, $\nabla \mathcal { L } ( \boldsymbol { w } ^ { * } ) = 0$ . Together with $\nabla \mathcal { K } ( s ^ { * } ) = 0$ , we conclude that $( w ^ { * } , s ^ { * } )$ is a stationary point of $\mathcal { H } ( w , s )$ 

For system (5), the proof follows the same reasoning. 

## A.3 EXAMPLES OF CAUTIOUS DYNAMICS

In this section, we instantiate results on Adam, Signed Momentum, and Lion, along with their cautious variants. While the standard methods are widely used, our focus is on introducing and analyzing the cautious versions of these algorithms to better understand their dynamics and stability. 

## A.3.1 CAUTIOUS ADAM

Adam: We begin by recalling the continuous-time dynamics of the Adam optimizer: 

$$
\frac {\mathrm{d}}{\mathrm{d} t} \boldsymbol {w} _ {t} = - \frac {\boldsymbol {m} _ {t}}{\sqrt {\boldsymbol {v} _ {t}} + \epsilon},
$$

$$
\frac {d}{d t} \pmb {m} _ {t} = \beta_ {1} \cdot (\nabla \mathcal {L} (\pmb {w} _ {t}) - \pmb {m} _ {t}),
$$

$$
\frac {d}{d t} \pmb {v} _ {t} = \beta_ {2} \cdot (\nabla \mathcal {L} (\pmb {w} _ {t}) ^ {\odot 2} - \pmb {v} _ {t}),
$$

where the associated Hamiltonian is given by: 

$$
\mathcal {H} (\boldsymbol {w} _ {t}, \boldsymbol {m} _ {t}, \boldsymbol {v} _ {t}) = \mathcal {L} (\boldsymbol {w} _ {t}) + \frac {1}{2 \beta_ {1}} \left\langle \frac {\boldsymbol {m} _ {t}}{\sqrt {\boldsymbol {v} _ {t}} + \epsilon}, \boldsymbol {m} _ {t} \right\rangle .\tag{12}
$$

The time evolution of the Hamiltonian can be derived as: 

$$
\frac {d \mathcal {H} (\boldsymbol {w} _ {t} , \boldsymbol {m} _ {t} , \boldsymbol {v} _ {t})}{d t} = - \frac {\beta_ {2}}{4 \beta_ {1}} \left\langle \frac {\boldsymbol {m} _ {t} ^ {\odot 2}}{\boldsymbol {v} _ {t} ^ {3 / 2} + \epsilon}, \nabla \mathcal {L} (\boldsymbol {w} _ {t}) ^ {\odot 2} \right\rangle - \left(1 - \frac {\beta_ {2}}{4 \beta_ {1}}\right) \left\langle \frac {\boldsymbol {m} _ {t}}{\sqrt {\boldsymbol {v} _ {t}} + \epsilon}, \boldsymbol {m} _ {t} \right\rangle .
$$

For stability, we require $\begin{array} { r } { \frac { d \mathcal { H } } { d t } \le 0 } \end{array}$ , which leads to $\beta _ { 1 } \geq \frac { \beta _ { 2 } } { 4 }$ 

Cautious Adam: We now introduce a cautious variant of Adam. In this case, the update for ${ \pmb w } _ { t }$ is modified by introducing an indicator function that enforces alignment between the gradient and the momentum: 

$$
\frac {\mathrm{d}}{\mathrm{d} t} \boldsymbol {w} _ {t} = - \frac {\mathbb {I} (\nabla \mathcal {L} (\boldsymbol {w} _ {t}) \circ \boldsymbol {m} _ {t} > 0) \circ \boldsymbol {m} _ {t}}{\sqrt {\boldsymbol {v} _ {t}} + \epsilon}.
$$

The dynamics for ${ \mathbf { } } m _ { t }$ and ${ \mathbf { } } v _ { t }$ remain the same. It is straightforward to verify that the loss function $\mathcal { L } ( { \boldsymbol w } _ { t } )$ serves as a Lyapunov function: 

$$
\frac {\mathrm{d}}{\mathrm{d} t} \mathcal {L} (\boldsymbol {w} _ {t}) = - \nabla \mathcal {L} (\boldsymbol {w} _ {t}) ^ {\top} \frac {\mathbb {I} (\nabla \mathcal {L} (\boldsymbol {w} _ {t}) \circ \boldsymbol {m} _ {t} > 0) \circ \boldsymbol {m} _ {t}}{\sqrt {\boldsymbol {v} _ {t}} + \epsilon} \leq 0.
$$

Meanwhile, by Theorem 2.1, $\begin{array} { r } { \mathcal { H } ( \pmb { w } _ { t } , \pmb { m } _ { t } , \pmb { v } _ { t } ) = \mathcal { L } ( \pmb { w } _ { t } ) + \frac { 1 } { 2 \beta _ { 1 } } \left. \frac { \pmb { m } _ { t } } { \sqrt { \pmb { v } _ { t } } + \epsilon } , \pmb { m } _ { t } \right. } \end{array}$ is also monotonically decreasing. Therefore, H remains a valid Hamiltonian for Cautious Adam. 

## A.3.2 CAUTIOUS SIGNED (POLYAK) MOMENTUM

Signed Momentum: The update rule for Signed Momentum is: 

$$
\begin{array}{l} \dot {\boldsymbol {w}} _ {t} = - \text {sign} (\boldsymbol {m} _ {t}), \\ \dot {\boldsymbol {m}} _ {t} = \nabla \mathcal {L} (\boldsymbol {w} _ {t}) - \boldsymbol {m} _ {t}. \end{array}
$$

The Hamiltonian for this system is: 

$$
\mathcal {H} (\boldsymbol {w} _ {t}, \boldsymbol {m} _ {t}) = \mathcal {L} (\boldsymbol {w} _ {t}) + \| \boldsymbol {m} _ {t} \| _ {1},
$$

with time evolution: 

$$
\frac {d \mathcal {H}}{d t} = - \| \boldsymbol {m} _ {t} \| _ {1}.
$$

Cautious Signed Momentum: The cautious variant modifies the update rule for ${ \pmb w } _ { t }$ by introducing an indicator: 

$$
\dot {\boldsymbol {w}} _ {t} = - \operatorname{sign} (\boldsymbol {m} _ {t}) \odot \mathbb {I} (\nabla \mathcal {L} (\boldsymbol {w} _ {t}) \circ \boldsymbol {m} _ {t} > 0).
$$

The Lyapunov function remains the loss $\mathcal { L } ( { \boldsymbol { \mathbf { \mathit { w } } } _ { t } } )$ , with: 

$$
\frac {d \mathcal {L} (\boldsymbol {w} _ {t})}{d t} = - \| \nabla \mathcal {L} (\boldsymbol {w} _ {t}) \odot \mathbb {I} (\nabla \mathcal {L} (\boldsymbol {w} _ {t}) \circ \boldsymbol {m} _ {t} > 0) \| _ {1}.
$$

By Theorem 2.1, $\mathcal { H } ( \pmb { w } _ { t } , \pmb { m } _ { t } ) = \mathcal { L } ( \pmb { w } _ { t } ) + \| \pmb { m } _ { t } \| _ { 1 }$ is also monotonically decreasing. Therefore, H remains a valid Hamiltonian for Cautious Signed Momentum. 

## A.3.3 CAUTIOUS LION

Lion: The dynamics of Lion Chen et al. (2023a) are given by: 

$$
\begin{array}{l} \dot {\boldsymbol {m}} _ {t} = \alpha \nabla \mathcal {L} (\boldsymbol {w} _ {t}) - \gamma \boldsymbol {m} _ {t}, \\ \dot {\boldsymbol {w}} _ {t} = - \mathrm{sign} (\tilde {\boldsymbol {m}} _ {t}), \end{array}
$$

where $\tilde { \pmb { m } } _ { t } = \pmb { m } _ { t } - \varepsilon \big ( \alpha \nabla \mathcal { L } ( \pmb { w } _ { t } ) + \gamma \pmb { m } _ { t } \big )$ . The associated Hamiltonian is: 

$$
\mathcal {H} (\boldsymbol {w} _ {t}, \boldsymbol {m} _ {t}) = \alpha \mathcal {L} (\boldsymbol {w} _ {t}) + (1 - \varepsilon \gamma) \| \boldsymbol {m} _ {t} \| _ {1}.
$$

The time derivative is: 

$$
\dot {\mathcal {H}} (\boldsymbol {w} _ {t}, \boldsymbol {m} _ {t}) = - (1 - \varepsilon \gamma) \| \tilde {\boldsymbol {m}} _ {t} \| _ {1} - \gamma \| \boldsymbol {m} _ {t} \| _ {1}.
$$

Cautious Lion: In the cautious version, the update for ${ \pmb w } _ { t }$ is modified: 

$$
\dot {\boldsymbol {w}} _ {t} = - \operatorname{sign} (\tilde {\boldsymbol {m}} _ {t}) \odot \mathbb {I} (\nabla \mathcal {L} (\boldsymbol {w} _ {t}) \circ \tilde {\boldsymbol {m}} _ {t} > 0).
$$

By Theorem 2.1, $\mathcal { H } ( \pmb { w } _ { t } , \pmb { m } _ { t } ) = \alpha \mathcal { L } ( \pmb { w } _ { t } ) + ( 1 - \varepsilon \gamma ) \| \pmb { m } _ { t } \| _ { 1 }$ is also monotonically decreasing. Therefore, H remains a valid Hamiltonian for Cautious Lion. 

Meanwhile, the Lyapunov function remains $\mathcal { L } ( { \boldsymbol { \mathbf { \mathit { w } } _ { t } } } )$ , with: 

$$
\frac {d \mathcal {L} (\boldsymbol {w} _ {t})}{d t} = - \| \nabla \mathcal {L} (\boldsymbol {w} _ {t}) \odot \mathbb {I} (\nabla \mathcal {L} (\boldsymbol {w} _ {t}) \circ \tilde {\boldsymbol {m}} _ {t} > 0) \| _ {1}.
$$

## A.4 DISCRETE TIME ANALYSIS

We analyze the discrete time case, demonstrating that cautious optimizers are at least as good as the original optimizers under mild conditions. 

Theorem 2.3. Consider (7) and (8) with the loss function $\mathcal { L } ( \cdot )$ µ-smooth. Assume the element-wise operator ϕ satisfies 

$$
\Delta (\boldsymbol {v} x) := - \boldsymbol {v} x ^ {\top} (1 - \phi (\boldsymbol {v} x)) \geq 0.
$$

starting from $( { \pmb w } _ { t } , { \pmb s } _ { t } ) = ( \overline { { { \pmb w } } } _ { t } , \overline { { { \pmb s } } } _ { t } )$ , we have 

$$
\mathcal {L} (\overline {{\boldsymbol {w}}} _ {t + 1}) \leq \mathcal {L} (\boldsymbol {w} _ {t + 1}),
$$

which holds for step size $\begin{array} { r } { \epsilon _ { t } \leq \frac { 2 \Delta ( \overline { { \pmb { v } u } } _ { t } \circ \overline { { \pmb { v } g _ { t } } } ) } { \mu \| \overline { { \pmb { v } r } } _ { t } \| ( 2 \cdot \| \overline { { \pmb { u } } } _ { t } \| + \| \overline { { \pmb { v } r } } _ { t } \| ) } } \end{array}$ , where $\overline { { \pmb { v } r } } _ { t } = \overline { { \pmb { u } } } _ { t } \circ ( { \bf 1 } - \overline { { \pmb { v } \phi } } _ { t } )$ and $\overline { { \boldsymbol { v } g } } _ { t } = \nabla \mathcal { L } ( \overline { { \boldsymbol { w } } } _ { t } )$ 

Proof. First, we calculate the difference between ${ \pmb w } _ { t + 1 }$ and $\overline { { \pmb { w } } } _ { t + 1 }$ , then we use µ-smooth condition to bound the difference between $\mathcal { L } ( \boldsymbol { w } _ { t + 1 } )$ and $\mathcal { L } ( \overline { { { \boldsymbol { w } } } } _ { t + 1 } )$ 

$$
\overline {{\boldsymbol {w}}} _ {t + 1} - \boldsymbol {w} _ {t + 1} = - \epsilon_ {t} (\overline {{\boldsymbol {u}}} _ {t} \circ \overline {{\boldsymbol {v}}} \phi_ {t} - \boldsymbol {u} _ {t}) = - \epsilon_ {t} (\overline {{\boldsymbol {u}}} _ {t} \circ \overline {{\boldsymbol {v}}} \phi_ {t} - \overline {{\boldsymbol {u}}} _ {t}) = \epsilon_ {t} \overline {{\boldsymbol {v}}} r _ {t}.
$$

Let us use µ-smooth condition to bound the L difference 

$$
\begin{array}{l} \mathcal {L} (\overline {{\boldsymbol {w}}} _ {t + 1}) - \mathcal {L} (\boldsymbol {w} _ {t + 1}) \\ \quad \leq \epsilon_ {t} \nabla \mathcal {L} (\boldsymbol {w} _ {t + 1}) ^ {\top} \overline {{\boldsymbol {v r}}} _ {t} + \frac {\mu}{2} \epsilon_ {t} ^ {2} \| \overline {{\boldsymbol {v r}}} _ {t} \| ^ {2} \\ \quad = \epsilon_ {t} \left(\nabla \mathcal {L} (\boldsymbol {w} _ {t}) + \nabla \mathcal {L} (\boldsymbol {w} _ {t + 1}) - \nabla \mathcal {L} (\boldsymbol {w} _ {t})\right) ^ {\top} \overline {{\boldsymbol {v r}}} _ {t} + \frac {\mu}{2} \epsilon_ {t} ^ {2} \| \overline {{\boldsymbol {v r}}} _ {t} \| ^ {2} \\ \quad = \epsilon_ {t} \nabla \mathcal {L} (\boldsymbol {w} _ {t}) ^ {\top} \overline {{\boldsymbol {v r}}} _ {t} + \epsilon_ {t} \left(\nabla \mathcal {L} (\boldsymbol {w} _ {t + 1}) - \nabla \mathcal {L} (\boldsymbol {w} _ {t})\right) ^ {\top} \overline {{\boldsymbol {v r}}} _ {t} + \frac {\mu}{2} \epsilon_ {t} ^ {2} \| \overline {{\boldsymbol {v r}}} _ {t}   \| ^ {2} \\ \quad \leq \epsilon_ {t} \nabla \mathcal {L} (\boldsymbol {w} _ {t}) ^ {\top} \overline {{\boldsymbol {v r}}} _ {t} + \epsilon_ {t} \| \nabla \mathcal {L} (\boldsymbol {w} _ {t + 1}) - \nabla \mathcal {L} (\boldsymbol {w} _ {t}) \|   \| \overline {{\boldsymbol {v r}}} _ {t} \| + \frac {\mu}{2} \epsilon_ {t} ^ {2} \| \overline {{\boldsymbol {v r}}} _ {t} \| ^ {2} \\ \quad = \epsilon_ {t} \overline {{\boldsymbol {v g}}} _ {t} ^ {\top} \overline {{\boldsymbol {v r}}} _ {t} + \epsilon_ {t} ^ {2} \mu   \| \overline {{\boldsymbol {v u}}} _ {t} \|   \| \overline {{\boldsymbol {v r}}} _ {t} \| + \frac {\mu}{2} \epsilon_ {t} ^ {2}   \| \overline {{\boldsymbol {v r}}} _ {t} \| ^ {2} \\ \quad / / \overline {{\boldsymbol {v g}}} _ {t} ^ {\top} \overline {{\boldsymbol {v r}}} _ {t} \leq 0. \\ = - \epsilon_ {t} \Delta (\overline {{\boldsymbol {u}}} _ {t} \circ \overline {{\boldsymbol {v g}}} _ {t}) + \epsilon_ {t} ^ {2} \mu   \| \overline {{\boldsymbol {v u}}} _ {t} \|   \| \overline {{\boldsymbol {v r}}} _ {t} \| + \frac {\mu}{2} \epsilon_ {t} ^ {2}   \| \overline {{\boldsymbol {v r}}} _ {t} \| ^ {2} \\ \leq 0.    / / B y t h e c h o i c e o f   \epsilon_ {t}. \end{array}
$$

Theorem A.3. Assume $\mathcal { L } ( w )$ is µ-smooth and differentiable, and the element-wise operator ϕ satisfies $x \cdot \phi ( x ) \geq 0$ as shown in Theorem 2.1. With $( \overline { { \pmb { w } } } _ { t } , \overline { { \pmb { s } } } _ { t } )$ following the update in (8) with constant step size ϵ: 

$$
\begin{array}{r l} & {\overline {{\boldsymbol {u}}} _ {t} = \boldsymbol {u} _ {t} (\overline {{\boldsymbol {w}}} _ {t}, \overline {{\boldsymbol {s}}} _ {t})} \\ & {\overline {{\boldsymbol {w}}} _ {t + 1} = \overline {{\boldsymbol {w}}} _ {t} - \epsilon \overline {{\boldsymbol {u}}} _ {t} \circ \overline {{\boldsymbol {v}}} \phi_ {t}} \\ & {\overline {{\boldsymbol {s}}} _ {t + 1} = \overline {{\boldsymbol {s}}} _ {t} + \boldsymbol {v} _ {t} (\overline {{\boldsymbol {w}}} _ {t}, \overline {{\boldsymbol {s}}} _ {t}).} \end{array}
$$

Assume $\epsilon > 0 ,$ we have: 

$$
\frac {1}{T} \sum_ {t = 1} ^ {T} \| \mathcal {L} (\overline {{\boldsymbol {w}}} _ {t}) \circ \overline {{\boldsymbol {u}}} _ {t} \| _ {\phi} \leq \frac {\mathcal {L} (\overline {{\boldsymbol {w}}} _ {1}) - \mathcal {L} (\overline {{\boldsymbol {w}}} ^ {*})}{T \epsilon} + \frac {\mu \epsilon}{2 T} B _ {T},
$$

where $\begin{array} { r } { B _ { T } = \sum _ { t = 1 } ^ { T } \left. \overline { { \pmb { u } } } _ { t } \right. ^ { 2 } , \overline { { \pmb { w } } } ^ { \ast } = a r g m i n _ { w } \mathcal { L } ( \pmb { w } ) } \end{array}$ , andfor notions, we write $\| \pmb { x } \| _ { \phi } = \pmb { x } ^ { \top } \phi ( \pmb { x } ) , \forall \pmb { x }$ 

Proof. Using the µ-smoothness of $\mathcal { L } ( w )$ , we expand $\mathcal { L } ( \overline { { \mathbf { w } } } _ { t + 1 } ) - \mathcal { L } ( \overline { { \mathbf { w } } } _ { t } )$ 

$$
\mathcal {L} (\overline {{\boldsymbol {w}}} _ {t + 1}) - \mathcal {L} (\overline {{\boldsymbol {w}}} _ {t}) \leq \nabla \mathcal {L} (\overline {{\boldsymbol {w}}} _ {t}) ^ {\top} (\overline {{\boldsymbol {w}}} _ {t + 1} - \overline {{\boldsymbol {w}}} _ {t}) + \frac {\mu}{2} \left\| \overline {{\boldsymbol {w}}} _ {t + 1} - \overline {{\boldsymbol {w}}} _ {t} \right\| ^ {2}.
$$

Substitute $\overline { { \pmb { w } } } _ { t + 1 } - \overline { { \pmb { w } } } _ { t } = - \epsilon \overline { { \pmb { u } } } _ { t } \circ \overline { { \pmb { v } \phi } } _ { t }$ to get: 

$$
\mathcal {L} (\overline {{\boldsymbol {w}}} _ {t + 1}) - \mathcal {L} (\overline {{\boldsymbol {w}}} _ {t}) \leq - \epsilon \cdot \nabla \mathcal {L} (\overline {{\boldsymbol {w}}} _ {t}) ^ {\top} \left(\overline {{\boldsymbol {u}}} _ {t} \circ \overline {{\boldsymbol {v} \phi_ {t}}}\right) + \frac {\mu \epsilon^ {2}}{2} \left\| \overline {{\boldsymbol {u}}} _ {t} \circ \overline {{\boldsymbol {v} \phi_ {t}}} \right\| ^ {2}.
$$

Simplify using the definition of $\left\| \cdot \right\| _ { \phi }$ and $\lVert \cdot \rVert _ { \Phi _ { t } }$ 

$$
\nabla \mathcal {L} (\overline {{\boldsymbol {w}}} _ {t}) ^ {\top} \left(\overline {{\boldsymbol {u}}} _ {t} \circ \overline {{\boldsymbol {v}}} \phi_ {t}\right) = \| \mathcal {L} (\overline {{\boldsymbol {w}}} _ {t}) \circ \overline {{\boldsymbol {u}}} _ {t} \| _ {\phi},
$$

which gives: 

$$
\mathcal {L} (\overline {{\boldsymbol {w}}} _ {t + 1}) - \mathcal {L} (\overline {{\boldsymbol {w}}} _ {t}) \leq - \epsilon \left(\| \mathcal {L} (\overline {{\boldsymbol {w}}} _ {t}) \circ \overline {{\boldsymbol {u}}} _ {t} \| _ {\phi}\right) + \frac {\mu \epsilon^ {2}}{2} \| \overline {{\boldsymbol {u}}} _ {t} \| ^ {2}.
$$

Summing over $t = 1 , \ldots , T ,$ , we obtain a telescoping sum: 

$$
\mathcal {L} (\overline {{\boldsymbol {w}}} _ {T + 1}) - \mathcal {L} (\overline {{\boldsymbol {w}}} _ {1}) \leq - \epsilon \sum_ {t = 1} ^ {T} \left(\| \mathcal {L} (\overline {{\boldsymbol {w}}} _ {t}) \circ \overline {{\boldsymbol {u}}} _ {t} \| _ {\phi}\right) + \frac {\mu \epsilon^ {2}}{2} \sum_ {t = 1} ^ {T} \| \overline {{\boldsymbol {u}}} _ {t} \| ^ {2}.
$$

Rearranging, dividing by $T \epsilon$ , and noting $\mathcal { L } ( W _ { T + 1 } ) \ge \mathcal { L } ( W ^ { * } )$ , we get: 

$$
\frac {1}{T} \sum_ {t = 1} ^ {T} \| \mathcal {L} (\overline {{\boldsymbol {w}}} _ {t}) \circ \overline {{\boldsymbol {u}}} _ {t} \| _ {\phi} \leq \frac {\mathcal {L} (\overline {{\boldsymbol {w}}} _ {1}) - \mathcal {L} (\overline {{\boldsymbol {w}}} ^ {*})}{T \epsilon} + \frac {\mu \epsilon}{2 T} B _ {T},
$$

where $\begin{array} { r } { B _ { T } = \sum _ { t = 1 } ^ { T } \Vert \overline { { \mathbf { u } } } _ { t } \Vert ^ { 2 } } \end{array}$ . This concludes the proof. 

Theorem A.4. Consider updates (7) and (8). Assuming $\mathcal { L } ( \cdot )$ is µ-smooth and the scaled step size $\epsilon _ { k } \alpha _ { k } \le \sigma$ , and consider thefollowing maskfunction: 

$$
\overline {{{{\boldsymbol {v} \phi}}}} _ {k} = \alpha_ {k} \mathbb {I} (\nabla \mathcal {L} (\overline {{{{\boldsymbol {w}}}}} _ {k}) \circ \overline {{{{\boldsymbol {v} u}}}} _ {k} \geq \frac {\mu \sigma}{2} \overline {{{{\boldsymbol {v} u}}}} _ {k} \circ \overline {{{{\boldsymbol {v} u}}}} _ {k}),
$$

we have 

$$
\mathcal {L} (\overline {{\boldsymbol {w}}} _ {k + 1}) \leq \mathcal {L} (\overline {{\boldsymbol {w}}} _ {k}).
$$

which holdsfor any step size $\epsilon _ { k } \geq 0$ 

Proof. By smoothness, we have 

$$
\begin{array}{l} \mathcal {L} (\overline {{\boldsymbol {w}}} _ {k + 1}) - \mathcal {L} (\overline {{\boldsymbol {w}}} _ {k}) \\ \qquad \leq \nabla \mathcal {L} (\overline {{\boldsymbol {w}}} _ {k}) ^ {\top} (\overline {{\boldsymbol {w}}} _ {k + 1} - \overline {{\boldsymbol {w}}} _ {k}) + \frac {\mu}{2} \| \overline {{\boldsymbol {w}}} _ {k + 1} - \overline {{\boldsymbol {w}}} _ {k} \| _ {2} ^ {2} \\ \qquad = - \epsilon_ {k} \nabla \mathcal {L} (\overline {{\boldsymbol {w}}} _ {k}) ^ {\top} (\overline {{\boldsymbol {v} \phi}} _ {k} \circ \overline {{\boldsymbol {u}}} _ {k}) + \frac {\mu \epsilon_ {k} ^ {2}}{2} \| \overline {{\boldsymbol {v} \phi}} _ {k} \circ \overline {{\boldsymbol {u}}} _ {k} \| _ {2} ^ {2} \\ \qquad = \epsilon_ {k} \alpha_ {k} \left(- \nabla \mathcal {L} (\overline {{\boldsymbol {w}}} _ {k}) ^ {\top} (\mathbb {I} (\nabla \mathcal {L} (\overline {{\boldsymbol {w}}} _ {k}) \circ \overline {{\boldsymbol {v u}}} _ {k} \geq \frac {\mu \sigma}{2} \overline {{\boldsymbol {v u}}} _ {k} \circ \overline {{\boldsymbol {v u}}} _ {k}) \circ \overline {{\boldsymbol {u}}} _ {k}) + \frac {\mu \epsilon_ {k} \alpha_ {k}}{2} \right. \| \mathbb {I} (\nabla \mathcal {L} (\overline {{\boldsymbol {w}}} _ {k}) \circ \overline {{\boldsymbol {v u}}} _ {k} \geq \frac {\mu \sigma}{2} \overline {{\boldsymbol {v u}}} _ {k} \circ \overline {{\boldsymbol {v u}}} _ {k}) \\ \qquad \leq \epsilon_ {k} \alpha_ {k} \left(- \nabla \mathcal {L} (\overline {{\boldsymbol {w}}} _ {k}) ^ {\top} (\mathbb {I} (\nabla \mathcal {L} (\overline {{\boldsymbol {w}}} _ {k}) \circ \overline {{\boldsymbol {v u}}} _ {k} \geq \frac {\mu \sigma}\mathrm{2} \overline {{\boldsymbol {v u}}} _ {k} \circ \overline {{\boldsymbol {v u}}} _ {k}) \circ \overline {{\boldsymbol {u}}} _ {k}) + \frac {\mu \sigma}{2} \| \mathbb {I} (\nabla \mathcal {L} (\overline {{\boldsymbol {w}}} _ {k}) \circ \overline {{\boldsymbol {v u}}} _ {k} \geq \frac {\mu \sigma}{2} \overline {{\boldsymbol {v u}}} _ {k} \circ \overline {{\boldsymbol {v u}}} _ {k}) \circ \overline {{\boldsymbol {\tau}}} _ {\mathrm{e}}\right) \\ \qquad \leq 0. \end{array}
$$

## A.5 INNER PRODUCT MASKS

Out of theoretical interest, let us consider a case of using an inner product mask: 

$$
\overline {{\boldsymbol {v} \phi}} _ {k} = \mathbb {I} (\overline {{\boldsymbol {u}}} _ {k} ^ {\top} \overline {{\boldsymbol {v}}} g _ {k} > 0), \qquad \boldsymbol {v} g _ {k} = \nabla \mathcal {L} (\overline {{\boldsymbol {w}}} _ {k}),
$$

which yields a scalar that applies to the entire update vector. In comparison, the element-wise mask in the paper treats each element separately. 

This case is interesting because, on convex functions, the cautious optimizers is always no worse than the base optimizers, regardless of the step size choices. 

Theorem A.5. Assume $\mathcal { L } ( \cdot )$ is convex, and $\overline { { \pmb { v } \phi } } _ { k } = \mathbb { I } ( \overline { { \pmb { u } } } _ { k } ^ { \top } \nabla \mathcal { L } ( \overline { { \pmb { w } } } _ { k } ) \geq 0 )$ . Then, starting from the same point $( { \pmb w } _ { k } , { \pmb s } _ { k } ) = ( { \overline { { \pmb w } } } _ { k } , { \overline { { \pmb s } } } _ { k } )$ , we have 

$$
\mathcal {L} (\overline {{\boldsymbol {w}}} _ {k + 1}) \leq \mathcal {L} (\boldsymbol {w} _ {k + 1}).
$$

which holds for any step size $\epsilon _ { k } \geq 0 .$ 

Proof. If $\overline { { \mathbf { u } } } _ { k } ^ { \top } \nabla \mathcal { L } ( \overline { { \mathbf { w } } } _ { k } ) > 0$ , we have $\overline { { v \phi } } _ { k } = 1$ , and $\pmb { w } _ { k + 1 } = \overline { { \pmb { w } } } _ { k + 1 }$ , and ${ \mathcal { L } } ( \overline { { \mathbf { w } } } _ { k + 1 } ) = { \mathcal { L } } ( \mathbf { w } _ { k + 1 } )$ 

If $\overline { { \mathbf { u } } } _ { k } ^ { \top } \nabla \mathcal { L } ( \overline { { \mathbf { w } } } _ { k } ) \leq 0$ , we have $\boldsymbol { v } \phi _ { k } = 0$ and $\overline { { \pmb { w } } } _ { k + 1 } = \pmb { w } _ { k }$ . By the convexity of ${ \mathcal { L } } ,$ we have 

$$
\begin{array}{c} \mathcal {L} (\boldsymbol {w} _ {k + 1}) - \mathcal {L} (\overline {{\boldsymbol {w}}} _ {k + 1}) = \mathcal {L} (\boldsymbol {w} _ {k} - \epsilon \boldsymbol {u} _ {k}) - \mathcal {L} (\boldsymbol {w} _ {k}) \\ \qquad \qquad \qquad \geq - \epsilon \boldsymbol {u} _ {k} ^ {\top} \nabla \mathcal {L} (\boldsymbol {w} _ {k}) > 0. \end{array}
$$

This proves the result. 

Corollary A.6. Consider the elementary testfunction: 

$$
\mathcal {L} (\boldsymbol {w}) = \frac {1}{2} \| \boldsymbol {a} \circ \boldsymbol {w} \| _ {2} ^ {2}.
$$

where $\pmb { a } \in \mathbb { R } ^ { d }$ is a non-zero vector. This is commonly used as the testbedfor optimization algorithms in theoretical analysis. 

Assume $\mathbf { \Delta } \mathbf { u } _ { k }$ and ${ \pmb v } _ { k }$ are element-wise mappings. We have ${ \mathcal { L } } ( \overline { { \mathbf { { w } } } } _ { k + 1 } ) \leq { \mathcal { L } } ( \mathbf { { w } } _ { k + 1 } )$ given $( { \pmb w } _ { k } , { \pmb s } _ { k } ) =$ $( \overline { { \boldsymbol { w } } } _ { k } , \overline { { \boldsymbol { s } } } _ { k } )$ , with either the inner product mask $\overline { { \pmb { v } \phi } } _ { k } = \mathbb { I } ( \overline { { \pmb { u } } } _ { k } ^ { \top } \nabla \mathcal { L } ( \overline { { \pmb { w } } } _ { k } ) \geq 0 )$ , or the element-wise mask $\overline { { \pmb { v } \phi } } _ { k } = \mathbb { I } ( \overline { { \pmb { u } } } _ { k } \circ \nabla \mathcal { L } ( \overline { { \pmb { w } } } _ { k } ) \geq 0 )$ 

Proof. The inner product case is implied by Theorem A.5. For element-wise mask, since the loss is an element-wise sum, and the update functions are element-wise mappings, which can apply Theorem A.5 on each element. 

This argument, of course, can be extended to general convex and separable loss functions of form $\begin{array} { r } { \mathcal { L } ( \mathbf { \boldsymbol { w } } ) \equiv \sum _ { i } \mathcal { L } _ { i } ( w _ { i } ) } \end{array}$ . □ 

Theorem A.7. Consider update (8). Assuming $\mathcal { L } ( \cdot )$ is µ-smooth, and consider thefollowing mask function: 

$$
\overline {{\boldsymbol {v} \phi}} _ {k} = \alpha_ {k} \mathbb {I} (\nabla \mathcal {L} (\overline {{\boldsymbol {w}}} _ {k}) ^ {\top} \overline {{\boldsymbol {v} u}} _ {k} \geq \frac {\alpha_ {k} \mu \epsilon_ {k}}{2} \| \overline {{\boldsymbol {v} u}} _ {k} \| ^ {2}),
$$

where $\alpha _ { k }$ is a scaling factor. 

We have 

$$
\mathcal {L} (\overline {{\boldsymbol {w}}} _ {k + 1}) \leq \mathcal {L} (\overline {{\boldsymbol {w}}} _ {k}).
$$

which holds for any step size $\epsilon _ { k } \geq 0 .$ 

Ablation on ϕ: To analyze the impact of cautious masking, we conduct ablation studies on different choices of $\phi .$ Our results show that the performance is not sensitive to the specific choice of $\phi ,$ as all cautious variants incorporating these functions consistently outperform GDM. This underscores the robustness and effectiveness of cautious masking, as shown in Figure 3. 

The tested $\phi$ functions are defined as follows: 

$$
\begin{array}{l} \phi_ {c} = \mathbb {I} (x > 0) - c \mathbb {I} (x \leq 0), \\ \phi_ {\text {inner}} = \mathbb {I} (x ^ {\top} y \geq 0). \end{array}\tag{13}
$$

These results highlight that cautious masking enhances optimization stability and performance across diverse settings. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/cf6503dd-a7d4-4104-9ea0-a4d743d42f4b/d33ea9f63a46be17d17678463c44b38ad2e10cb3ccc3dee608436edfcd997b5e.jpg)



Figure 3: This plot presents an ablation study evaluating the performance of various $\phi$ configurations. We compare the loss curves for different ϕ choices. Across all tested configurations, the cautious GDM variants, $\phi _ { c }$ with $c = 0 . 0 1 , 0 . 1$ , and $\phi _ { \mathrm { i n n e r } }$ as defined in (13), consistently outperform standard GDM with optimal hyper-parameter configration (Goh, 2017).


## A.6 PESUDO CODE

See Algorithm 3 for the Cautious variant of the Lion optimizer. 

Algorithm 3 C-Lion Optimizer
Require: learning rate $\epsilon$ , momentum coefficient $\beta_1, \beta_2 \in [0,1)$ , weight decay factor $\gamma$ 1: Initialize parameter vector $w_t$ 2: Initialize $t = 0$ , $m_0 = 0$ 3: while $w_t$ not converged do
4: $t \leftarrow t + 1$ 5: $g_t \leftarrow \nabla_w \mathcal{L}_t(w_{t-1})$ {Get gradients at timestep $t$ }
6: $u_t \leftarrow \text{sign}(\beta_1 m_{t-1} + (1 - \beta_1) \cdot g_t)$ {get the signed update}
7: $m_t \leftarrow \beta_2 m_{t-1} + (1 - \beta_2) \cdot g_t$ {update momentum}
8: $\phi_t \leftarrow \mathbb{I}(u_t \circ g_t \geq 0)$ // Compute alignment mask
9: $\overline{\epsilon}_t = \epsilon_t \frac{d}{\|\phi_t\|_0 + 1}$ // Scale lr, $d$ is dimension of $\phi_t$ 10: $w_t \leftarrow w_{t-1} - \overline{\epsilon}_t \phi_t \circ u_t$ // Masked update
11: $w_t \leftarrow w_t - \epsilon \gamma w_t$ {Weight decay}
12: end while 

## B CAUTIOUS MASK STUDY

Figure 4 shows the ratio of active dimensions $r _ { k } = \mathrm { n n z } ( \overline { { \pmb { v } u } } _ { k } \circ \overline { { \pmb { v } g } } _ { k } > 0 ) / \mathsf { d i m } ( \overline { { \pmb { v } u } } _ { k } \circ \overline { { \pmb { v } g } } _ { k } )$ across the training iteration k on LLaMA 100M. It see that it decreases from 1 and stabilizes around 0.55. This suggests that the scaling factor generally lies within the range [1, 1.55]. The stable ratio does not change significantly across different models. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/cf6503dd-a7d4-4104-9ea0-a4d743d42f4b/402d198c6773bc47363a3ba2d8d6089058812b0124d57ccc1460fc965014bb32.jpg)


$$
r (\mathbf {x}) = \frac {\mathrm{nnz} (\mathbf {x} > 0)}{\dim (\mathbf {x})}
$$

## C ADDITIONAL DETAILS ON LLM SCALING

The training sequence length is set to be 4096 and we perform coordinate-descent over discrete grids for all optimizer hyperparameters (lr, weight decay, warmup, $\beta _ { 1 } , \beta _ { 2 } , \epsilon ,$ max-grad-norm, batch size). The following tables 5, 6 show the optimal set of parameters for each scale. 

## D ADDITIONAL DETAILS ON MINI-IMAGENET

Here you can find the hyperparameter search of the Mini-ImageNet experiments. The findings are consistent with our LLM experiments. 

<table><tr><td rowspan="2">Hyperparameter</td><td colspan="4">AdamW</td><td colspan="4">C-AdamW</td></tr><tr><td>130M</td><td>300M</td><td>520M</td><td>1.2B</td><td>130M</td><td>300M</td><td>520M</td><td>1.2B</td></tr><tr><td>Learning rate (lr)</td><td>0.008</td><td>0.008</td><td>0.004</td><td>0.002</td><td>0.008</td><td>0.008</td><td>0.008</td><td>0.006</td></tr><tr><td>Weight decay (wd)</td><td>0.1</td><td>0.1</td><td>0.2</td><td>0.2</td><td>0.1</td><td>0.1</td><td>0.1</td><td>0.1</td></tr><tr><td>Warmup steps</td><td>2000</td><td>2000</td><td>1000</td><td>1000</td><td>2000</td><td>2000</td><td>2000</td><td>2000</td></tr><tr><td><eq>\beta_1</eq></td><td>0.9</td><td>0.9</td><td>0.9</td><td>0.9</td><td>0.95</td><td>0.98</td><td>0.98</td><td>0.98</td></tr><tr><td><eq>\beta_2</eq></td><td>0.98</td><td>0.98</td><td>0.98</td><td>0.98</td><td>0.98</td><td>0.98</td><td>0.98</td><td>0.98</td></tr><tr><td>ε</td><td>1e-20</td><td>1e-10</td><td>1e-10</td><td>1e-05</td><td>1e-15</td><td>1e-25</td><td>1e-25</td><td>1e-16</td></tr><tr><td>Max-grad-norm</td><td>1</td><td>1</td><td>1</td><td>2</td><td>1</td><td>2</td><td>1</td><td>1</td></tr><tr><td>Batch size</td><td>128</td><td>128</td><td>256</td><td>256</td><td>128</td><td>128</td><td>256</td><td>256</td></tr></table>


Table 5: Optimal hyperparameters (from coordinate-descent over discrete grids) for AdamW and C-AdamW across model scales.


<table><tr><td>Hyperparameter</td><td>Range</td></tr><tr><td>Learning rate (lr)</td><td>{0.001, 0.002, 0.004, 0.006, 0.008, 0.010, 0.012, 0.014, 0.016, 0.018, 0.020}</td></tr><tr><td>Weight decay (wd)</td><td>{0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4}</td></tr><tr><td>Warmup steps</td><td>{500, 1000, 2000, 4000, 6000}</td></tr><tr><td><eq>\beta_1</eq></td><td>{0.85, 0.90, 0.92, 0.95, 0.98, 0.99}</td></tr><tr><td><eq>\beta_2</eq></td><td>{0.96, 0.98, 0.99, 0.995, 0.9995}</td></tr><tr><td><eq>\epsilon</eq></td><td>{1e-30, 1e-25, 1e-20, 1e-15, 1e-10, 1e-5}</td></tr><tr><td>Max-grad-norm</td><td>{0.5, 1, 2, 4}</td></tr><tr><td>Batch size</td><td>{32, 64, 128, 256, 512}</td></tr></table>


Table 6: Discrete grid for hyperparameter search


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/cf6503dd-a7d4-4104-9ea0-a4d743d42f4b/7590fe28d76b30e2eae92487da5581618ee7920ff01e4514f1eb251cd05ea2e0.jpg)



Figure 5: AdamW/C-AdamW learning rate search on Mini-ImageNet


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/cf6503dd-a7d4-4104-9ea0-a4d743d42f4b/3028907fa0492b88b8b9adb6dd2e973d42f247c396488a0310c8110986d9a4d9.jpg)



Figure 6: LaProp/C-LaProp learning rate search on Mini-ImageNet


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/cf6503dd-a7d4-4104-9ea0-a4d743d42f4b/290f3cac30d0649b08275f971c6ea0b781cc953001e3ec3c95f752613cd222a0.jpg)



Figure 7: MARS/C-MARS learning rate search on Mini-ImageNet


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/cf6503dd-a7d4-4104-9ea0-a4d743d42f4b/a1282c8a7f0addcfaf2dd4a4b07a7d276b858ba900df4fb5dbb691cfd8576f77.jpg)



Figure 8: This experiment follows the same setup as the previous 100M Llama experiment in 1


## E ADDITIONAL EXPERIMENTS

## E.1 LEARNING RATE SCHEDULER ABLATION

We notice a concurrent work (Bergsma et al., 2025) suggests that reducing learning rate straight to zero with linear decay is beneficial. To preliminarily validate the robustness of cautious optimizer, we took the 100M model setup in our LLM pretraining experiment and swap in the new scheduler. We found cautious optimizers stably outperform baselines. 

## E.2 BATCH SIZE ABLATION ON 60M

We take the 60M parameter model and perform a sweep on Batch Sizes and train for 1.2 billion tokens (1× Chinchilla) on C4 and we are reporting perplexity (lower the better). 

<table><tr><td>Batch Size (tokens)</td><td>24K</td><td>120K</td><td>600K</td></tr><tr><td>AdamW</td><td>38.5</td><td>37.2</td><td>41.7</td></tr><tr><td>C-AdamW</td><td>37.2</td><td>36.2</td><td>40.6</td></tr></table>


Table 7: Perplexity of AdamW vs. C-AdamW across different batch sizes.


## E.3 MUON (JORDAN ET AL., 2024)

We tested a preliminary version of C-Muon (Fig 9) and found that results encouraging as follows. Although at the time of submission we are not able to verify it at larger scale, it could potentially be a very interesting future direction. 

## E.4 POST-TRAINING LLM

To further test cautious optimizers in language modeling tasks, we perform two post-training experiments. First, we instruction-tune Qwen2.5-1.5B-Instruct (Yang et al., 2024) on the PowerInfer/QWQ LONGCOT-500K (PowerInfer, 2024), a dataset focusing on enhancing the model’s reasoning abilities by data distilled from QwQ-32B-Preview (Qwen, 2024). Then we follow the experiment setting in (Huang et al., 2024) to perform RLHF alignment with PPO (Schulman et al., 2017) on EleutherAI/pythia-1b-deduped. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/cf6503dd-a7d4-4104-9ea0-a4d743d42f4b/1439f3948da3ccc57dda073985ed93340d32a9109dc8f284e24cf9407d98ee8d.jpg)



Figure 9: Following the same 100M model training setup in 1, we also preliminarily test the latest popular optimizer Muon Jordan et al. (2024) and find cautious improves upon the baseline.


Observation: In both experiments, we find under same training steps and PPO episodes, cautious optimizers obtain lower training loss as well as higher rewards. This indicates potential of our proposed method beyond pretraining tasks. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/cf6503dd-a7d4-4104-9ea0-a4d743d42f4b/33db37bd25160a20d055e4451bf403c09f1cbfb6b6932808b5f04f9c58d07e73.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/cf6503dd-a7d4-4104-9ea0-a4d743d42f4b/b5e40574af5bec8ff7d339d6600f6eaba0b717f92bb2b442ce3cec356cf2cf8c.jpg)


Figure 10: (a) shows training loss on the task of instruction finetuing Qwen2.5-1.5B-instruct on the distilled dataset from QwQ for 500K question and answer pairs. (b) shows RLHF reward with PPO for 1 million episodes. In both cases, C-AdamW outperforms AdamW. 

## E.5 PRETRAINING MASKED AUTOENCODERS (MAES)

Masked Autoencoders (MAEs) (He et al., 2022) have emerged as a powerful approach for pretraining Vision Transformers (ViTs) (Dosovitskiy, 2020) on large-scale datasets like ImageNet-1K (Russakovsky et al., 2015). This task involves reconstructing 75% of randomly masked image patches, a challenging objective that requires extensive training over hundreds of epochs and millions of images. The primary goal is to learn robust visual representations that are generalizable across downstream vision tasks. The quality of these representations is typically measured by the final evaluation loss, which reflects how accurately the model reconstructs masked test images; lower evaluation loss indicates higher-quality representations. The results of our experiments are summarized in Figure 11, where we compare the performance of the cautious optimizer against the AdamW baseline. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/cf6503dd-a7d4-4104-9ea0-a4d743d42f4b/4b283abadddbc88297a7276717f7ccea918af40a31e97dbb3c3445d7b69400ab.jpg)


Observation: From figure 11, we observe that the cautious optimizer achieves lower evaluation loss faster compared to AdamW. This result highlights the effectiveness of the cautious approach in improving the precision of reconstruction and, consequently, the quality of the learned visual representations. 

<table><tr><td># Params</td><td><eq>\beta_1</eq></td><td><eq>\beta_2</eq></td><td>Learning rate</td><td>weight decay</td><td>Batch Size</td></tr><tr><td>110 M</td><td>0.9</td><td>0.999</td><td><eq>1.5\times 10^{-4}</eq></td><td>0.05</td><td>4096</td></tr></table>


Table 8: Hyperparameters for MAE experiment. This experiment follows the optimal setup provided in He et al. (2022)


## F MISCELLANEOUS

The masking and scaling would incur some additional costs. However, in practice we observe that the impact is minimum to overall throughput in the Distributed Data Parallel setting. For our 100M model runs on 16 GPUs, AdamW has token throughput of 579383 token/s, whereas C-AdamW has 551483 tokens/s, which is around 3% difference in training throughput. Note that this is comparing our naive cautious implementation against the fused pytorch implementation. 

As for other distributed training setting, such as tensor parallel, masking operation is element-wise hence not communication-bound, whereas scaling would require global statistics. However, the communication is also minimum, since only a single floating-point number (the local mean) needs to be all-gathered by other workers. Given modern GPU bandwidth, this should incur only minor overhead. 


Table 9: Training efficiency comparison between AdamW and C-AdamW.


<table><tr><td>Method</td><td>Wall-Clock (h)</td><td>Throughput (token/s)</td></tr><tr><td>AdamW</td><td>10.547</td><td>571,839.76</td></tr><tr><td>C-AdamW</td><td>10.567</td><td>560,488.63</td></tr></table>