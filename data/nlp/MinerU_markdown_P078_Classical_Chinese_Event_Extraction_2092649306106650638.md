# Revisiting Classical Chinese Event Extraction with Ancient Literature Information

Xiaoyi Bao<sup>1</sup>, Zhongqing Wang<sup>2</sup>∗, Jinghang Gu<sup>1</sup>∗, and Chu-Ren Huang<sup>1</sup> <sup>1</sup>The Hong Kong Polytechnic University, Hong Kong, China <sup>2</sup>Natural Language Processing Lab, Soochow University, Suzhou, China p2213545413@outlook.com wangzq@suda.edu.cn {jinghang.gu,churen.huang}@polyu.edu.hk 

## Abstract

The research on classical Chinese event extraction trends to directly graft the complex modeling from English or modern Chinese works, neglecting the utilization of the unique characteristic of this language. We argue that, compared with grafting the sophisticated methods from other languages, focusing on classical Chinese’s inimitable source of Ancient Literature could provide us with extra and comprehensive semantics in event extraction. Motivated by this, we propose a Literary Vision-Language Model (VLM) for classical Chinese event extraction, integrating with literature annotations, historical background and character glyph to capture the inner- and outer-context information from the sequence. Extensive experiments build a new state-of-the-art performance in the GuwenEE, CHED datasets, which underscores the effectiveness of our proposed VLM, and more importantly, these unique features can be obtained precisely at nearly zero cost. Our code is publicly available at https://github.com/HoraceXIaoyiBao/ ACL25-CCEE. 

## 1 Introduction

Classical Chinese, as a written form of the Chi-nese, had been widely used in recording thousandsyears of history, culture and science, being a im-portant carrier of digital humanities. Many studieshave endeavored to the fundamental tasks such asPoS tagging (Tian and Guo, 2022) and segmenta-tion (Huang et al., 2010), as well as downstreamtasks like poetry generation (Zhang and Lapata,2014) and translation (Wang et al., 2023a) in clas-sical Chinese. However, only a few works fo-cus on event extraction, it aims to extract eventsfrom the literature, each of which consists of fourtypes of elements: a trigger and multiple argu-ments are the raw spans in the input text, and theywill be classified into a corresponding event typeor role type. For instance, an event of war inFigure 1: a war event triggered by “引师” (com-mand), the Person argument is “高祖” (emperorgaozu), Weather argument is “雨霁” (stop of rain),Place argument is “霍邑” (huoyi). Event extrac-tion could have a significant contribution for ana-lyzing valuable historical literature, facilitating thepreservation and inheritance of historical and cul-tural heritage.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/c0f0cf6c-021e-4bc6-9c3a-ac6f9fea297e/6acb5c06fd3855d5a56f9ded519a1940598a935d35192604256e9f5c5cc9373e.jpg)



Figure 1: Example of how ancient literature help classical Chinese event extraction.


However, the majority of previous studies tend to directly implement techniques from English (Ji et al., 2021; Tang et al., 2024; Lin et al., 2020; Liu et al., 2021) or modern Chinese (Congcong et al., 2023; Wang et al., 2023b) on classical Chinese datasets. Despite their effectiveness, previous works’ modelings are not unique to classical Chinese: either phonetic symbols (Huang et al., 2010), syntax structures (Wang et al., 2023b) or punctuation (Wang and Li, 2024) are also inclusive to English or modern Chinese, making their systems sub-optimal for classical Chinese. 

In this study, we shift our attention to the long-existing yet often neglected characteristic of classi-cal Chinese data: They are coming from AncientLiterature. Behind which enriched resources canbe utilized to enhance extraction of events suchas historical background, annotation (注解) andthe glyph they are written with. We first illustratethe value of annotation from literature in Figure 1b), where the “趣”(interesting) could be misunder-stood but easily corrected into “趋” (go) with theprecise annotation and the following “霍邑” ob-viously should be a Place. The contribution ofliterature background is shown with the case “高祖”(gaozu) in Figure 1 c), which is not a commonword in modern Chinese but could be interruptedas “唐高祖李渊”(Yuan Li) if we know the samplecomes from “旧唐书” (book of Tang) and “高祖”should be a Person. The final is the glyph of Chi-nese they are written in the literature. To illustrate,the shape of the “雨” (rain) in “雨霁”(stop of rain)directly evolved from the silhouette of shape ofrain and exist as a shared radical in “霁” as shownin Figure 1 a). This intuitive glyphic representa-tion facilitates the straightforward extraction andclassification of “雨霁” as a Weather.

However, it is challenging to incorporate ancient literature into event extraction. The difficulties are threefold: The first is, which aspects of information should we grasp from the ancient literature as it is unclear how literature information impacts the extraction. Secondly, how can we precisely obtain the literature information as the samples could be distributed from a hundred books? Finally, even with current powerful large language models (LLMs), modeling the obtained information is complicate due to their varied forms, which can be continuous paragraphs (background), discrete dictionaries (annotation), or even visual images (glyph), not only across text forms, but even across modalities. 

In this study, we tackle the above challenges with proposed Literary Vision-Language Model for classical Chinese event extraction. We first show the three aspects of background, annotation and glyph from ancient literature are the unique features that could greatly contribute to the event extraction in Section 3, where we further introduce the innovative methodologies for precisely gasping the three aspects from ancient literature in the corresponding subsections. We finally modeling the literature information with fusion instruction and visual bridging in the proposed Literary Vision-Language Model (VLM) as shown in Figure 5, which can cover both the continuous and discrete texts along with the visual glyphic image, deciphering the interplay across different aspects. 

The detailed evaluation shows that our proposed model significantly advances the SOTA performance on several benchmarks, validating the value of the literature information in downstream classical Chinese tasks. To the best of our knowledge, we are the first to use the unique source of ancient literature in classical Chinese works. 

## 2 Related Works

## 2.1 Event Extraction

Event extraction works have indeed leveraged modeling methods from diverse perspectives, from the sequence tagging (Chen et al., 2015; Wang et al., 2019b; Sha et al., 2016; Lin et al., 2020; Fan and He, 2023), structure and graph (Cui et al., 2020a; Yang et al., 2023; Lin et al., 2020; Liu et al., 2023) to multi-modalities(Li et al., 2023b,a; Nguyen et al., 2023; Majumder et al., 2020; Zhang et al., 2024a). Recent trends have shifted towards harnessing the power of LLMs to generate the target sequence (Lu et al., 2021; Liu et al., 2023; Yang et al., 2023; Zhang et al., 2024b). 

Besides, the works on Chinese event extraction tend to tailor their methods specifically to the unique characteristics of the Chinese language (Chen and Ji, 2009; Li and Zhou, 2012; Li et al., 2012; Ding et al., 2019), For instance, Xu et al. (2020) addressed the issue of overlapping roles, while Shen et al. (2020) introduced hierarchical event features. Separately, Lin et al. (2018) approached with a hybrid representation for each character. 

## 2.2 Classical Chinese Event Extraction

The majority of research on classical Chinese invariably implement English techniques on classical Chinese datasets directly (Wang et al., 2023b; Ji et al., 2021; Congcong et al., 2023; Tang et al., 2024; Wang et al., 2023c). Among them, Congcong et al. (2023) implements the sequence tagging with pre-trained Guwen-BERT while Tang et al. (2024) showcase the weak performance of several LLMs on classical Chinese task. Some works further design the model based on the features of Chinese (Wang et al., 2023b). For instance, Huang et al. (2010) adopt the phonetic symbols, Wang et al. (2023b) parse the text for syntax structures and Wang and Li (2024) employ the punctuation. However, these modelings could also be applied on modern Chinese, are not exclusive and optimal to classical Chinese. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/c0f0cf6c-021e-4bc6-9c3a-ac6f9fea297e/09795fcceb1e65bc39817b9e28feb020b976b2996ae2fa0dddf5a37f016c829e.jpg)



Figure 2: Illustration of ancient literature information.


Different from previous studies, our method stands out as the first to argue for the proprietary modeling for classical Chinese tasks. We introduce an innovative approach that manipulates ancient literature, the unique origin of classical Chinese, with the literary vision-language models specifically accommodating it. 

## 3 Value and Extraction of Ancient Literature Information

In this secton, We showcase how the ancient literature can help improve the task from the following three aspects along with the extraction of them. 

## Backgrounds

Backgrounds is a feature that provides essential macro background knowledge for a specific sample as shown in Figure 2. Specifically, the backgrounds information contains 3 types: 

• Literature source has been used for providing the most direct and structured knowledge for a sample (Kasanishi et al., 2023; Naik et al., 2022). We however is the first to stand for it’s works in classical Chinese works, to illustrate： 

“遂听信计，部署诸将所击”

(Took Xin’s advice, attacked with the generals) 

where “信” could be easily interpreted into “采纳/听从” (take) with a wrong segmentation of“听信” (take), but could be corrected into “韩信” (Xin Han, exists as the Person argument) ifwe know the sample comes from “史记·淮阴侯列传” and the protagonist of it is “韩信”.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/c0f0cf6c-021e-4bc6-9c3a-ac6f9fea297e/ab8098e738b358fb9bd907659cc778138213363960ee122baf0be694d6357dcc.jpg)



Figure 3: Process of backgrounds construction.


• Historical background can paint a somehowmuch larger image for us and give us extra infor-mation in a more indirect way such as the histor-ical background “唐朝” (Tang Dynasty) of thegiven example in Figure 2. By which we canidenify the “高祖” (Gaozu) as “唐高祖李渊”(Tang Gaozu Yuan Li) otherwise it could be mis-understood as “汉高祖刘邦” (Han Gaozu BangLiu) or “魏高祖曹丕” (Wei Gaozu Pi Cao).

• Biographical protagonist contains the identityof the protagonist, has been proofed (Ruppen-hofer et al., 2020; Palmero Aprosio and Tonelli,2015) that can help us effectively differentiatebetween people with common appellations asmost of the ancient literature are wrote in bio-graphical formation (纪传体). For instance:

“而上又上泰山，有秘祠其颠”

(The emperor climbed the mountain Tai again, held a secret rituals at the peak.) 

the first “上” (emperor, exists as Person) couldbe garbled with the second “上” (climb). How-ever, if been awared of the protagonist of thesample is “汉武帝” (emperor wu of han), it willbe easier for us to distinguish the two “上”.

As the value of background has been illustrated,we subsequently proceed to obtain it. Backgroundintrinsically is a cluster of information that form asa tree rooted at the specific literature source, withvarious aspects derived from it. We thus grasp thebackground information with proposed pipelinestarting at extracting the source of the sample asillustrated in the Figure 3. The unprocessed sam-ple is initially searched in Chinese Text Project (中国哲学书电子化计划)<sup>1</sup> to identify the specific an-cient literary source accurately. Once this sourceis identified, it is used to gather additional relevantbackground details from Wikipedia. These detailsare then converted into actionable instructions inthe following section.

## Annotation

Annotation (注解) refers to the contemporary Chi-nese interpretations crafted by expert historiansfor ancient literary works. Leveraging annotationscan enable us to connect classical Chinese taskwith modern Chinese. There are 3 types included:

• Semantic variation (古今异义) is an signifi-cant challenge for downstream tasks (Perroneet al., 2019). We thus annotates the wordswhose semantic meaning has been changedwhen compared with the time of writing the lit-erature. For instance:

## “予除右丞相兼枢密使”

(I was appointed as the Right Chancellor and Minister of the Imperial Secretariat.) 

the meaning of the word “除” (appoint, exists asthe event trigger) is different from its meaningof remove in modern Chinese.

• Phonetic loan character (通假字) representscharacters which have been formed from “bor-rowed” characters of homophonous (or near-homophonous) morphemes (Jiang and Ren,2023) such as the “趣” and “趋” in Figure 2.

• Human name is more intuitive when compared with the previous two, explaining the specific part of the sample as the human name to prevent misunderstanding since the sample is picked and cut out of context. For instance: 

“尚悲感，发病恍惚”

(Shang felt sorrowful, with dizziness.) 

“尚” could be wrongly interpreted as still but itactually refers to “夏侯尚” (Shang Xiahou) asa Person argument.

For extraction, the annotations are more dis-crete when compared with the backgrounds, theirsemantic meanings are paired up with the corre-sponding words. We thus extract the glyph followthe manner of dictionary. Initially, the input sen-tence undergoes segmentation with the pre-trainedclassical Chinese tokenizer Jiayan<sup>2</sup>. Subsequently,the tokenized tokens are retrieved from the An-cient Chinese Dictionary (古代汉语词典) to pro-cure their modern Chinese interpretations.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/c0f0cf6c-021e-4bc6-9c3a-ac6f9fea297e/7ed007803ed87a5af0369651fc2c42c43128b14feeaeb4b5c4aba041dc9b44d0.jpg)



Figure 4: Construction of glyphic image.


## Glyph

As a hieroglyphic language, glyph of charactersthat the literature are written in could contains en-riched semantic information, which could be illus-trated from two aspects. The first is the shapeglyph, such as the shape of the character “山”,“雨”and “门” in Figure 2, directly evolved from theactual silhouette of the mountain, rain and door re-spectively. This intuitive glyphic representation fa-cilitates the straightforward semantic understand-ing of them. Furthermore, there are also existconnections in radical glyph plays a critical rolein communicating the semantic essence of charac-ters such as the shared radical of “雨” (rain) be-tween “雷”(thunder) and “霆” (lightning), the “讠”(speech) between “讲” (speak) and “辩” (argue).

However, obtaining glyph is different from theprevious two information, it comes from visualmodality. Previous works’ splitting into radicalsor characters involves the affordable retraining ofLLMs. We thus adopt the sentence-level glyph im-age (Bao et al., 2024) to capture the glyph informa-tion. Specifically, given a sentence, each characteris transformed to an image of size p q with a spe-cific font Song (宋体) as shown in Figure 4. Then,a sentence containing N characters is constructedin a image where the characters are concatenatedsequentially follow the writing order of from leftto right and starting a new line below.

Considering there are also exist connections inglyphic level for annotations such as the sharedradical “辶” in “趣” and “趋” in Figure 2 d),we further build connections between classical in-put and modern annotation in the glyphic image.Specifically，we follow the layout of explanatorybook (注 解 本), placing each annotation under-neath its corresponding word as an active visualbridging between classical and modern Chinese.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/c0f0cf6c-021e-4bc6-9c3a-ac6f9fea297e/c8d2b0eae9c0ad7abbf6ca04148e831f1f97960b42a55ac1dbb28a64667ca5eb.jpg)



Figure 5: The illustration of our Literary Vision-Language Model.


## 4 Classical Chinese Event Extraction via Literary Vision-Language Model

Even if the information is obtained, how to modeling them is another challenge due to their various forms. In this study, we propose Literary Vision-Language Model as shown in Figure 5. Specifically, we propose a fusion instruction to combine all the above information, which contains mixed visual and textual contents. We will first introduce the modeling of the visual glyph and then cover the rest with fusion instruction. 

## 4.1 Modeling of Character Glyph

Vision Transformer (ViT) is employed as our image encoder for extraction glyphic information, which is excels at distilling high-level visual features from raw images. We adopt the Sequence Order Alignment (Bao et al., 2024) method to align with the textual writing order of raw sentence. 

Specifically, the input image is divided into patches, and embedded into visual tokens with position encoding. The grids are then flatten into a sequence that align with the textual tokens in the inputted review as shown in Figure 5. The patch 1 will be placed at the start of the flattened sequence, followed by the adjacent patch. Once reaching the end of a line, it will continue in a zigzag pattern, moving to the rightmost patch in the line below. With the alignment method, the visual tokens are in the same order with the textual tokens, which then augmented with positional encodings to obtain the final image representations $x _ { v }$ 

## 4.2 Fusion Instruction

We specifically design the instructions for modeling the rest background and annotation, which also responsible for guiding the model to fuse the textual contents with visual representations $x _ { v }$ obtained. The fusion instruction are designed as shown in Figure $5 ,$ which include a guiding instruction $x _ { t \_ b e f o r e }$ and $\boldsymbol { x } _ { t \_ a f t e r }$ at both before and after the visual tokens, the instructions $x _ { t \_ b e f o r e }$ before it covers the extracted background information while the $\boldsymbol { x } _ { t \_ a f t e r }$ after covers the annotations along with the specific sentence for extracting. 

We subsequently fuse the obtained the vision encoder’s output as visual tokens $x _ { v }$ and the tokenized text as textual tokens $x _ { t \_ b e f o r e }$ and $\boldsymbol { x } _ { t \_ a f t e r }$ for LLM’s processing. These tokens are merged to create the input x: 

$$
x = [ x _ {t \_ b e f o r e}, x _ {v}, x _ {t \_ a f t e r} ]\tag{1}
$$

Given the fused sequence $x = x _ { 1 } , . . . , x _ { \left| x \right| }$ as input, The decoder predicts token-by-token. At the i-th step of generation, the decoder predicts the $i -$ th token $y _ { i }$ and decoder state $h _ { i } ^ { d }$ as: 

$$
y _ {i}, h _ {i} ^ {d} = ([ h _ {1} ^ {d}, \dots , h _ {i - 1} ^ {d} ], y _ {i - 1})\tag{2}
$$

The conditional probability of the whole output sequence $p ( y | x )$ is progressively combined by the probability of each step $p ( y _ { i } | y _ { < i } , x )$ 

$$
p (y | x) = \prod_ {i = 1} ^ {| y |} p (y _ {i} | y _ {<   i}, x)\tag{3}
$$

where $y _ { < i } = y _ { 1 } . . . y _ { i - 1 }$ , and $p ( y _ { i } | y _ { < i } , x )$ are the probabilities over target vocabulary V. 

<table><tr><td rowspan="3">Method</td><td colspan="6">GuwenEE</td><td colspan="6">CHED</td></tr><tr><td colspan="3">Tri-I</td><td colspan="3">Tri-C</td><td colspan="3">Tri-I</td><td colspan="3">Tri-C</td></tr><tr><td>P.</td><td>R.</td><td>F1.</td><td>P.</td><td>R.</td><td>F1.</td><td>P.</td><td>R.</td><td>F1.</td><td>P.</td><td>R.</td><td>F1.</td></tr><tr><td>Chinese-Bert-CRF</td><td>0.427</td><td>0.471</td><td>0.447</td><td>0.304</td><td>0.342</td><td>0.321</td><td>0.736</td><td>0.699</td><td>0.717</td><td>0.701</td><td>0.665</td><td>0.682</td></tr><tr><td>Guwen-Bert-CRF</td><td>0.476</td><td>0.507</td><td>0.490</td><td>0.368</td><td>0.399</td><td>0.382</td><td>0.752</td><td>0.713</td><td>0.731</td><td>0.733</td><td>0.709</td><td>0.719</td></tr><tr><td>DMBert</td><td>0.498</td><td>0.529</td><td>0.512</td><td>0.391</td><td>0.415</td><td>0.402</td><td>0.771</td><td>0.724</td><td>0.746</td><td>0.740</td><td>0.715</td><td>0.727</td></tr><tr><td>Mengzi-T5-base</td><td>0.551</td><td>0.512</td><td>0.531</td><td>0.432</td><td>0.403</td><td>0.418</td><td>0.783</td><td>0.735</td><td>0.758</td><td>0.751</td><td>0.726</td><td>0.738</td></tr><tr><td>LLaMA-3-8B</td><td>0.526</td><td>0.520</td><td>0.523</td><td>0.411</td><td>0.409</td><td>0.410</td><td>0.769</td><td>0.707</td><td>0.736</td><td>0.742</td><td>0.693</td><td>0.716</td></tr><tr><td>LLaMA-3-Chinese-8B</td><td>0.537</td><td>0.534</td><td>0.535</td><td>0.433</td><td>0.421</td><td>0.426</td><td>0.773</td><td>0.721</td><td>0.745</td><td>0.751</td><td>0.699</td><td>0.724</td></tr><tr><td>ChatGLM-3-6B</td><td>0.502</td><td>0.505</td><td>0.503</td><td>0.391</td><td>0.301</td><td>0.341</td><td>0.739</td><td>0.677</td><td>0.709</td><td>0.685</td><td>0.636</td><td>0.659</td></tr><tr><td>InternLM-2-7B</td><td>0.524</td><td>0.533</td><td>0.527</td><td>0.417</td><td>0.406</td><td>0.411</td><td>0.767</td><td>0.731</td><td>0.748</td><td>0.715</td><td>0.734</td><td>0.724</td></tr><tr><td>Qwen-2-7B</td><td>0.531</td><td>0.523</td><td>0.526</td><td>0.428</td><td>0.422</td><td>0.425</td><td>0.781</td><td>0.753</td><td>0.766</td><td>0.763</td><td>0.729</td><td>0.746</td></tr><tr><td>NPN</td><td>0.539</td><td>0.533</td><td>0.534</td><td>0.409</td><td>0.473</td><td>0.438</td><td>0.843</td><td>0.731</td><td>0.782</td><td>0.776</td><td>0.676</td><td>0.721</td></tr><tr><td>TLNN</td><td>0.551</td><td>0.536</td><td>0.543</td><td>0.436</td><td>0.447</td><td>0.441</td><td>0.822</td><td>0.763</td><td>0.791</td><td>0.772</td><td>0.701</td><td>0.734</td></tr><tr><td>ONEIE</td><td>0.572</td><td>0.531</td><td>0.550</td><td>0.457</td><td>0.441</td><td>0.448</td><td>0.826</td><td>0.792</td><td>0.808</td><td>0.761</td><td>0.739</td><td>0.749</td></tr><tr><td>Ours</td><td>0.568</td><td>0.580</td><td>0.574</td><td>0.461</td><td>0.470</td><td>0.466</td><td>0.811</td><td>0.831</td><td>0.822</td><td>0.751</td><td>0.770</td><td>0.761</td></tr></table>


Table 1: Comparison with baselines in Event Detection.


The objective functions is to maximize the output target sequence $X _ { T }$ probability given the review sentence $X _ { O }$ . Therefore, we optimize the negative log-likelihood loss function: 

$$
\mathcal {L} = \frac {- 1}{| \tau |} \sum_ {(X _ {O}, X _ {T}) \in \tau} \log p (X _ {T} | X _ {O}; \theta)\tag{4}
$$

where $\theta$ is the model parameters, and $( X _ { O } , X _ { T } )$ is a (sentence, target) pair in training set τ, then 

$$
\log p (X _ {T} | X _ {O}; \theta) = \sum_ {i = 1} ^ {n} \log p (x _ {T} ^ {i} | x _ {T} ^ {1}, x _ {T} ^ {2}, \dots x _ {T} ^ {i - 1}, X _ {O}; \theta)\tag{5}
$$

where $p ( x _ { T } ^ { i } | x _ { T } ^ { 1 } , x _ { T } ^ { 2 } , . . . x _ { T } ^ { i - 1 } , X _ { O } ; \theta )$ is calculated by decoder. 

## 5 Experiment

## 5.1 Dataset and Experiment Setting

In this study, we use GuwenEE<sup>3</sup> for Event Ex-traction and CHED2023 (Congcong et al., 2023)for Event Detection. We employ the pre-trainedweight InternLM-XComposer2-VL (Dong et al.,2024) for our Vision-Language Model and LoRAfine-tune the LLM parameters. The image size isset to $4 9 0 \times 4 9 0$ . The glyph is interpreted withtraditional Chinese and Song (宋体). Our experi-ments are carried out with Nvidia RTX A6000.

We use the same criteria as (Zhang et al., 2019; Wadden et al., 2019) for evaluation. A Trigger is correctly identified (Tri-I) if its offsets match a ground truth trigger. It is correctly classified (Tri-C) if its event type also matches. An Argument is correctly identified (Arg-I) if its offsets and event type match a ground truth argument. It is correctly classified (Arg-C) if its role label also matches. 

## 5.2 Main Results

Our baselines could be divided into three parts, including common methods with Chinese language models, such as sequence tagging: Chinese-Bert-CRF (Cui et al., 2020b), Guwen-Bert-CRF (Datahammer, 2020), DMBert (Wang et al., 2019a), and generative models: Mengzi-T5 (Zhang et al., 2021), LLaMA-3-8B (AI@Meta, 2024) LLaMA-3-Chinese-8B (Cui et al., 2023) ChatGLM-3- 6B (Zeng et al., 2023) and Qwen-2-7B (qwe, 2024). We also have feature-enriched models, include NPN (Lin et al., 2018), TLNN (Ding et al., 2019), ONEIE (Lin et al., 2020). 

As shown in Table 1 and Table 2, we first find that, among the common methods, the methods with language models that specifically pretrained on Chinese or classical Chinese can outperform the others(Guwen-Bert versus Chinese-Bert, LLaMA-3-Chinese versus LLaMA-3), revealing the advantage of having a language model pretrained on target language in event extraction. In addition, the methods integrate specific designs for Chinese surpass the common methods above, showing us the value of language adaptation and localization in designing model. 

Moreover, our proposed model exhibits significant improvements over all prior studies $( p \digamma <$ 0.05), demonstrating the efficacy of literature information when applied with large language models for classical Chinese event extraction. To the best of our knowledge, this is the first attempt to leverage literature information in event extraction. 

<table><tr><td rowspan="2">Method</td><td colspan="3">Arg-I</td><td colspan="3">Arg-C</td></tr><tr><td>P.</td><td>R.</td><td>F1.</td><td>P.</td><td>R.</td><td>F1.</td></tr><tr><td>Mengzi-T5-base</td><td>0.377</td><td>0.368</td><td>0.372</td><td>0.354</td><td>0.361</td><td>0.357</td></tr><tr><td>LLaMA-3-8B</td><td>0.369</td><td>0.363</td><td>0.365</td><td>0.337</td><td>0.342</td><td>0.339</td></tr><tr><td>ChatGLM-3-6B</td><td>0.356</td><td>0.355</td><td>0.355</td><td>0.323</td><td>0.321</td><td>0.322</td></tr><tr><td>InternLM-2-7B</td><td>0.362</td><td>0.359</td><td>0.360</td><td>0.341</td><td>0.336</td><td>0.338</td></tr><tr><td>Qwen-2-7B</td><td>0.394</td><td>0.388</td><td>0.391</td><td>0.363</td><td>0.371</td><td>0.366</td></tr><tr><td>ONEIE</td><td>0.383</td><td>0.396</td><td>0.389</td><td>0.367</td><td>0.374</td><td>0.370</td></tr><tr><td>Ours</td><td>0.398</td><td>0.413</td><td>0.406</td><td>0.376</td><td>0.388</td><td>0.382</td></tr></table>


Table 2: Results in Argument Extraction in GuwenEE.


## 5.3 Contribution of Literature Information

After analyzing the overall performance, a natural question arises: How much does the literature information contribute to it? To investigate this, we gradually incorporate various literature information into VLM, starting from the backgrounds up to the visual glyphs. We use "Basic" in Table 3 to refer to the removing of all of the literature information, relying solely on the raw sentence. 

As depicted in Table 3, when using only raw sentences, the performance is notably low, which is excepted since the VLM is not pre-trained on classical Chinese. Significantly improved performance is observed when the background information is included, we attribute this as it provides the macro background knowledge of the sample. Furthermore, the annotation also contributes positively to event extraction, inside which the annotation provides a precise way of communicating the classical and modern Chinese, the semantic variation also stands out as it can detailed explain the changes of semantics. Besides, the glyphic image and visual bridging are designed to actively guide the model to integrate glyphic information, proving that grasping the glyphic feature though visual modality by sentence-level image for capturing more shared glyphes is practical. 

Additionally, our proposed model, which combines all the of the above features, achieves the best performance and showcases the value of literature information in event extraction. 

## 6 Analysis and Discussion

## 6.1 LLMs’ Knowledge in Ancient Literature

We subsequently offer another analysis for our results from the perspective of analyzing LLM’s knowledge in ancient literature. Specifically, we adopt the perplexity metrics (Wan et al., 2024) to evaluate different models’ knowledge on proposed aspects of ancient literature. The perplexities will be calculated over the spans of background, annotation and raw sample in GuwenEE’s test set, lower values represent more knowledge of the model in this aspect. A perplexity of the rest span will also be calculated to show the level of the model’s knowledge in non-classical-Chinese language for comparison, marked as “Plain Text”. 

<table><tr><td rowspan="2">Method</td><td colspan="2">GuwenEE</td><td>CHED</td></tr><tr><td>Tri-C</td><td>Arg-C</td><td>Tri-C</td></tr><tr><td>Basic</td><td>0.411</td><td>0.338</td><td>0.724</td></tr><tr><td>Background</td><td></td><td></td><td></td></tr><tr><td>+Literature source</td><td>0.427</td><td>0.348</td><td>0.731</td></tr><tr><td>+Historical background</td><td>0.434</td><td>0.356</td><td>0.736</td></tr><tr><td>+Biographical protagonist</td><td>0.423</td><td>0.340</td><td>0.729</td></tr><tr><td>+All</td><td>0.438</td><td>0.368</td><td>0.746</td></tr><tr><td>Annotation</td><td></td><td></td><td></td></tr><tr><td>+Semantic variation</td><td>0.430</td><td>0.353</td><td>0.735</td></tr><tr><td>+Phonetic loan character</td><td>0.426</td><td>0.351</td><td>0.728</td></tr><tr><td>+Human Name</td><td>0.424</td><td>0.339</td><td>0.733</td></tr><tr><td>+All</td><td>0.433</td><td>0.359</td><td>0.739</td></tr><tr><td>Glyph</td><td></td><td></td><td></td></tr><tr><td>+Sentence image</td><td>0.422</td><td>0.358</td><td>0.741</td></tr><tr><td>+Sentence image, Visual bridging</td><td>0.439</td><td>0.366</td><td>0.744</td></tr><tr><td>Ours</td><td>0.466</td><td>0.382</td><td>0.761</td></tr></table>


Table 3: The contribution of the literature information.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/c0f0cf6c-021e-4bc6-9c3a-ac6f9fea297e/a69aa1fb17b6f9ab175723225c5fea0de77b1c1f7c7698a92f507889b3187f6c.jpg)



Figure 6: LLMs’ perplexities on proposed aspects.


As shown in Figure 6, current mainstream LLMs have a significant higher perplexity on the classical Chinese aspects when compared with plain texts, validating their lack of corresponding knowledge and our motivation of making up for it. On the other hand, after fine-tuning, our model keeps the values of literature aspects closing to the plain text on the test set, showing the ancient literature knowledge have been effectively injected and facilitating the successful extraction. 

<table><tr><td>Input</td><td>Subtask</td><td>w/o Literature</td><td>w Literature</td><td>Remark</td></tr><tr><td>丞相守道曰:汉文字恐初亦未必能如此</td><td>Argument Identification</td><td>丞相 ✗ Prime minister</td><td>丞相守道√ Prime minister Shoudao</td><td>完颜守道 (Shoudao Wanyan) is a person&#x27;s in Jing Dynasty.</td></tr><tr><td>天子诏帝朝会乘舆升殿</td><td>Trigger Identification</td><td>升殿 ✗ Went to palace</td><td>诏√ Convene</td><td>The radical “i”(speech) in “诏” (edict) indicating a action.</td></tr><tr><td>已而信约,与赵、魏合从畔秦</td><td>Trigger Identification</td><td>合从 ✗ Join</td><td>畔√ Betray</td><td>“畔”(bank) is the same as “叛”(betray).</td></tr><tr><td>帝次山阳,闻败,卷甲与数十人造江上征问</td><td>Trigger Classification</td><td>次:人生-死亡 ✗ Life-Death</td><td>次:军事-备战-驻扎√ Military-War-Garrison</td><td>“次”(times) could represent “驻扎”(garrison).</td></tr></table>


Table 4: Case study


<table><tr><td rowspan="2" colspan="2">Method</td><td colspan="2">GuwenEE</td><td>CHED</td></tr><tr><td>Tri-C</td><td>Arg-C</td><td>Tri-C</td></tr><tr><td colspan="2">Basic</td><td>0.411</td><td>0.338</td><td>0.724</td></tr><tr><td rowspan="4">Source</td><td>InternLM-2-7B</td><td>0.423</td><td>0.353</td><td>0.734</td></tr><tr><td>Qwen-2-7B</td><td>0.431</td><td>0.361</td><td>0.748</td></tr><tr><td>LLaMA-3-8B</td><td>0.436</td><td>0.359</td><td>0.744</td></tr><tr><td>Literature (Ours)</td><td>0.466</td><td>0.382</td><td>0.761</td></tr></table>


Table 5: Comparison with various information sources.


## 6.2 Impact of Literature Information Source

As we pay efforts to obtain the precise literatureinformation from Wikipedia and 古代汉语词典,a intuitive question is Can the powerful largelanguage models replace them? We thus inves-tigate this question by replacing all our literatureinformation sources with LLMs. Specifically, forall the information shown in Figure 2, we designprompts (in Appendix A) to fetch the informationfrom LLM at one stop, and solely rely on the out-put information to proceed the entire workflow.

As shown in Table 5, the human-made literature sources have shown their value again: our way of obtain the precise literature information beats all of the LLM sources, giving us a conclusion that current level of LLMs still can not replace the precise human-made literature as they are suffering from the hallucination and the lack of pre-trained knowledge in various specific minor domains, the reliable human-made literature still can be a valuable ally for us in the era of large language models. 

## 6.3 Analysis of Data Efficiency

Compared with the raw sentence, one of the advantages of literature information is that there are large amount of shared knowledge such as the historic background (e.g., two samples come from the same dynasty), making it easier to build semantic connection across samples with a small size of training data. We thus investigate how the literature improves the data efficiency of our model by comparing with using raw sentence under limited 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/c0f0cf6c-021e-4bc6-9c3a-ac6f9fea297e/8cecf8025e7f85af67edbc070a39789db3ce81a6e80d7031b6c7449902a4b1af.jpg)



Figure 7: Improvement of data efficiency.


training data in Figure 7. 

From the figure, we find that the more training data, the higher performance our proposed model can reach. Moreover, the advantage of the performance brought by the literature information increases under limited data size, showing the superiority of literature information in low resource situation where a pool of shared knowledge can be easily build compared with raw sentences. 

## 7 Cases Study

We launch case studies to make a more intuitive comparison between the VLMs with and without literature information in Table 4. 

We demonstrate that historic background canhelp effectively identify the argument target in thefirst examples. The “守道” is a person in Jing(金)Dynasty and the VLM without literature informa-tion ignores it, resulting in wrong identification.

Furthermore, we illustrate that the glyph canbetter convey semantics in the second example,where the “讠”(speech) in “诏”(edict) indicating“诏” as an action and the VLM can easily identifythe right target with it.

The third and fourth example showcase thevalue of annotation: the “畔” was interpreted asa bank of river while the “次” was interpreted asa death event without modern Chinese annotationwhile ours can correctly comprehend them as “叛乱”(betray) and “驻扎”(garrison) respectively.

## 8 Conclusion

In this study, we explore how to improve the classical Chinese event extraction from the inimitable characteristics of it. By leveraging the longexisting yet often overlooked origin of literature of classical Chinese, our proposed LVLM achieves SOTA performance in several benchmarks without the need for complex and costly annotation. 

Furthermore, our results validate that for the works focusing on the minor languages, solely relaying on the LLMs’ knowledge whose major training corpus is in English may not be an optimal option. Instead, the reliable human-made features still can be a precise and valuable ally for us. 

## Limitations

The limitations of our work can be stated from two perspectives. Firstly, besides the literature information, there is another feature whose effect on downstream tasks is not yet known: pronunciation. In future research, further exploration of the impact of pronunciation, especially the variation of pronunciation between ancient and modern Chinese could provide valuable insights. 

Secondly, our focus has been primarily on a single language. While we have achieved promising results in this language, it is important to acknowl edge that the generalizability of our approach is limited since other languages may not have the source of ancient literature. 

## Acknowledgments

We would like to thank the anonymous reviewers for their excellent feedback. This work is supported by The Hong Kong Polytechnic University Projects (#P0048932, #P0051089). 

## References



2024. Qwen2 technical report. 





AI@Meta. 2024. Llama 3 model card. 





Xiaoyi Bao, Jinghang Gu, Zhongqing Wang, Minjie Qiang, and Chu-Ren Huang. 2024. Employing glyphic information for Chinese event extraction with vision-language model. In Findings of the Association for Computational Linguistics: EMNLP 





2024, pages 1068–1080, Miami, Florida, USA. Association for Computational Linguistics. 





Yubo Chen, Liheng Xu, Kang Liu, Daojian Zeng, and Jun Zhao. 2015. Event extraction via dynamic multipooling convolutional neural networks. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 167–176, Beijing, China. Association for Computational Linguistics. 





Zheng Chen and Heng Ji. 2009. Language specific issue and feature exploration in Chinese event extraction. In Proceedings of Human Language Technologies: The 2009 Annual Conference ofthe North American Chapter of the Association for Computational Linguistics, Companion Volume: Short Papers, pages 209–212, Boulder, Colorado. Association for Computational Linguistics. 





Wei Congcong, Feng Zhenbing, Huang Shutan, Li Wei, and Shao Yanqiu. 2023. CHED: A cross-historical dataset with a logical event schema for classical Chinese event detection. In Proceedings ofthe 22nd Chinese National Conference on Computational Linguistics, pages 875–888, Harbin, China. Chinese Information Processing Society of China. 





Shiyao Cui, Bowen Yu, Tingwen Liu, Zhenyu Zhang, Xuebin Wang, and Jinqiao Shi. 2020a. Edgeenhanced graph convolution networks for event detection with syntactic relation. In Findings of the Associationfor Computational Linguistics: EMNLP 2020, pages 2329–2339, Online. Association for Computational Linguistics. 





Yiming Cui, Wanxiang Che, Ting Liu, Bing Qin, Shijin Wang, and Guoping Hu. 2020b. Revisiting pretrained models for Chinese natural language processing. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 657–668, Online. Association for Computational Linguistics. 





Yiming Cui, Ziqing Yang, and Xin Yao. 2023. Efficient and effective text encoding for chinese llama and alpaca. arXiv preprint arXiv:2304.08177. 





Datahammer. 2020. Guwen-bert-base. 





Ning Ding, Ziran Li, Zhiyuan Liu, Haitao Zheng, and Zibo Lin. 2019. Event detection with triggeraware lattice neural network. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 347–356, Hong Kong, China. Association for Computational Linguistics. 





Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Xilin Wei, Songyang Zhang, Haodong Duan, Maosong Cao, Wenwei Zhang, Yining Li, Hang Yan, Yang Gao, Xinyue Zhang, Wei Li, Jingwen Li, Kai Chen, Conghui He, 





Xingcheng Zhang, Yu Qiao, Dahua Lin, and Jiaqi Wang. 2024. Internlm-xcomposer2: Mastering freeform text-image composition and comprehension in vision-language large model. 





Zhiyuan Fan and Shizhu He. 2023. Efficient data learning for open information extraction with pre-trained language models. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 13056–13063, Singapore. Association for Computational Linguistics. 





Hen-Hsen Huang, Chuen-Tsai Sun, and Hsin-Hsi Chen. 2010. Classical Chinese sentence segmentation. In CIPS-SIGHAN Joint Conference on Chinese Language Processing. 





Zijing Ji, Yuxin Shen, Yining Sun, Tian Yu, and Xin Wang. 2021. C-clue: A benchmark of classical chinese based on a crowdsourcing system for knowl edge graph construction. In Knowledge Graph and Semantic Computing: Knowledge Graph Empowers New Infrastructure Construction, pages 295–301, Singapore. Springer Singapore. 





Lihong Jiang and Mingshan Ren. 2023. A Brief Introduction to Form and Meaning ofChinese Characters & Dictionaries, pages 9–48. Springer Nature Singapore, Singapore. 





Tetsu Kasanishi, Masaru Isonuma, Junichiro Mori, and Ichiro Sakata. 2023. SciReviewGen: A large-scale dataset for automatic literature review generation. In Findings of the Association for Computational Linguistics: ACL 2023, pages 6695–6715, Toronto, Canada. Association for Computational Linguistics. 





Jiaqi Li, Chuanyi Zhang, Miaozeng Du, Dehai Min, Yongrui Chen, and Guilin Qi. 2023a. Three stream based multi-level event contrastive learning for textvideo event extraction. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 1666–1676, Singapore. Association for Computational Linguistics. 





Peifeng Li and Guodong Zhou. 2012. Employing morphological structures and sememes for Chinese event extraction. In Proceedings of COLING 2012, pages 1619–1634, Mumbai, India. The COLING 2012 Organizing Committee. 





Peifeng Li, Guodong Zhou, Qiaoming Zhu, and Libin Hou. 2012. Employing compositional semantics and discourse consistency in Chinese event extraction. In Proceedings of the 2012 Joint Conference on Empirical Methods in Natural Language Processing and Computational Natural Language Learning, pages 1006–1016, Jeju Island, Korea. Association for Computational Linguistics. 





Ruiqi Li, Patrik Haslum, and Leyang Cui. 2023b. EDeR: Towards understanding dependency relations between events. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 14969–14983, Singapore. Association for Computational Linguistics. 





Hongyu Lin, Yaojie Lu, Xianpei Han, and Le Sun. 2018. Nugget proposal networks for Chinese event detection. In Proceedings of the 56th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 1565–1574, Melbourne, Australia. Association for Computational Linguistics. 





Ying Lin, Heng Ji, Fei Huang, and Lingfei Wu. 2020. A joint neural model for information extraction with global features. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 7999–8009, Online. Association for Computational Linguistics. 





Jian Liu, Dianbo Sui, Kang Liu, Haoyan Liu, and Zhe Zhao. 2023. Learning with partial annotations for event detection. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 508– 523, Toronto, Canada. Association for Computational Linguistics. 





Jiangwei Liu, Jingshu Zhang, Xiaohong Huang, and Liangyu Min. 2021. Syntactic-gcn bert based chinese event extraction. 





Yaojie Lu, Hongyu Lin, Jin Xu, Xianpei Han, Jialong Tang, Annan Li, Le Sun, Meng Liao, and Shaoyi Chen. 2021. Text2Event: Controllable sequence-tostructure generation for end-to-end event extraction. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 2795–2806, Online. Association for Computational Linguistics. 





Bodhisattwa Prasad Majumder, Navneet Potti, Sandeep Tata, James Bradley Wendt, Qi Zhao, and Marc Najork. 2020. Representation learning for information extraction from form-like documents. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 6495– 6504, Online. Association for Computational Linguistics. 





Aakanksha Naik, Sravanthi Parasa, Sergey Feldman, Lucy Lu Wang, and Tom Hope. 2022. Literatureaugmented clinical outcome prediction. In Findings of the Association for Computational Linguistics: NAACL 2022, pages 438–453, Seattle, United States. Association for Computational Linguistics. 





Khanh Duy Nguyen, Zixuan Zhang, Reece Suchocki, Sha Li, Martha Palmer, Susan Windisch Brown, Jiawei Han, and Heng Ji. 2023. RESIN-EDITOR: A schema-guided hierarchical event graph visualizer and editor. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 365–372, Singapore. Association for Computational Linguistics. 





Alessio Palmero Aprosio and Sara Tonelli. 2015. Recognizing biographical sections in Wikipedia. In Pro-





ceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, pages 811– 816, Lisbon, Portugal. Association for Computational Linguistics. 





Valerio Perrone, Marco Palma, Simon Hengchen, Alessandro Vatri, Jim Q. Smith, and Barbara McGillivray. 2019. GASC: Genre-aware semantic change for Ancient Greek. In Proceedings of the 1st International Workshop on Computational Approaches to Historical Language Change, pages 56– 66, Florence, Italy. Association for Computational Linguistics. 





Josef Ruppenhofer, Ines Rehbein, and Carolina Flinz. 2020. Fine-grained named entity annotations for German biographic interviews. In Proceedings of the Twelfth Language Resources and Evaluation Conference, pages 4605–4614, Marseille, France. European Language Resources Association. 





Lei Sha, Jing Liu, Chin-Yew Lin, Sujian Li, Baobao Chang, and Zhifang Sui. 2016. RBPB: Regularization-based pattern balancing method for event extraction. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1224– 1234, Berlin, Germany. Association for Computational Linguistics. 





Shirong Shen, Guilin Qi, Zhen Li, Sheng Bi, and Lusheng Wang. 2020. Hierarchical Chinese legal event extraction via pedal attention mechanism. In Proceedings of the 28th International Confer ence on Computational Linguistics, pages 100–113, Barcelona, Spain (Online). International Committee on Computational Linguistics. 





Xuemei Tang, Qi Su, Jun Wang, and Zekun Deng. 2024. CHisIEC: An information extraction corpus for Ancient Chinese history. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 3192–3202, Torino, Italia. ELRA and ICCL. 





Yanzhi Tian and Yuhang Guo. 2022. Ancient Chinese word segmentation and part-of-speech tagging using data augmentation. In Proceedings of the Second Workshop on Language Technologies for Historical and Ancient Languages, pages 146–149, Marseille, France. European Language Resources Association. 





David Wadden, Ulme Wennberg, Yi Luan, and Hannaneh Hajishirzi. 2019. Entity, relation, and event extraction with contextualized span representations. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 5784– 5789, Hong Kong, China. Association for Computational Linguistics. 





Fanqi Wan, Xinting Huang, Deng Cai, Xiaojun Quan, Wei Bi, and Shuming Shi. 2024. Knowledge fusion 





of large language models. In The Twelfth International Conference on Learning Representations. 





Hao Wang, Hirofumi Shimizu, and Daisuke Kawahara. 2023a. Kanbun-LM: Reading and translating classical Chinese in Japanese methods by language models. In Findings of the Association for Computational Linguistics: ACL 2023, pages 8589– 8601, Toronto, Canada. Association for Computational Linguistics. 





Ping Wang, Shitou Zhang, Zuchao Li, and Jingrui Hou. 2023b. Enhancing Ancient Chinese understanding with derived noisy syntax trees. In Proceedings of the 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 4: Student Research Workshop), pages 83–92, Toronto, Canada. Association for Computational Linguistics. 





Xiaozhi Wang, Xu Han, Zhiyuan Liu, Maosong Sun, and Peng Li. 2019a. Adversarial training for weakly supervised event detection. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 998–1008, Minneapolis, Minnesota. Association for Computational Linguistics. 





Xiaozhi Wang, Ziqi Wang, Xu Han, Zhiyuan Liu, Juanzi Li, Peng Li, Maosong Sun, Jie Zhou, and Xiang Ren. 2019b. HMEAE: Hierarchical modular event argument extraction. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 5777–5783, Hong Kong, China. Association for Computational Linguistics. 





Xuebin Wang and Zhenghua Li. 2024. Two sequence labeling approaches to sentence segmentation and punctuation prediction for classic Chinese texts. In Proceedings of the Third Workshop on Language Technologies for Historical and Ancient Languages (LT4HALA) @ LREC-COLING-2024, pages 237– 241, Torino, Italia. ELRA and ICCL. 





Yuxuan Wang, Jack Wang, Dongyan Zhao, and Zilong Zheng. 2023c. Rethinking dictionaries and glyphs for Chinese language pre-training. In Findings of the Associationfor Computational Linguistics: ACL 2023, pages 1089–1101, Toronto, Canada. Association for Computational Linguistics. 





Nuo Xu, Haihua Xie, and Dongyan Zhao. 2020. A novel joint framework for multiple Chinese events extraction. In Proceedings of the 19th Chinese National Conference on Computational Linguistics, pages 950–961, Haikou, China. Chinese Information Processing Society of China. 





Yuqing Yang, Qipeng Guo, Xiangkun Hu, Yue Zhang, Xipeng Qiu, and Zheng Zhang. 2023. An AMRbased link prediction approach for document-level event argument extraction. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 





Zhuosheng Zhang, Hanqing Zhang, Keming Chen, Yuhang Guo, Jingyun Hua, Yulong Wang, and Ming Zhou. 2021. Mengzi: Towards lightweight yet ingenious pre-trained models for chinese. 





Aohan Zeng, Xiao Liu, Zhengxiao Du, Zihan Wang, Hanyu Lai, Ming Ding, Zhuoyi Yang, Yifan Xu, Wendi Zheng, Xiao Xia, Weng Lam Tam, Zixuan Ma, Yufei Xue, Jidong Zhai, Wenguang Chen, Peng Zhang, Yuxiao Dong, and Jie Tang. 2023. Glm-130b: An open bilingual pre-trained model. 





Xingxing Zhang and Mirella Lapata. 2014. Chinese poetry generation with recurrent neural networks. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 670–680, Doha, Qatar. Association for Computational Linguistics. 





pages 12876–12889, Toronto, Canada. Association for Computational Linguistics. 



![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/c0f0cf6c-021e-4bc6-9c3a-ac6f9fea297e/af17ee22cbc0b2aed0866408ea2a203fe46751dc2e163f48fb2b8510f315fed8.jpg)


## A Prompts for Querying LLMs

In Figure 8, we give the prompts for querying with LLMs about the literature information to check if the LLMs can replace the precise information obtaining method we proposed, which is investigated in Section 6.2. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/c0f0cf6c-021e-4bc6-9c3a-ac6f9fea297e/432198aed1d09879aaf99f8ad79e5fd0a0a3804512bec06eabb30fff8464a655.jpg)



Figure 8: Illustration of the prompt.
