---
title: 'Large Concept Models'
date: 2026-01-15T14:54:13+00:00
draft: false
description: 'Paper Reading Notes: "Large Concept Models: Language Modeling in a Sentence Representation Space"'
ShowWordCount: true
ShowReadingTime: false
tags:
  - 'attention'
  - 'reasoning'
---


<aside>

Paper link: [https://arxiv.org/pdf/2412.08821](https://arxiv.org/pdf/2412.08821)

</aside>

## 1 Introduction and Motivation

### 1.1 Why Token-Level Generation is Not Enough

Modern large language models (LLMs) are predominantly trained as token-level sequence predictors, generating text autoregressively by estimating the conditional probability of the next token given a preceding context [3, 4]. This paradigm has proven remarkably effective at scale, enabling fluent generation, in-context learning, and broad task generalization. Nevertheless, token-level modeling imposes structural constraints that become more visible as models are pushed toward long-horizon reasoning, planning, and semantic control.

At the token level, meaning is not represented explicitly but is distributed across long sequences of surface-level symbols. High-level semantic structure—such as intent, discourse organization, or abstract relations—must be inferred implicitly from token patterns. Long-range dependencies therefore require maintaining extended contexts, increasing computational cost and making behavior sensitive to context truncation. Moreover, semantic redundancy is pervasive: the same underlying meaning is repeatedly re-encoded across many tokens even when the task primarily concerns semantic coherence rather than lexical detail.

Tokenization further complicates semantic modeling. Subword segmentation introduces language and morphology dependent artifacts that influence representation quality and cross-lingual alignment [5, 6, 7]. While multilingual tokenizers mitigate some issues, the modeling unit remains fundamentally tied to surface form rather than meaning.

These observations suggest that challenges such as brittleness in long-horizon generation, lack of planning, and sensitivity to phrasing may arise not only from architecture, but also from the representational granularity at which language is modeled.

**1.2** **From Tokens to Concepts: High-Level Idea of LCMs**

Large Concept Models (LCMs) propose an alternative abstraction by shifting the primary generative space from tokens to sentence-level semantic representations, referred to as *concepts* [1]. Instead of generating text directly in token space, LCMs operate in a semantic embedding space where each point represents the meaning of an entire sentence or utterance. Generation is performed by modeling transitions between concepts, with surface text produced only as a final decoding step.

This reframing introduces two key ideas. First, it makes semantic abstraction a first-class modeling objective rather than an emergent property of token-level optimization: meaning is represented explicitly in a continuous concept space, decoupled from linguistic realization. Second, it enables generative models—most notably diffusion models—to operate directly on semantic representations, allowing generation via iterative refinement rather than strict left-to-right prediction.

By modeling in concept space, LCMs aim to compress redundant surface information and emphasize high-level structure. This compression acts as an inductive bias, encouraging the model to retain what is semantically salient while discarding incidental lexical variation. In principle, such representations should be better suited for long-horizon generation, semantic planning, and multilingual generalization.

Importantly, LCMs are not proposed as wholesale replacements for token-level LLMs. Instead, they occupy a complementary point in the design space: token-level models provide fine-grained lexical control, whereas concept-level models prioritize semantic coherence and abstraction.

In the remainder of this essay, we review sentence-level semantic representations and their limitations for generation, describe the diffusion-based LCM architecture and an alternative quantized variant, discuss fragility and failure modes, and situate the approach within broader trends in representation-centric modeling and planning-oriented generation.

## 2 Background: Sentence-Level Semantics as a Modeling Substrate

### 2.1 Sentence Embeddings and Semantic Similarity

Sentence-level semantic representations map an entire sentence or utterance into a fixed-dimensional vector that captures its meaning. Unlike token embeddings, which represent local linguistic units, sentence embeddings are designed to encode holistic semantic content and support comparison across sentences of varying length and surface form. They are widely used for semantic textual similarity, retrieval, clustering, and paraphrase detection [8].

A key property of high-quality sentence embeddings is semantic smoothness: sentences with similar meanings are mapped to nearby points, even when their lexical realization differs substantially. This abstraction aligns with how humans often reason about language—at the level of propositions or intents rather than individual words—making sentence embeddings a promising substrate for more compact semantic modeling than token sequences.

However, sentence embeddings are typically learned in a discriminative setting. Models such as Sentence-BERT are optimized to distinguish semantically similar from dissimilar sentence pairs, not to support generation [8]. As a result, while these embeddings are effective for semantic comparison, they are not automatically well-structured for defining a generative process over language.

### 2.2 Language-Agnostic Concept Spaces and SONAR

For concept-level modeling to be meaningful, the underlying semantic space must be robust across languages and domains. If semantically equivalent sentences in different languages are mapped far apart, concept-level modeling would reintroduce language-specific artifacts at a higher level. This motivates language-agnostic sentence encoders.

In the LCM framework, this role is fulfilled by SONAR, a multilingual and multimodal sentence embedding model [2]. SONAR maps sentences from many languages into a shared embedding space where translations and paraphrases are close regardless of surface form. This alignment allows LCMs to model meaning independently of the language in which it is expressed.

SONAR also introduces modularity: semantic encoding is provided by a pretrained encoder, while generative modeling is handled separately. The downside is dependence on the embedding space: biases, blind spots, or missing distinctions in SONAR are inherited by the concept model, a trade-off that reappears in the discussion of limitations.

### 2.3 Why Generative Modeling in Embedding Space is Hard

Continuous sentence embeddings pose a challenge for generative modeling. Unlike tokens, which form a discrete vocabulary with natural categorical distributions, embeddings live in a high-dimensional continuous space. Defining meaningful likelihoods there is non-trivial, especially when the goal is to generate diverse yet semantically coherent samples.

Autoregressive generation does not translate cleanly to sentence embeddings: there is no natural ordering over dimensions, and naive regression tends to produce averaged representations that lack semantic sharpness. Similar issues are well known in other continuous domains such as image generation.

LCMs address this by adopting diffusion-based generative modeling in concept space [1]. Diffusion models define a tractable likelihood via a gradual noising process and learn to reverse it through iterative denoising. By operating directly on semantic embeddings, diffusion enables concept generation without discretizing meaning prematurely.

## 3 Large Concept Models: Diffusion-Based Concept Generation

Large Concept Models depart from token-level language modeling both in representation and in the generative mechanism. Rather than predicting discrete symbols autoregressively, LCMs generate sentence-level semantic representations directly in a continuous embedding space using diffusion models [1]. This section summarizes the diffusion-based architecture, its coarse-to-fine behavior, and the implications of modeling language as semantic trajectories rather than token sequences.

### 3.1 Concept Generation as a Diffusion Process

Diffusion models define a generative process over continuous spaces by gradually transforming structured data into noise and learning to reverse this process [1]. In LCMs, the data distribution consists of sentence embeddings produced by a pretrained concept encoder such as SONAR. The forward process incrementally adds Gaussian noise to these embeddings, while the learned reverse process denoises step by step to recover a meaningful concept representation.

At generation time, an LCM starts from pure noise and progressively denoises it into a semantic embedding representing a plausible sentence-level meaning. The final embedding is then decoded into text. Diffusion is thus not an implementation detail but an enabling choice for stable likelihood-based generation in a continuous semantic space.

### 3.2 Implicit Coarse-to-Fine Semantic Generation

A key empirical observation is that diffusion-based concept generation exhibits a coarse-to-fine semantic structure [1]. Early denoising steps tend to establish high-level semantic properties—such as topic, intent, or overall meaning—while later steps refine details within that semantic frame.

This resembles a planning-like process: rather than committing to early discrete decisions, the model can adjust global semantic structure throughout the denoising trajectory. In contrast, token-level generation must encode global planning implicitly across many localized token decisions.

### 3.3 Conditioning and Context Integration

LCMs can be conditioned on context by incorporating previous concept embeddings or external information into the diffusion model [1]. Concept-level context is typically represented as a sequence of embeddings corresponding to prior sentences, which guides generation of the next concept.

Because these context representations are semantic abstractions, conditioning is compact and meaning-oriented. Compared to token-level conditioning, this reduces sensitivity to context length, but it also shifts the notion of coherence: LCMs primarily enforce semantic consistency rather than lexical or syntactic continuity.

### 3.4 Decoding Concepts into Text

Once a concept embedding has been generated, it must be mapped back into natural language. This is handled by a decoder trained to generate text conditioned on a semantic embedding [1].

The decoder can be viewed as a realization module translating meaning into linguistic form.

This separation enables modularity (e.g., pairing the same concept generator with different decoders), but it also introduces ambiguity: multiple valid surface realizations may correspond to the same concept embedding, reducing fine-grained lexical control. Final text quality therefore depends jointly on the concept generator and decoder calibration.

**3.5** **Quantized LCMs and Alternative Generative Paths**

The paper also explores Quantized LCMs, where continuous embeddings are discretized using residual vector quantization (RVQ) and modeled autoregressively [1]. This aligns more closely with traditional token-level modeling and supports standard autoregressive transformers.

Empirically, diffusion-based LCMs outperform quantized variants on semantic generation quality and robustness [1]. Quantization introduces additional reconstruction error, while diffusion preserves continuity throughout generation. This comparison emphasizes a core lesson of the paper: for sentence-level meaning, iterative refinement in a continuous space is more effective than imposing discrete structure too early.

## 4 Connections to Related Work

Large Concept Models (LCMs) sit at the intersection of representation learning, abstraction-oriented reasoning, compression-based inductive biases, and theoretical analyses of transformer capabilities [10, 9, 12]. Rather than introducing a new token-level reasoning mechanism, LCMs reframe language modeling by changing the unit of representation. 

### 4.1 Representation Choice and Theoretical Expressivity

Theoretical work on transformer expressivity investigates what transformers can compute under constraints such as finite precision, bounded context length, or restricted depth [12]. A recurring theme is that expressivity claims are meaningful only relative to the representations a model operates on.

LCMs engage with this literature not by modifying transformer architectures, but by changing the representation unit. By operating on sentence-level embeddings rather than long token This subsequences, LCMs reduce effective sequence length while increasing semantic density.

gests that some limitations often attributed to token-level transformers—such as sensitivity to long contexts or brittleness in long-horizon coherence—may partly reflect representational granularity rather than intrinsic architectural limits.

This reading aligns with the transformer expressivity, which emphasized that theoretical results depend critically on representational assumptions. LCMs make this dependence concrete by demonstrating how changing representation alone can alter the practical difficulty profile of language modeling tasks.

### 4.2 Abstraction and Intermediate Structure in Reasoning Systems

Another relevant line of work concerns algorithmic and neuro-symbolic approaches to reasoning, which aim to introduce explicit intermediate structure to support systematic generalization [13]. Classical neuro-symbolic systems rely on symbolic representations and hand-designed rules, while token-level neural language models largely avoid explicit structure.

LCMs occupy an intermediate position. Although they do not introduce symbols in the classical sense, quantized variants impose a discrete abstraction layer via RVQ. These discrete units compress token-level variability while preserving semantic similarity, acting as a soft abstraction mechanism rather than a symbolic language.

LCMs suggest that useful intermediate structure can arise from representation design itself, without requiring explicit symbolic rules or hand-crafted algorithmic components.

### 4.3 Compression and Representation-Centric Inductive Biases

Compression-based perspectives argue that compact representations support generalization, either through MDL-style arguments or the information bottleneck framework [9]. Closely related work in representation learning emphasizes that representation choice can be as important as architectural innovation [10].

LCMs instantiate these ideas directly. Quantization constrains representational capacity, forcing semantic information to be concentrated in a limited number of codebooks. Empirical results suggest that early codebooks capture most high-level semantic content, while later codebooks refine finer details [1, 11]. This supports the view that sentence-level meaning can often be encoded compactly.

## 5 Fragility, Limitations, and Failure Modes

While Large Concept Models offer a compelling alternative to token-level modeling, the paper identifies limitations and sources of fragility that follow directly from operating at the level of semantic abstractions [1].

### 5.1 Fragility of Concept-Level Representations

A central challenge is the *fragility* of concept representations. Because LCMs generate meaning in a continuous embedding space, small perturbations in the diffusion trajectory can lead to large semantic shifts in the final embedding. Unlike token-level models, where errors are discrete and localized, diffusion-based generation can accumulate global changes over many denoising steps.

This sensitivity can manifest as semantic drift: generated concepts remain locally coherent but gradually deviate from the intended meaning or context. Over long sequences, such drift can compound, yielding text that is plausible sentence-by-sentence but globally inconsistent. The paper notes that diffusion-based LCMs may be less robust to noise or conditioning errors than token-level autoregressive models, particularly without strong contextual anchors.

### 5.2 Loss of Fine-Grained Control

Another limitation is controllability. Concept embeddings abstract away surface form, making it difficult to enforce precise lexical choices, syntactic constraints, or stylistic requirements [1]. Multiple surface realizations may correspond to the same concept embedding, and the decoder receives only limited guidance on which realization is preferred.

This is especially relevant for tasks requiring exact phrasing or strict adherence to prompts. Token-level models can condition directly on words and structures, though at the cost of longer contexts and more redundant representations. LCMs trade surface-level control for semantic abstraction.

### 5.3 Dependence on Pretrained Encoders and Decoders

LCMs inherit biases and limitations from pretrained components. The concept encoder determines which distinctions are preserved in the embedding space, and the decoder governs how meaning is mapped back to text. If relevant distinctions are not represented well by the encoder, they cannot be reliably recovered downstream.

Because the encoder is typically frozen, the model cannot adapt the semantic space to downstream objectives. While this stabilizes training, it constrains expressivity and domain adaptation. The paper emphasizes that LCM performance is bounded by the quality and coverage of the underlying semantic representations [1].

### 5.4 Comparison with Quantized and Autoregressive Variants

The paper contrasts diffusion-based LCMs with quantized autoregressive variants and observes that while diffusion improves semantic quality, it can introduce instability [1]. Autoregressive models commit monotonically and often localize errors, whereas diffusion revises global representations throughout generation.

This highlights a broader trade-off: diffusion provides flexibility and refinement, but can sacrifice predictability. The fragility observed in LCMs is therefore a structural consequence of continuous, iterative generation at the semantic level.

## 6 Future Directions: Toward Explicit Planning in Concept Space

The LCM framework is presented not as a complete solution to semantic reasoning, but as an enabling abstraction for architectures that reason and plan more explicitly in semantic space [1].

### 6.1 Implicit vs. Explicit Planning

Diffusion-based concept generation already resembles a weak form of planning: instead of committing early to discrete tokens, the model refines a global semantic representation across denoising steps, allowing global revisions that are hard in token-by-token generation.

However, this remains *implicit*. The model does not represent goals, subgoals, or constraints explicitly, nor does it maintain interpretable intermediate plans. As a result, diffusion in concept space improves semantic coherence but does not guarantee consistency, faithfulness, or long-horizon reasoning.

The paper motivates future work on integrating *explicit planning modules* on top of concept representations [1]. Concept space offers a natural interface for such modules: plans could be encoded as trajectories, constraints, or objectives in semantic space rather than sequences of surface tokens.

### 6.2 Concept Space as a Planning Interface

One direction is to treat concept embeddings as states in a planning problem. A planner could propose high-level semantic trajectories that satisfy constraints such as topic consistency, grounding, or task objectives, while diffusion (or related generative mechanisms) could realize these plans robustly.

This aligns with classical planning ideas while avoiding brittle symbolic representations. Concepts are continuous, semantically meaningful, and learned from data, making them suitable for optimization-based or search-based planning methods. The paper suggests that combining concept-level planning with generative refinement could yield more reliable long-horizon behavior [1].

### 6.3 Hybrid Architectures and Long-Horizon Generation

Another extension concerns hybrid architectures combining concept-level and token-level modeling. Token models remain superior for enforcing local syntactic and lexical constraints, while concept-level models provide compact long-range semantic state. A hybrid could use concept-level diffusion for global semantic planning and token-level generation for precise realization.

Such hybrids could mitigate fragility and controllability issues: concept-level planning would maintain global coherence, while token-level decoding would handle local correctness. The paper positions LCMs as a potential backbone for such systems rather than a replacement for existing language models [1].

### 6.4 Beyond Text: Multimodal and Interactive Extensions

Because the concept space is language-agnostic and potentially multimodal, future work could extend LCMs beyond text generation [2]. Planning over concepts could involve speech, images, or actions, enabling models that reason across modalities at a semantic level.

Interactive settings are also promising: users could steer or correct generation at the level of meaning rather than wording, using concept-level representations as a shared semantic state.

## 7 Conclusion

Large Concept Models represent a deliberate shift in how language generation is framed: away from token-by-token surface realization and toward modeling meaning directly in a semantic space [1]. By elevating sentence-level concepts to the primary modeling unit and applying diffusion-based generation over these representations, LCMs challenge the assumption that scaling token-level models alone is sufficient for robust semantic reasoning and long-horizon coherence.

The central contribution of the LCM framework is a reframing of the language modeling problem. Diffusion in concept space allows global semantic structure to be refined iteratively, offering natural mechanism for high-level coherence that is difficult to achieve with strictly autoregressive token generation. At the same time, the paper highlights trade-offs, including fragility, reduced lexical control, and strong dependence on pretrained semantic encoders.

Rather than presenting these limitations as immediate shortcomings, the LCM perspective motivates future extensions toward explicit planning modules operating over concept representations. Concept space emerges as a promising interface for planning and long-horizon reasoning—neither purely symbolic nor tied to surface form.

LCMs reinforce a recurring theme: progress in language modeling does not come solely from scaling architectures or data, but also from rethinking representation. By making semantic abstraction explicit and generative, LCMs provide both a practical approach and a lens for revisiting questions of reasoning, planning, and meaning in neural language models.

## References

[1] LCM Team et al. Large Concept Models: Language Modeling in a Sentence Representation Space. arXiv:2412.08821, 2024. [https://arxiv.org/abs/2412.08821](https://arxiv.org/abs/2412.08821)

[2] P. Duquenne et al. SONAR: Sentence-Level Multimodal and Language-Agnostic Representations. arXiv:2308.11466, 2023. [https://arxiv.org/abs/2308.11466](https://arxiv.org/abs/2308.11466)

[3] A. Vaswani et al. Attention Is All You Need. NeurIPS, 2017. [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

[4] T. Brown et al. Language Models are Few-Shot Learners. NeurIPS, 2020. [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)

[5] R. Sennrich, B. Haddow, and A. Birch. Neural Machine Translation of Rare Words with Subword Units. ACL, 2016. [https://arxiv.org/abs/1508.07909](https://arxiv.org/abs/1508.07909)

[6] T. Kudo. Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates. ACL, 2018. [https://arxiv.org/abs/1804.10959](https://arxiv.org/abs/1804.10959)

[7] P. Rust, J. Pfeiffer, I. Vuli´c, S. Ruder, and E. Gurevych. How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models. ACL, 2021. [https://arxiv.org/abs/2012.15613](https://arxiv.org/abs/2012.15613)

[8] N. Reimers and I. Gurevych. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP-IJCNLP, 2019. [https://arxiv.org/abs/1908.10084](https://arxiv.org/abs/1908.10084)

[9] N. Tishby, F. C. Pereira, and W. Bialek. The Information Bottleneck Method. arXiv:physics/0004057, 2000. [https://arxiv.org/abs/physics/0004057](https://arxiv.org/abs/physics/0004057)

[10] Y. Bengio, A. Courville, and P. Vincent. Representation Learning: A Review and New Perspectives. IEEE TPAMI, 2013. [https://arxiv.org/abs/1206.5538](https://arxiv.org/abs/1206.5538)

[11] N. Zeghidour et al. SoundStream: An End-to-End Neural Audio Codec. IEEE/ACM TASLP, 2021. [https://arxiv.org/abs/2107.03312](https://arxiv.org/abs/2107.03312)

[12] W. Merrill and A. Sabharwal. The Expressive Power of Transformers with Chain of Thought. ICLR, 2024. [https://arxiv.org/abs/2310.07923](https://arxiv.org/abs/2310.07923)

[13] G. Weiss, Y. Goldberg, and E. Yahav. Thinking Like Transformers. ICML, 2021. [https://arxiv.org/abs/2106.06981](https://arxiv.org/abs/2106.06981)