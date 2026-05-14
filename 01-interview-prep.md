# Interview Prep — ELF: Embedded Language Flows

**Authors:** Hu, Qiu, Lu, Zhao, Li, Kim, Andreas, He (MIT)
**Arxiv:** https://arxiv.org/abs/2605.10938 (May 2026)
**Studied:** 2026-05-14

---

## One-sentence elevator
Run continuous-time flow matching in a frozen pretrained embedding space; reuse the same network's final timestep as the decoder via the tied unembedding matrix; this minimalist design beats the leading discrete and continuous diffusion language models at lower compute and fewer sampling steps.

## What's novel
- **No separate decoder.** Prior continuous DLMs (Diffusion-LM, CDCD, latent DLMs) train a separate decoder to map embeddings back to tokens. ELF observes the final flow-matching step naturally does this if the unembedding matrix is weight-tied to the input embedding — single network does everything.
- **Stays continuous until the last step.** Most prior continuous DLMs apply per-step cross-entropy supervision (forces the trajectory to "snap" toward valid tokens at every $t$). ELF lets the flow run free in $\mathbb{R}^d$ for $t \in (0, 1)$ and only discretizes at $t = 1$ — gives the flow maximum freedom.
- **Plug-in CFG.** Because ELF mirrors image-domain flow matching structurally, classifier-free guidance from the image literature transfers directly. They use training-time CFG (single forward pass) to avoid the 2× inference cost.
- **Headline numbers.** 105M params (vs 170M for baselines), 10× fewer training tokens, fewer sampling steps, lower generative perplexity. Tested on OpenWebText, WMT14 De→En, and summarization.

## What's mathematically clever
- The training velocity target $z_1 - z_0$ along a linear path $z_t = (1-t)z_0 + t z_1$ is the simplest possible flow matching ansatz, but it composes perfectly with weight-tied softmax decoding at $t = 1$: the unembedding $W_E$ that produced $z_1$ from the input token IS the decoder.
- Training-time CFG: instead of inference-time $v_\text{cfg} = \omega v(z_t|c) + (1-\omega) v(z_t|\emptyset)$ requiring two forward passes, train the net to directly output $v_\text{cfg}$. Same inference cost as unguided sampling.

## What I'd push back on
- **Pretrained contextual embeddings are doing a lot of work.** The ablation shows pretrained $>$ non-contextual $>$ random — but they don't quantify how much of the headline win is the flow matching innovation vs the embedding choice. Likely a fair share of the gap to discrete baselines comes from the embedding manifold being a good starting point.
- **GPT-2-Large as the perplexity evaluator** is sensible but biased: models that match GPT-2's distribution win the metric. Diversity is measured via unigram entropy, but unigram entropy is a weak diversity signal — it doesn't catch repetition or mode collapse at the n-gram level.
- **No comparison against modern autoregressive baselines.** The compared models (MDLM, SEDD, Diffusion-LM) are all diffusion-flavored. The interesting question — is ELF competitive with a similarly-sized AR transformer at the same compute? — is left unanswered.
- **"Minimal adaptation to discrete domain"** somewhat undersells the role of self-conditioning, training-time CFG, and the embedding choice. The minimalism is in the formal structure; the empirical setup has several engineering pieces.

## Open questions
- Does ELF still win when the autoregressive baseline gets the same training budget? "Diffusion beats AR at small scale, AR catches up at scale" is a recurring pattern; the paper's compute regime is small.
- The "continuous-until-last-step" design predicts that the flow trajectory should pass through *invalid* embedding regions (no nearby token). What does the intermediate $z_t$ actually look like — does it stay on a low-dimensional manifold, or does it wander? An interpretability probe would be informative.
- Can ELF extend to multimodal (text + image) by sharing the embedding space? The structural symmetry with image-domain flow matching makes this natural.

## My proposed extensions
*(See [`05-improvements.tex`](./05-improvements.tex) for the full set; top 3 distilled here.)*

- **Scheduled CFG instead of fixed $\omega$.** Replace the single guidance scalar with a schedule $\omega(t)$ — push hard near the boundaries of the path and lightly in the middle. Prototype in `improvements/cfg-schedule.py` demonstrates the schedule sweep on a toy Gaussian-mixture target.
- **Theoretical bound on the linear path's optimality.** Prove (or disprove) that under the embedding-space's local geometry, the linear interpolation is the optimal transport coupling. If not, a curved path could improve quality at the same step count.
- **AR-baseline-controlled comparison.** Train an autoregressive transformer at matched parameter + token budget on the same OWT setup and report alongside Table 1. This is the controlled comparison the field actually needs.
