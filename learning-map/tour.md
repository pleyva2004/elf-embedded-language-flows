# Learning-Map Tour — ELF: Embedded Language Flows

> Hu, Qiu, Lu, Zhao, Li, Kim, Andreas, He (2026). *Embedded Language Flows*. arXiv:2605.10938.

---

## 1. Reader's Contract

**Audience.** Math-grad to ML researcher. The reader is comfortable with measure-theoretic probability, ODEs/SDEs at the level of a first graduate course, and basic Riemannian geometry; familiarity with Transformers and language modeling is assumed but not deep expertise in diffusion-language work.

**Background.** Before reading the paper itself, walk the [math-foundations](https://github.com/pleyva2004/math-foundations) tour at the **Graduate** entry point. In particular, the reader who has not internalized continuity equations / Liouville flow on densities should warm up there. Foundations links are given concept-by-concept in Section 2.

**Expected reading time.** 3 to 4 hours, broken roughly as
- 30 min — foundations refresher (Section 2 of this tour)
- 90 min — first read of `02-math-deep-dive.md` end-to-end
- 30 min — `05-improvements.tex` (the proposed extensions and their proofs)
- 60 min — running `cfg-schedule.py`, sketching an experiment, taking your own notes in `03-opinions.md`

**Elevator (one sentence).** ELF runs continuous-time flow matching in a *frozen pretrained embedding space* and reuses *the same network's final timestep* as the decoder via the *weight-tied unembedding matrix*.

**Why it matters.** The headline win is structural, not just empirical: discrete-token diffusion models (MDLM, SEDD) and continuous-noise diffusion language models had been treated as separate research programs. ELF shows that the second program, done in the *right* embedding space and decoded by the *same* network, dominates the first while shedding the separate-decoder overhead.

---

## 2. Foundations Walk

Topologically ordered. Each row links into [math-foundations](https://github.com/pleyva2004/math-foundations); the "needs this for" column points back into the math deep dive (`02-math-deep-dive.md`).

| # | Concept | Pacing | Why ELF needs this |
|---|---------|--------|--------------------|
| 06 | [Linear Maps and Matrices](https://github.com/pleyva2004/math-foundations/blob/main/concepts/06-linear-maps/README.md) | skim | The token embedding $W_E$ and its transpose $W_E^\top$ are the load-bearing linear maps; weight-tying says the two are the same matrix. |
| 14 | [Gradient and Jacobian](https://github.com/pleyva2004/math-foundations/blob/main/concepts/14-gradient-jacobian/README.md) | read | The velocity field $v_\theta(z_t, t)$ is trained against a Jacobian-defined target; gradient flow on the FM loss is the optimization story. |
| 24 | [Probability Density Functions](https://github.com/pleyva2004/math-foundations/blob/main/concepts/24-pdf/README.md) | read | $z_0 \sim \mathcal{N}(0, I)$ and $z_1 \sim p_{\text{data}}$ are densities on $\mathbb{R}^d$; ELF is a transport between them. |
| 25 | [Expectation](https://github.com/pleyva2004/math-foundations/blob/main/concepts/25-expectation/README.md) | skim | The flow-matching loss $\mathcal{L}_\text{FM} = \mathbb{E}_{t, z_0, z_1} \lVert v_\theta(z_t, t) - (z_1 - z_0) \rVert^2$ is an expectation over a triple. |
| 28 | [Change of Variables (Probability)](https://github.com/pleyva2004/math-foundations/blob/main/concepts/28-change-of-variables-probability/README.md) | drill into proof | The continuity equation $\partial_t p_t + \nabla \cdot (p_t v_t) = 0$ is the infinitesimal change-of-variables that makes flow matching well-defined. |
| 29 | [Ordinary Differential Equations](https://github.com/pleyva2004/math-foundations/blob/main/concepts/29-ode/README.md) | read | Inference is a deterministic ODE: $dz_t/dt = v_\theta(z_t, t)$, integrated from $t=0$ to $t=1$ with an Euler or Heun integrator. |
| 32 | [Brownian Motion / Wiener Process](https://github.com/pleyva2004/math-foundations/blob/main/concepts/32-brownian-motion/README.md) | skim | Background: continuous-time noise diffusion (the lineage ELF lives in) is built on Brownian motion; ELF's *deterministic* flow drops the noise term but inherits the framing. |
| 33 | [Stochastic Differential Equations](https://github.com/pleyva2004/math-foundations/blob/main/concepts/33-sde/README.md) | skim | DDPM/SEDD are SDEs; flow matching is the deterministic ODE limit. Knowing both side-by-side clarifies what ELF gives up (stochasticity) and what it gains (determinism, inversion). |
| 37 | [Cross-Entropy](https://github.com/pleyva2004/math-foundations/blob/main/concepts/37-cross-entropy/README.md) | read | The final-step loss is cross-entropy between $W_E^\top z_1$ (logits) and the true token, tying the continuous flow to a discrete categorical objective. |
| 38 | [KL Divergence](https://github.com/pleyva2004/math-foundations/blob/main/concepts/38-kl-divergence/README.md) | skim | CFG can be re-derived as a KL-regularised posterior interpolation; useful framing for the "why does $\omega > 1$ help?" question. |
| 40 | [Information Geometry (Fisher Metric)](https://github.com/pleyva2004/math-foundations/blob/main/concepts/40-information-geometry/README.md) | drill into proof | The proposed *curved-path* improvement (Math #1) lives on a Riemannian manifold whose metric is the Fisher information at each embedding. |
| 41 | [Optimal Transport (Wasserstein)](https://github.com/pleyva2004/math-foundations/blob/main/concepts/41-optimal-transport/README.md) | drill into proof | Linear interpolation is the Euclidean Wasserstein-2 geodesic; the improvement proposal asks whether the *true* OT geodesic on the embedding manifold differs. |

---

## 3. Paper Concepts Walk

Lifted directly from `02-math-deep-dive.md`. These are the ELF-specific moving parts; each has at most a few lines of formalism.

| Concept | What it does in the paper | Reference |
|---------|---------------------------|-----------|
| **Linear interpolation path** | Defines the trajectory $z_t = (1-t) z_0 + t z_1$ along which the velocity target is computed. The constant velocity $z_1 - z_0$ is the regression target. | Eq. (1) — Setup & Notation |
| **Velocity-field prediction** | The network $v_\theta(z_t, t)$ is trained on $\mathcal{L}_\text{FM} = \mathbb{E}\lVert v_\theta(z_t, t) - (z_1 - z_0) \rVert^2$. This is rectified-flow training transplanted into embedding space. | Eq. (2) — Central Object |
| **Weight-tied unembedding** | At $t=1$, the same embedding matrix $W_E$ used to encode tokens is reused (transposed) to decode: $\text{logits} = W_E^\top \cdot z_1$, then softmax-cross-entropy. No separate decoder head. | Final-Step Discretization |
| **Classifier-free guidance (CFG)** | $v_\text{cfg}(z_t \mid c) = \omega v_\theta(z_t \mid c) + (1-\omega) v_\theta(z_t \mid \emptyset)$. ELF imports CFG wholesale from image diffusion. | Eq. (4) — Classifier-Free Guidance |
| **Self-conditioning** | The conditioning signal $c$ is *the model's own intermediate prediction*, not an external label. CFG without an external conditioner. | Self-Conditioning |
| **Training-time CFG** | The network is trained to *output* $v_\text{cfg}$ directly in one forward pass, instead of running two forward passes per inference step. Halves inference cost. | Eq. (4) discussion |
| **Pretrained contextual embeddings** | $W_E$ is initialised from a frozen pretrained encoder (BERT-style). The ablation in the paper shows non-contextual or random embeddings collapse the lift. | Embedding Choice |
| **Latent-diffusion ancestry** | ELF is structurally identical to Latent Diffusion (Rombach et al. 2022) with two specialisations: encoder is the embedding matrix, decoder is the *transpose* of the same matrix. | Connections |

---

## 4. Improvements Walk

Each row in `05-improvements.tex` is graded as a **PROOF** (math goes into a `.tex` derivation) or a **MEASUREMENT** (numbers come from a Python script). Deferred items lack the compute budget for a full retrain and are flagged accordingly.

| # | Type | Title | Artefact | Status |
|---|------|-------|----------|--------|
| Math 1 | **PROOF** | When is the linear path optimal? | [`proofs/curved-path-optimality.tex`](../proofs/curved-path-optimality.tex) | live |
| Math 2 | **MEASUREMENT** | Decomposing embedding-quality vs flow-formulation contributions | (counterfactual training run) | **deferred** — needs full retrain |
| Code 1 | **MEASUREMENT** | Scheduled CFG: $\omega(t)$ rather than constant $\omega$ | [`improvements/cfg-schedule.py`](../improvements/cfg-schedule.py) (extended with `measure()`) | live |
| Code 2 | **MEASUREMENT** | Multimodal embedding sharing prototype | [`improvements/multimodal-embed-share.py`](../improvements/multimodal-embed-share.py) | NEW sketch prototype |
| Exp 1 | **MEASUREMENT** | AR-baseline-controlled comparison at matched compute | (105M-param decoder-only AR run) | **deferred** — needs full training run |
| Exp 2 | **MEASUREMENT** | Multimodal embedding sharing experiment | [`improvements/multimodal-embed-share.py`](../improvements/multimodal-embed-share.py) | live (sketch only) |
| Theory 1 | **PROOF** | ELF as Latent Diffusion with weight-tied decoder | [`proofs/elf-as-LDM.tex`](../proofs/elf-as-LDM.tex) | live |
| Theory 2 | **PROOF** | ELF as discrete information bottleneck | [`proofs/elf-info-bottleneck.tex`](../proofs/elf-info-bottleneck.tex) | **deferred** — formalisation pending |

**Pacing recommendation.** Read Math 1 and Theory 1 in full; they are tractable in a sitting and are the cleanest derivations. Run `cfg-schedule.py` interactively; the toy result is the only piece of original-data evidence in this study and is the single most interview-portable artefact. Glance at Theory 2 and the deferred experiments — they are framed as "what would we measure if we had the GPUs", not as standing claims.

---

## 5. What To Do Next

Three concrete action items, ordered by leverage-per-hour.

1. **Most promising proposal — scheduled CFG.** The U-shaped $\omega(t) = \omega_{\max} \cdot (2t(1-t))^{-\beta}$ schedule (Code Improvement 1) is concrete, prototyped on a 2D toy, and *requires no retraining* to test on a real ELF checkpoint — only inference-time modification. Reach out to the ELF authors with the schedule and the toy result; ask whether they will swap it into one of the released checkpoints.

2. **Single most valuable experiment — the AR baseline at matched compute.** Train a 105M-parameter decoder-only Transformer on the same OpenWebText corpus for the same number of tokens as ELF's headline checkpoint. This is the single missing row of Table 1 that would let a reader judge whether ELF's wins are over the right class of baseline. The compute is non-trivial but bounded ($\sim$1 B tokens, $\sim 105$ M params).

3. **Follow-on paper direction — "How much of ELF is the manifold, how much is the flow?"** Combine Math Improvement 2 (counterfactual: discrete diffusion on the same pretrained embedding space) with the scheduled-CFG result. The paper writes itself: *"ELF's lift decomposes into $X\%$ from the embedding manifold and $Y\%$ from the flow; here is the right CFG schedule for both regimes."* This is the natural next-paper for someone with full compute access; for an interview-grade study, the writeup of the conjecture and the toy-scale evidence is enough.

---

*Last updated: 2026-05-15. Tour artefact for v1.9 per-paper-tour rollout.*
