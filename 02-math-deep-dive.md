# Math Deep Dive — ELF: Embedded Language Flows

**Arxiv:** https://arxiv.org/abs/2605.10938
**Studied:** 2026-05-14

> Mathematician-grade walk-through. Definitions, derivations, load-bearing assumptions. No paraphrase.

---

## Setup & Notation

Let $\mathcal{V}$ be a finite token vocabulary and $E: \mathcal{V} \to \mathbb{R}^d$ a token-embedding function (so $E(v)$ is a $d$-dimensional vector). The data distribution is over sequences $x = (x_1, \ldots, x_L) \in \mathcal{V}^L$.

Define the *clean* embedding tensor for sequence $x$ as $z_1 := \big(E(x_1), \ldots, E(x_L)\big) \in \mathbb{R}^{L \times d}$. Sample noise $z_0 \sim \mathcal{N}(0, I)$ in the same shape. The flow matching path parameterised by $t \in [0, 1]$ is linear:

$$
z_t = (1 - t) z_0 + t z_1. \tag{1}
$$

Equivalently $z_t = z_1 + (1-t)(z_0 - z_1)$, and $\frac{dz_t}{dt} = z_1 - z_0$ is the *target velocity*.

ELF operates in this continuous embedding space at every $t \in (0, 1)$ and only discretizes at $t = 1$ via an unembedding/decoding map $\pi_\theta: \mathbb{R}^{L \times d} \to \Delta(\mathcal{V})^L$ (one categorical per position).

## Central Object — Flow Matching Velocity in Embedding Space

A neural network $v_\theta(z_t, t)$ is trained to predict the target velocity $z_1 - z_0$, minimising:

$$
\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{x, z_0, t}\Big[ \big\| v_\theta(z_t, t) - (z_1 - z_0) \big\|^2 \Big], \tag{2}
$$

where $t \sim \mathrm{Uniform}(0, 1)$, $z_0 \sim \mathcal{N}(0, I)$, $x \sim p_{\text{data}}$, and $z_1 = E(x)$.

Equivalently the network can be reparameterized to predict $\hat{x}_\theta(z_t, t) := \mathbb{E}[z_1 \mid z_t]$ (the "clean prediction"), and velocity recovered as $v_\theta = \hat{x}_\theta - z_0$ when the path is linear and the noise is fixed. ELF uses the $\hat{x}_\theta$ parameterisation throughout.

### Inference (sampling)

Start from $z_0 \sim \mathcal{N}(0, I)$. Solve the ODE
$$
\frac{dz_t}{dt} = v_\theta(z_t, t), \quad t \in [0, 1)
$$
with a numerical integrator (Euler with $N$ steps, or higher-order). At $t \to 1$, instead of a separate decoder, the same network is repurposed: $\pi_\theta(z_t \to 1)$ produces logits over $\mathcal{V}^L$ via the unembedding matrix $W_E^\top$ (tied to the input embedding). Final discrete output is $\arg\max_v \pi_\theta(\cdot)$ or a categorical sample.

## Final-Step Discretization

Where prior continuous DLMs require a separately trained decoder (autoregressive or non-autoregressive), ELF observes that:

$$
\pi_\theta(\hat{x}_\theta(z_t, t)) := \mathrm{softmax}\big( W_E \cdot \hat{x}_\theta(z_t, t) \big)
$$

with $W_E \in \mathbb{R}^{|\mathcal{V}| \times d}$ being the same matrix used to embed tokens at input (weight tying). The categorical at the final timestep gives discrete tokens. Training-time loss for this final step is the cross-entropy:

$$
\mathcal{L}_{\text{CE}}(\theta) = -\mathbb{E}_{x, z_0}\Big[ \sum_{i=1}^{L} \log \pi_\theta(x_i \mid \hat{x}_\theta(z_{t=1}, 1)) \Big].
$$

The full training objective is a weighted sum
$$
\mathcal{L}(\theta) = \mathcal{L}_{\text{FM}}(\theta) + \alpha \cdot \mathcal{L}_{\text{CE}}(\theta), \tag{3}
$$
with $\alpha$ a small constant (the paper reports good performance across a wide $\alpha$ range, indicating $\mathcal{L}_{\text{FM}}$ does most of the work).

## Classifier-Free Guidance in Flow Matching

CFG combines a *conditional* and an *unconditional* velocity field with linear extrapolation:

$$
v_{\text{cfg}}(z_t \mid c) = \omega \cdot v_\theta(z_t \mid c) + (1 - \omega) \cdot v_\theta(z_t \mid \emptyset), \tag{4}
$$

where $c$ is a conditioning signal (in ELF: from *self-conditioning* — the model's own intermediate prediction), $\emptyset$ is a null/unconditional token, and $\omega \geq 1$ is the guidance scale.

Naive inference requires two forward passes per step (one with $c$, one with $\emptyset$), doubling compute. ELF adopts *training-time CFG*: train the network to directly output $v_\text{cfg}$ in a single forward pass, eliminating the inference overhead. This is the same trick used in Visual Generation Without Guidance and Mean Flows.

The empirical observation: increasing $\omega$ lowers generative perplexity (better samples under GPT-2-Large evaluation) but reduces unigram entropy (less diversity). The Pareto front is the quality-diversity trade-off swept by varying $\omega$.

## Self-Conditioning

The conditioning $c$ in ELF is not external — it is the model's own intermediate prediction. Specifically, at training step, sample a random fraction of the path; let $\tilde{c} = \hat{x}_\theta(z_{t'}, t')$ for some $t' \in [0, 1]$; condition the velocity prediction on $\tilde{c}$. With dropout on $\tilde{c}$, the model learns both conditional ($\tilde{c}$ available) and unconditional ($\tilde{c} \to \emptyset$) regimes simultaneously, enabling the training-time CFG.

## Embedding Choice

The encoder $E$ that maps tokens to embeddings can be:
- pretrained contextual (e.g., a transformer's input embeddings — wins by a clear margin in the ablations)
- non-contextual learned (just a lookup table — worse)
- frozen random (a Random projection — surprisingly OK for some configurations but worse than pretrained)

Pretrained contextual embeddings are the sweet spot. The paper's intuition: a well-structured embedding manifold makes the velocity field easier to learn because nearby embeddings correspond to semantically related tokens.

## Load-Bearing Assumptions

| Assumption | Used in | Failure mode if violated |
|---|---|---|
| Linear interpolation path $z_t = (1-t)z_0 + t z_1$ | Eq. (1), training velocity target | A non-linear path (e.g., DDPM cosine schedule) changes the velocity target form; many results would re-derive |
| Weight-tied embedding/unembedding matrix $W_E$ | Final-step discretization without separate decoder | Untied tokens would require a separate decoder — losing the headline "no separate decoder" advantage |
| Pretrained contextual embeddings preserve semantic structure | Empirical lift over random / non-contextual | Random embeddings $\implies$ velocity field has to do double-duty (learn semantic structure and learn velocity), training is harder |
| Self-conditioning provides useful $c$ for CFG | Training-time CFG without external conditioning | If self-conditioning is too noisy, CFG either provides no lift or hurts |
| GPT-2-Large is a faithful proxy for "quality" | Entire empirical headline | GPT-2-Large gives low PPL to its own distribution — measuring against it tilts the evaluation toward GPT-2-like outputs |

## Gaps Flagged

- **Why is the linear path optimal?** The paper picks linear interpolation by convention from rectified flow. No ablation against curved paths (e.g., variance-preserving SDE, OT-style coupling). For language specifically, the optimal $z_0 \to z_1$ path is an empirical open question.
- **CFG monotonicity claim.** The paper observes that increasing $\omega$ monotonically improves generative perplexity (over the swept range $\omega \in [1, 3]$). This is empirical; no derivation. There's likely a Pareto frontier beyond which $\omega$ saturates or hurts. The paper doesn't sweep high enough to find it.
- **The shared-weight network does TWO jobs** (denoising + final-step decoding). The paper claims this works empirically but does not analyse: why doesn't the network "specialise" the final step at the cost of intermediate denoising? Implicit answer: $\mathcal{L}_\text{FM}$ at $t < 1$ provides enough gradient signal to keep intermediate denoising sharp. Worth a formal capacity argument.
- **The training-time CFG trick is presented without a derivation showing it yields the same $v_\text{cfg}$ at inference as inference-time CFG.** The image-domain papers it cites have the derivation; the assumption that it transfers to embedding-space flow matching is not verified, only validated.

## Alternative Formulations

- **As DDPM in embedding space.** ELF can be reformulated as DDPM with a noise schedule that matches the linear flow path; flow matching is conceptually equivalent to DDPM up to time reparameterisation under Gaussian noising of fixed variance.
- **As Latent Diffusion (LDM).** ELF is LDM with: (a) a frozen pretrained encoder (the embedding matrix), (b) no separate decoder (the same network does final-step decoding). It's "LDM with weight-tied unembedding as decoder."
- **As discrete diffusion with infinite embedding dimension.** In the limit where the embedding dimension $d \gg |\mathcal{V}|$ and embeddings are mutually orthogonal, embedding-space diffusion reduces to discrete diffusion. ELF's win is that practical $d$ is much smaller than $|\mathcal{V}|$ and the manifold is non-uniform, which is where the lift comes from.

## Connections

- Rectified Flow (Liu et al., 2022): the linear path comes directly from RF.
- Flow Matching (Lipman et al., 2022): the velocity-prediction parameterisation.
- DiT / Stable Diffusion: the CFG and self-conditioning patterns lifted from image-domain.
- Latent Diffusion (LDM, Rombach et al., 2022): the "operate in a learned latent space" idea.
- MDLM, SEDD (discrete DLMs): the comparison baselines.
- Diffusion-LM (Li et al., 2022): pioneering continuous DLM with per-step CE loss — ELF specifically avoids the per-step CE to maximise flow flexibility.
