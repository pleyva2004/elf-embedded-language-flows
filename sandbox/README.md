# Sandbox: elf-embedded-language-flows

> Minimal runnable experiment probing one claim from **ELF: Embedded Language Flows** (Hu et al., 2026, [arxiv:2605.10938](https://arxiv.org/abs/2605.10938)).

This is a subdirectory of the larger study repo — see [`../README.md`](../README.md) for the full artifact set.

---

## The claim being probed

> *Flow matching in continuous embedding space, with weight-tied unembedding for final-step discretization, can recover discrete tokens without a separate decoder.*

If true, then a toy flow-matching pipeline trained on a small vocabulary should: (a) follow a clean linear path from noise to embedding, and (b) at $t = 1$, the closest-embedding-by-dot-product (i.e., softmax over $W_E z$) should land on the correct token a non-trivial fraction of the time.

## Experiment design

1. Create a synthetic vocabulary of $V = 64$ tokens with random $d = 32$-dimensional embeddings $W_E \in \mathbb{R}^{V \times d}$.
2. Sample $z_1 = W_E[v]$ for random target token $v$.
3. Sample $z_0 \sim \mathcal{N}(0, I)$, build $z_t = (1-t)z_0 + t z_1$ for $t \in [0, 1]$.
4. Train a tiny MLP $\hat{x}_\theta(z_t, t)$ to predict $z_1$ from $z_t$ (the "clean prediction" head).
5. At inference: integrate $\frac{dz_t}{dt} = \hat{x}_\theta - z_0$ from $z_0$ via Euler ($N$ steps).
6. At final step: $\arg\max_v (W_E \hat{x}_\theta)$. Measure recovery accuracy vs $N$.

Report accuracy at $N \in \{1, 4, 16, 64\}$ vs a random-token baseline.

## Expected output

- Accuracy should grow with $N$ (more integration steps → closer to true $z_1$).
- Even at $N = 1$ accuracy should beat random (1/64 ≈ 1.6%).
- Demonstrates the *weight-tied unembedding* idea: no separate decoder, just softmax over $W_E$.

## What would falsify the claim

- If recovery accuracy stays near random (1.6%) regardless of $N$, the flow-to-discrete mapping isn't learning the embedding manifold.
- If accuracy plateaus far below 50% with large $N$, the toy result doesn't transfer; the paper's claim relies on something specific to learned (pretrained) embeddings that random embeddings can't substitute for.

## Run

```bash
pip install -r requirements.txt
python experiment.py
```

CPU-runnable, <60s. Trains a tiny MLP (~10K params) and reports accuracy across step counts.

## Files

- `experiment.py` — main script
- `requirements.txt` — pinned deps

## Notes

This is a sandbox-grade probe of the *final-step weight-tied unembedding* idea, NOT a faithful reproduction of ELF's training. The real ELF uses ~105M params on OpenWebText with pretrained contextual embeddings; the probe here uses random embeddings on a synthetic vocabulary. Negative results here would not invalidate the paper, but positive results suggest the core mechanism transfers down to tiny scales.
