"""Multimodal shared-embedding compatibility probe (ELF Experimental #2).

Tied to 05-improvements.tex, Experimental Extension #2: share the embedding
space across text and image so one ELF flow-matching network operates on a
joint manifold (CLIP-like alignment), with per-modality decoding heads.

SKETCH ONLY — random unit-norm embeddings stand in for real CLIP encoders.
Demonstrates the *measurement infrastructure* (within- vs cross-modal cosine
separation), not a result. Random embeddings will show ~0 separation, the
negative control. Real CLIP encoders should give separation ~0.3-0.5.
"""
from __future__ import annotations

import torch; import torch.nn.functional as F


def embed_text_tokens(vocab_size: int = 20, d: int = 32) -> torch.Tensor:
    """Return (vocab_size, d) unit-norm random text embeddings (sketch)."""
    g = torch.Generator().manual_seed(1)
    return F.normalize(torch.randn(vocab_size, d, generator=g), dim=-1)


def embed_image_patches(n_patches: int = 20, d: int = 32) -> torch.Tensor:
    """Return (n_patches, d) unit-norm random image-patch embeddings (sketch)."""
    g = torch.Generator().manual_seed(2)
    return F.normalize(torch.randn(n_patches, d, generator=g), dim=-1)


def _pairwise_cos(a: torch.Tensor, b: torch.Tensor, exclude_diag: bool) -> torch.Tensor:
    sim = a @ b.T
    if exclude_diag:
        sim = sim[~torch.eye(sim.shape[0], dtype=torch.bool)]
    return sim.flatten()


def compatibility_score(text_emb: torch.Tensor, img_emb: torch.Tensor) -> dict:
    """Mean cosine similarity within-text, within-image, cross-modal, separation."""
    wt = _pairwise_cos(text_emb, text_emb, exclude_diag=True)
    wi = _pairwise_cos(img_emb, img_emb, exclude_diag=True)
    cm = _pairwise_cos(text_emb, img_emb, exclude_diag=False)
    within_mean = 0.5 * (wt.mean().item() + wi.mean().item())
    return {
        "within_text_mean_cos": wt.mean().item(),
        "within_text_std_cos": wt.std().item(),
        "within_image_mean_cos": wi.mean().item(),
        "within_image_std_cos": wi.std().item(),
        "cross_modal_mean_cos": cm.mean().item(),
        "cross_modal_std_cos": cm.std().item(),
        "separation": within_mean - cm.mean().item(),
    }


def measure() -> dict:
    """Run the full sketch probe and return compatibility scores."""
    torch.manual_seed(0)
    t = embed_text_tokens()
    i = embed_image_patches()
    return compatibility_score(t, i)


def main() -> None:
    torch.manual_seed(0)
    t = embed_text_tokens(); i = embed_image_patches()
    s = compatibility_score(t, i)
    print("=== Embedding compatibility probe (SKETCH — random embeddings) ===")
    print(f"Text vocab: {t.shape[0]} tokens, dim {t.shape[1]}")
    print(f"Image patches: {i.shape[0]} patches, dim {i.shape[1]}\n")
    print("=== Cosine-similarity distribution ===")
    print(f"Within text:  mean={s['within_text_mean_cos']:+.3f}, std={s['within_text_std_cos']:.3f}")
    print(f"Within image: mean={s['within_image_mean_cos']:+.3f}, std={s['within_image_std_cos']:.3f}")
    print(f"Cross-modal:  mean={s['cross_modal_mean_cos']:+.3f}, std={s['cross_modal_std_cos']:.3f}\n")
    print(f"=== Verdict === Separation = {s['separation']:+.3f}")
    print("Random embeddings show no modality-specific structure (expected).")
    print("Real CLIP-aligned encoders should give separation ~0.3-0.5,")
    print("justifying ELF Experimental #2 (shared embedding + per-modality head).")
    measure()


if __name__ == "__main__":
    main()
