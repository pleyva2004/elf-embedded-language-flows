"""Toy probe of ELF's weight-tied unembedding for final-step discretization.

Trains a tiny MLP to predict z_1 (clean embedding) from z_t along a linear
flow path. At inference, Euler-integrates and decodes via argmax(W_E @ z).

CPU-runnable, <60s.
"""

import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
random.seed(0)

V = 64          # vocabulary size
D = 32          # embedding dim
HID = 64        # MLP hidden width
EPOCHS = 300
BS = 256
LR = 3e-3
STEP_COUNTS = [1, 4, 16, 64]


def make_embeddings(V: int, D: int) -> torch.Tensor:
    """Random unit-norm token embeddings."""
    W = torch.randn(V, D)
    return F.normalize(W, dim=-1)


class Velocity(nn.Module):
    def __init__(self, d: int, hid: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d + 1, hid),    # +1 for time embedding
            nn.SiLU(),
            nn.Linear(hid, hid),
            nn.SiLU(),
            nn.Linear(hid, d),
        )

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Predict the clean embedding z_1 directly (x-prediction)
        t_feat = t.view(-1, 1)
        return self.net(torch.cat([z_t, t_feat], dim=-1))


def train(W_E: torch.Tensor) -> Velocity:
    model = Velocity(D, HID)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        # Sample tokens, embeddings, noise, time
        v = torch.randint(0, V, (BS,))
        z1 = W_E[v]                           # (BS, D)
        z0 = torch.randn_like(z1)             # (BS, D)
        t = torch.rand(BS)                    # (BS,)
        z_t = (1 - t.unsqueeze(-1)) * z0 + t.unsqueeze(-1) * z1

        pred_z1 = model(z_t, t)
        loss = F.mse_loss(pred_z1, z1)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if (epoch + 1) % 100 == 0:
            print(f"  epoch {epoch+1:3d}  mse={loss.item():.4f}")

    return model


@torch.no_grad()
def euler_sample(model: Velocity, W_E: torch.Tensor, n_steps: int,
                 n_samples: int = 200) -> torch.Tensor:
    """Sample z_0, integrate to t=1, return predicted token ids."""
    z = torch.randn(n_samples, D)
    z0 = z.clone()

    # Use x-prediction: at each step compute velocity = (pred_z1 - z0) so the
    # path remains linear toward the model's prediction. Or equivalently
    # just step toward pred_z1.
    dt = 1.0 / n_steps
    t = 0.0
    for _ in range(n_steps):
        t_tensor = torch.full((n_samples,), t)
        pred_z1 = model(z, t_tensor)
        # Linear path velocity: v = (pred_z1 - z0) / (1 - t), but since we
        # don't have z0 at inference, use the rectified-flow shortcut:
        # v = pred_z1 - z (move toward the current prediction)
        v = pred_z1 - z
        z = z + v * dt
        t += dt

    # Final-step decoding: argmax over W_E @ z
    logits = z @ W_E.T  # (n_samples, V)
    return logits.argmax(dim=-1)


def main():
    print(f"Setup: V={V} tokens, D={D}-dim embeddings, HID={HID} MLP")
    print(f"Train: {EPOCHS} epochs of BS={BS}, LR={LR}")
    print()

    W_E = make_embeddings(V, D)

    # We don't have ground-truth target tokens at inference (the model is
    # unconditional). Instead, measure: for each predicted token, is it a
    # valid embedding (i.e. does decoding produce ANY token, and is the
    # distribution non-uniform)? Compare to argmax-of-random-z baseline.
    print("Training velocity MLP (x-prediction)…")
    model = train(W_E)
    print()

    print("Sampling + decoding at varying step counts:")
    print(f"{'N steps':>10}  {'token distribution entropy':>28}  "
          f"{'top-1 freq':>12}  {'unique tokens':>14}")
    print(f"{'-'*10}  {'-'*28}  {'-'*12}  {'-'*14}")

    # Baseline: argmax of pure random noise
    z_rand = torch.randn(500, D)
    base_ids = (z_rand @ W_E.T).argmax(dim=-1)
    base_counts = torch.bincount(base_ids, minlength=V).float()
    base_p = base_counts / base_counts.sum()
    base_ent = -(base_p * torch.log(base_p + 1e-12)).sum().item()
    print(f"{'BASELINE':>10}  {base_ent:>28.3f}  "
          f"{base_counts.max().item()/500:>12.3f}  "
          f"{(base_counts > 0).sum().item():>14d}")

    for N in STEP_COUNTS:
        ids = euler_sample(model, W_E, N, n_samples=500)
        counts = torch.bincount(ids, minlength=V).float()
        p = counts / counts.sum()
        ent = -(p * torch.log(p + 1e-12)).sum().item()
        top1 = counts.max().item() / 500
        unique = (counts > 0).sum().item()
        print(f"{N:>10d}  {ent:>28.3f}  {top1:>12.3f}  {unique:>14d}")

    print()
    print("Interpretation:")
    print(f"  Baseline (pure-noise argmax) entropy:  {base_ent:.3f}")
    print(f"  ELF-like sampling entropy at N=64:     measured above")
    print("  If the trained model produces SIMILAR coverage with comparable")
    print("  entropy at larger N, the weight-tied unembedding pathway is")
    print("  working — the network learned to land near valid embeddings.")
    print("  Note: this toy uses random embeddings; ELF's headline win comes")
    print("  from pretrained contextual embeddings (frozen).")


if __name__ == "__main__":
    sys.exit(main())
