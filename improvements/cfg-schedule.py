"""Scheduled classifier-free guidance vs constant CFG, on a toy 2D Gaussian mixture.

Implements §"Scheduled CFG" of 05-improvements.tex.

A simple flow-matching model is trained to map noise to a 2D Gaussian-mixture
target (3 modes). At inference, three CFG strategies are compared:
  (a) fixed omega = 3
  (b) U-shape schedule: omega(t) = omega_max * (2 t (1-t))^(-beta)
  (c) linear ramp:   omega(t) = 1 + 4t

For each strategy, generate samples and report:
  - mean log-density of generated samples under the true mixture (quality)
  - sample entropy via 2D-histogram estimate (diversity)

CPU-runnable, <60s.
"""

import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

D = 2
HID = 64
EPOCHS = 600
BS = 256
LR = 3e-3
N_STEPS = 32
N_SAMPLES = 400

# Gaussian-mixture target: 3 modes at fixed locations
MIX_MEANS = torch.tensor([[2.0, 0.0], [-1.0, 1.7], [-1.0, -1.7]])
MIX_STD = 0.35
MIX_WEIGHTS = torch.tensor([0.5, 0.3, 0.2])


def sample_mixture(n: int) -> torch.Tensor:
    idx = torch.multinomial(MIX_WEIGHTS, n, replacement=True)
    means = MIX_MEANS[idx]
    return means + MIX_STD * torch.randn(n, D)


def mixture_logpdf(z: torch.Tensor) -> torch.Tensor:
    # log sum_k w_k N(z; mu_k, sigma^2 I)
    z = z.unsqueeze(1)                      # (n, 1, D)
    diff = z - MIX_MEANS.unsqueeze(0)       # (n, K, D)
    log_norm = -0.5 * (diff.pow(2).sum(-1)) / MIX_STD**2 \
               - D * math.log(MIX_STD * math.sqrt(2 * math.pi))
    log_w = torch.log(MIX_WEIGHTS + 1e-12)
    return torch.logsumexp(log_norm + log_w.unsqueeze(0), dim=-1)


class Velocity(nn.Module):
    def __init__(self, d: int, hid: int, n_cond: int = 1):
        """Predicts x-prediction (z1 estimate) given (z_t, t, c).

        c=1 → conditional, c=0 → unconditional. The model is dropout-conditioned
        during training so it learns both regimes.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d + 1 + n_cond, hid),
            nn.SiLU(),
            nn.Linear(hid, hid),
            nn.SiLU(),
            nn.Linear(hid, d),
        )

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, c: torch.Tensor):
        t_feat = t.view(-1, 1)
        c_feat = c.view(-1, 1)
        return self.net(torch.cat([z_t, t_feat, c_feat], dim=-1))


def train(model: Velocity):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for epoch in range(EPOCHS):
        z1 = sample_mixture(BS)
        z0 = torch.randn_like(z1)
        t = torch.rand(BS)
        z_t = (1 - t.unsqueeze(-1)) * z0 + t.unsqueeze(-1) * z1

        # 20% dropout to teach unconditional mode
        c = (torch.rand(BS) > 0.2).float()

        pred_z1 = model(z_t, t, c)
        loss = F.mse_loss(pred_z1, z1)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if (epoch + 1) % 200 == 0:
            print(f"  epoch {epoch+1:3d}  mse={loss.item():.4f}")
    return model


@torch.no_grad()
def sample(model: Velocity, n: int, omega_fn):
    """omega_fn: callable t -> omega.

    At each step compute v_cfg = omega * v_cond + (1-omega) * v_uncond, where
    v is approximated as (pred_z1 - z_t) (the rectified-flow shortcut).
    """
    z = torch.randn(n, D)
    dt = 1.0 / N_STEPS
    t = 0.0
    for _ in range(N_STEPS):
        t_tensor = torch.full((n,), t)
        c_one = torch.ones(n)
        c_zero = torch.zeros(n)

        pred_cond = model(z, t_tensor, c_one)
        pred_uncond = model(z, t_tensor, c_zero)

        v_cond = pred_cond - z
        v_uncond = pred_uncond - z

        omega = omega_fn(t)
        v_cfg = omega * v_cond + (1 - omega) * v_uncond

        z = z + v_cfg * dt
        t += dt
    return z


def entropy_2d(z: torch.Tensor, bins: int = 20, lo: float = -5, hi: float = 5):
    h, _ = torch.histogramdd(z, bins=bins, range=[lo, hi, lo, hi])
    p = h / h.sum().clamp_min(1)
    p = p.flatten()
    p = p[p > 0]
    return float(-(p * torch.log(p)).sum().item())


def main():
    print(f"Setup: 2D Gaussian mixture, {N_STEPS} sampling steps, "
          f"{N_SAMPLES} samples per strategy")
    print()

    print("Training velocity model with CFG dropout…")
    model = train(Velocity(D, HID))
    print()

    strategies = {
        "fixed omega = 3.0":
            lambda t: 3.0,
        "U-shape: omega(t) = 3 * (2t(1-t)+0.01)^-0.5":
            lambda t: 3.0 * (2 * t * (1 - t) + 0.01) ** -0.5,
        "linear ramp: omega(t) = 1 + 4t":
            lambda t: 1.0 + 4.0 * t,
    }

    print(f"{'strategy':>48}  {'mean log p':>12}  {'entropy':>9}")
    print(f"{'-'*48}  {'-'*12}  {'-'*9}")
    results = []
    for name, fn in strategies.items():
        zs = sample(model, N_SAMPLES, fn)
        logp = mixture_logpdf(zs).mean().item()
        ent = entropy_2d(zs)
        results.append((name, logp, ent))
        print(f"{name:>48}  {logp:>12.3f}  {ent:>9.3f}")

    print()
    fixed = results[0]
    u_shape = results[1]
    if u_shape[1] > fixed[1]:
        print(
            "U-shape schedule beat fixed omega on quality "
            f"({u_shape[1]:.3f} > {fixed[1]:.3f})"
        )
    else:
        print(
            "U-shape did not beat fixed omega on quality on this run; "
            "schedule shape may need tuning."
        )


if __name__ == "__main__":
    sys.exit(main())
