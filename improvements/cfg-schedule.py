"""Scheduled classifier-free guidance vs constant CFG, on a toy 2D Gaussian mixture.

Implements §"Scheduled CFG" of 05-improvements.tex. A flow-matching model maps
noise to a 3-mode 2D mixture; three CFG strategies are compared at inference:
(a) fixed omega=3, (b) U-shape, (c) linear ramp. For each we report mean
log-density (quality) and 2D-histogram entropy (diversity). CPU-runnable, <60s.
"""
import math, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
D, HID, EPOCHS, BS, LR, N_STEPS, N_SAMPLES = 2, 64, 600, 256, 3e-3, 32, 400
MIX_MEANS = torch.tensor([[2.0, 0.0], [-1.0, 1.7], [-1.0, -1.7]])
MIX_STD = 0.35
MIX_WEIGHTS = torch.tensor([0.5, 0.3, 0.2])

def sample_mixture(n: int) -> torch.Tensor:
    idx = torch.multinomial(MIX_WEIGHTS, n, replacement=True)
    return MIX_MEANS[idx] + MIX_STD * torch.randn(n, D)

def mixture_logpdf(z: torch.Tensor) -> torch.Tensor:
    diff = z.unsqueeze(1) - MIX_MEANS.unsqueeze(0)
    log_norm = -0.5 * diff.pow(2).sum(-1) / MIX_STD**2 \
               - D * math.log(MIX_STD * math.sqrt(2 * math.pi))
    return torch.logsumexp(log_norm + torch.log(MIX_WEIGHTS + 1e-12).unsqueeze(0), dim=-1)

class Velocity(nn.Module):
    """Predicts z1 estimate given (z_t, t, c). c=1 cond, c=0 uncond (CFG dropout)."""
    def __init__(self, d: int, hid: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d + 2, hid), nn.SiLU(),
            nn.Linear(hid, hid), nn.SiLU(),
            nn.Linear(hid, d))
    def forward(self, z_t, t, c):
        return self.net(torch.cat([z_t, t.view(-1, 1), c.view(-1, 1)], dim=-1))

def train(model: Velocity, verbose: bool = True):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for epoch in range(EPOCHS):
        z1 = sample_mixture(BS); z0 = torch.randn_like(z1); t = torch.rand(BS)
        z_t = (1 - t.unsqueeze(-1)) * z0 + t.unsqueeze(-1) * z1
        c = (torch.rand(BS) > 0.2).float()  # 20% dropout → uncond
        loss = F.mse_loss(model(z_t, t, c), z1)
        opt.zero_grad(); loss.backward(); opt.step()
        if verbose and (epoch + 1) % 200 == 0:
            print(f"  epoch {epoch+1:3d}  mse={loss.item():.4f}")
    return model

@torch.no_grad()
def sample(model: Velocity, n: int, omega_fn):
    """v_cfg = omega * v_cond + (1-omega) * v_uncond, v ≈ pred_z1 - z_t."""
    z = torch.randn(n, D); dt, t = 1.0 / N_STEPS, 0.0
    one, zero = torch.ones(n), torch.zeros(n)
    for _ in range(N_STEPS):
        tt = torch.full((n,), t)
        v_cond = model(z, tt, one) - z
        v_uncond = model(z, tt, zero) - z
        omega = omega_fn(t)
        z = z + (omega * v_cond + (1 - omega) * v_uncond) * dt
        t += dt
    return z

def entropy_2d(z, bins=20, lo=-5.0, hi=5.0):
    h, _ = torch.histogramdd(z, bins=bins, range=[lo, hi, lo, hi])
    p = (h / h.sum().clamp_min(1)).flatten(); p = p[p > 0]
    return float(-(p * torch.log(p)).sum().item())

STRATEGIES = {
    "fixed":       (lambda t: 3.0,                                     "fixed omega = 3.0"),
    "u_shape":     (lambda t: 3.0 * (2 * t * (1 - t) + 0.01) ** -0.5,  "U-shape: omega(t)=3*(2t(1-t)+.01)^-.5"),
    "linear_ramp": (lambda t: 1.0 + 4.0 * t,                           "linear ramp: omega(t)=1+4t"),
}
_CACHE: dict = {}

def _eval_all(verbose: bool = False) -> dict:
    if "res" in _CACHE: return _CACHE["res"]
    torch.manual_seed(0)
    model = train(Velocity(D, HID), verbose=verbose)
    if verbose:
        print(f"\n{'strategy':>40}  {'mean log p':>12}  {'entropy':>9}")
        print(f"{'-'*40}  {'-'*12}  {'-'*9}")
    out = {}
    for key, (fn, label) in STRATEGIES.items():
        zs = sample(model, N_SAMPLES, fn)
        out[key] = (mixture_logpdf(zs).mean().item(), entropy_2d(zs))
        if verbose:
            print(f"{label:>40}  {out[key][0]:>12.3f}  {out[key][1]:>9.3f}")
    _CACHE["res"] = out
    return out

def measure() -> dict:
    """Quantitative comparison: fixed vs scheduled CFG on toy 2D mixture."""
    r = _eval_all(verbose=False)
    f_lp, f_ent = r["fixed"]; u_lp, u_ent = r["u_shape"]; l_lp, l_ent = r["linear_ramp"]
    scores = {"fixed": f_lp, "u_shape": u_lp, "linear_ramp": l_lp}
    return {
        "n_samples": N_SAMPLES, "n_steps": N_STEPS, "fixed_omega": 3.0,
        "fixed_log_p_mean": f_lp, "fixed_entropy_2d": f_ent,
        "u_shape_log_p_mean": u_lp, "u_shape_entropy_2d": u_ent,
        "linear_ramp_log_p_mean": l_lp, "linear_ramp_entropy_2d": l_ent,
        "best_strategy": max(scores, key=scores.get),
        "u_shape_quality_uplift": u_lp - f_lp,
        "claim_supported": u_lp > f_lp,
    }

def main():
    print(f"Setup: 2D Gaussian mixture, {N_STEPS} steps, {N_SAMPLES} samples/strategy\n"
          "Training velocity model with CFG dropout…")
    _eval_all(verbose=True)
    print("\nmeasure():")
    for k, v in measure().items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    sys.exit(main())
