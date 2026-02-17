#!/usr/bin/env python
# pyright: reportMissingImports=false
"""Smoke-test SONIC Sec. 3.2 auxiliary losses without a simulator.

This script optimizes the SonicUniversal actor adapter on synthetic data using
only the auxiliary losses (recon/token/cycle). It is intentionally minimal and
CPU-friendly, intended to quickly validate:
- forward() works for robot/human/hybrid command types
- get_aux_losses() returns all three losses
- losses can backpropagate and decrease after a few optimizer steps

Run:
    python scripts/smoke_sonic_aux_train.py --iters 200 --lr 1e-3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    from holosoma.sonic.policy.ppo_adapter import build_sonic_universal_actor_mean
except ModuleNotFoundError:
    # Allow running from a source checkout without installing holosoma.
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src" / "holosoma"))
    from holosoma.sonic.policy.ppo_adapter import build_sonic_universal_actor_mean


def _make_actor_obs(
    *,
    batch_size: int,
    dr: int,
    dh: int,
    dm: int,
    proprio_dim: int,
    cmd_type: int,
    device: torch.device,
) -> torch.Tensor:
    header = torch.zeros(batch_size, 6, device=device)
    header[:, 0] = float(dr)
    header[:, 1] = float(dh)
    header[:, 2] = float(dm)
    header[:, 3 + cmd_type] = 1.0

    # Synthetic commands:
    # - make g_h correlated with g_r to give token/cycle losses a learnable signal
    # - g_m is another correlated view (different dimension)
    g_r = torch.randn(batch_size, dr, device=device)
    g_h = torch.randn(batch_size, dh, device=device)
    shared = g_r.mean(dim=1, keepdim=True)
    g_h[:, : min(dh, 1)] = shared[:, : min(dh, 1)]

    g_m = torch.randn(batch_size, dm, device=device)
    g_m[:, : min(dm, 1)] = shared[:, : min(dm, 1)]

    proprio = torch.randn(batch_size, proprio_dim, device=device)
    return torch.cat([header, g_r, g_h, g_m, proprio], dim=1)


@torch.no_grad()
def _eval_losses(actor: torch.nn.Module, *, device: torch.device) -> dict[str, float]:
    actor.eval()
    out: dict[str, float] = {}
    for cmd_type, name in [(0, "robot"), (1, "human"), (2, "hybrid")]:
        obs = _make_actor_obs(
            batch_size=128,
            dr=32,
            dh=48,
            dm=64,
            proprio_dim=32,
            cmd_type=cmd_type,
            device=device,
        )
        _ = actor(obs)
        aux = actor.get_aux_losses()
        out[f"{name}/sonic_recon_loss"] = float(aux["sonic_recon_loss"].detach().cpu())
        out[f"{name}/sonic_token_loss"] = float(aux["sonic_token_loss"].detach().cpu())
        out[f"{name}/sonic_cycle_loss"] = float(aux["sonic_cycle_loss"].detach().cpu())
        out[f"{name}/sonic_aux_total"] = float(sum(aux.values()).detach().cpu())
    actor.train()
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    # Reference-aligned FSQ can require a bit more optimization steps than the
    # legacy uniform quantizer; keep this smoke test stable by default.
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    # Fixed dims to match the bundle parser contract.
    dr, dh, dm = 32, 48, 64
    proprio_dim = 32
    action_dim = 8

    actor = build_sonic_universal_actor_mean(
        input_dim=6 + dr + dh + dm + proprio_dim,
        output_dim=action_dim,
        hidden_dims=[256, 256],
        latent_dim=128,
        fsq_enabled=True,
        fsq_levels=8,
        recon_coef=1.0,
        token_coef=1.0,
        cycle_coef=1.0,
    ).to(device)

    # The SonicUniversal adapter lazily builds its internal modules on the first
    # forward pass (after it reads dr/dh/dm from the bundle header). Force that
    # build before creating the optimizer.
    _ = actor(
        _make_actor_obs(
            batch_size=2,
            dr=dr,
            dh=dh,
            dm=dm,
            proprio_dim=proprio_dim,
            cmd_type=0,
            device=device,
        )
    )

    opt = torch.optim.Adam(actor.parameters(), lr=args.lr)

    before = _eval_losses(actor, device=device)
    print("before", before)

    actor.train()
    for step in range(args.iters):
        # Mix command types in the batch.
        cmd_type = step % 3
        obs = _make_actor_obs(
            batch_size=256,
            dr=dr,
            dh=dh,
            dm=dm,
            proprio_dim=proprio_dim,
            cmd_type=cmd_type,
            device=device,
        )

        _ = actor(obs)
        aux = actor.get_aux_losses()
        aux_total = sum(aux.values())

        opt.zero_grad(set_to_none=True)
        aux_total.backward()
        opt.step()

    after = _eval_losses(actor, device=device)
    print("after", after)

    # Simple success criterion: averaged aux total decreases.
    before_avg = (before["robot/sonic_aux_total"] + before["human/sonic_aux_total"] + before["hybrid/sonic_aux_total"]) / 3.0
    after_avg = (after["robot/sonic_aux_total"] + after["human/sonic_aux_total"] + after["hybrid/sonic_aux_total"]) / 3.0
    print("avg_total", {"before": before_avg, "after": after_avg})

    # Avoid being too strict; we mainly want to catch wiring/backprop issues.
    if not (after_avg < before_avg):
        raise SystemExit(
            f"Smoke test failed: aux total did not decrease (before={before_avg:.6f}, after={after_avg:.6f})."
        )

    print("smoke_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
