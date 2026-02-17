#!/usr/bin/env python
# pyright: reportMissingImports=false
"""Smoke-test: PPO integrates SONIC Sec. 3.2 auxiliary losses into actor_loss.

This does NOT require a simulator. It directly calls `PPO._compute_ppo_loss`
with a lightweight fake `self` object that provides the minimal attributes
and an actor that wraps the real SonicUniversal adapter.

Pass criteria:
- output dict contains `sonic_recon_loss/sonic_token_loss/sonic_cycle_loss`
- `actor_loss ≈ base_actor_loss + sum(sonic_*_loss)`

Run:
  python scripts/smoke_ppo_sonic_aux_losses.py
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

try:
    from holosoma.agents.ppo.ppo import PPO
    from holosoma.sonic.policy.ppo_adapter import build_sonic_universal_actor_mean
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src" / "holosoma"))
    from holosoma.agents.ppo.ppo import PPO
    from holosoma.sonic.policy.ppo_adapter import build_sonic_universal_actor_mean


@dataclass
class _PPOCfg:
    clip_param: float = 0.2
    entropy_coef: float = 0.005
    symmetry_actor_coef: float = 0.0
    symmetry_critic_coef: float = 0.0
    value_loss_coef: float = 1.0
    desired_kl: float | None = 0.01
    schedule: str = "adaptive"


class _ActorWrapper:
    """Minimal actor API required by PPO._compute_ppo_loss."""

    def __init__(self, module: torch.nn.Module, *, action_dim: int, device: torch.device):
        self.actor_module = type("ActorModule", (), {"module": module})()
        self._action_dim = int(action_dim)
        self._device = device

        self.action_mean: torch.Tensor | None = None
        self.action_std: torch.Tensor | None = None
        self.entropy: torch.Tensor | None = None

    def act(self, obs_dict: dict[str, torch.Tensor]):
        actor_obs = obs_dict["actor_obs"]
        mu = self.actor_module.module(actor_obs)
        std = torch.full_like(mu, 0.6)
        dist = torch.distributions.Normal(mu, std)

        # PPO reads these tensors after calling act()
        self.action_mean = mu
        self.action_std = std
        # Entropy per-sample (sum over action dims), keep shape (B, 1)
        self.entropy = dist.entropy().sum(dim=-1, keepdim=True)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        assert self.action_mean is not None
        assert self.action_std is not None
        dist = torch.distributions.Normal(self.action_mean, self.action_std)
        # Return shape (B,) to match PPO math (it squeezes old log prob)
        return dist.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.actor_module.module(obs_dict["actor_obs"])


class _CriticWrapper:
    """Minimal critic API required by PPO._compute_ppo_loss."""

    def __init__(self, *, obs_dim: int, device: torch.device):
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 1),
        ).to(device)

    def evaluate(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.net(obs_dict["critic_obs"])


class _FakePPO:
    """Holds attributes/methods expected by PPO._compute_ppo_loss."""

    def __init__(self, *, actor, critic, device: torch.device):
        self.actor = actor
        self.critic = critic
        self.device = device

        self.config = _PPOCfg()
        self.use_symmetry = False
        self.is_multi_gpu = False
        self.gpu_world_size = 1

    # Reuse PPO's KL implementation.
    _compute_kl_div = PPO._compute_kl_div

    def _update_learning_rate(self, kl_mean: torch.Tensor):
        # no-op for smoke test
        _ = kl_mean


def main() -> int:
    torch.manual_seed(0)
    device = torch.device("cpu")

    # Bundle dims
    B = 64
    dr, dh, dm = 32, 48, 64
    proprio_dim = 32
    action_dim = 8

    # Build a real SonicUniversal actor mean module.
    sonic_module = build_sonic_universal_actor_mean(
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

    # Create a mixed-command batch.
    header = torch.zeros(B, 6, device=device)
    header[:, 0] = float(dr)
    header[:, 1] = float(dh)
    header[:, 2] = float(dm)
    # cycle through robot/human/hybrid
    for i in range(B):
        header[i, 3 + (i % 3)] = 1.0

    g_r = torch.randn(B, dr, device=device)
    g_h = torch.randn(B, dh, device=device)
    g_m = torch.randn(B, dm, device=device)
    proprio = torch.randn(B, proprio_dim, device=device)
    actor_obs = torch.cat([header, g_r, g_h, g_m, proprio], dim=1)

    # Critic obs can be same shape for smoke.
    critic_obs = actor_obs.detach().clone()

    actor = _ActorWrapper(sonic_module, action_dim=action_dim, device=device)
    critic = _CriticWrapper(obs_dim=critic_obs.shape[1], device=device)
    ppo = _FakePPO(actor=actor, critic=critic, device=device)

    # Produce a consistent old policy snapshot.
    actor.act({"actor_obs": actor_obs})
    mu = actor.action_mean.detach()
    sigma = actor.action_std.detach()
    dist = torch.distributions.Normal(mu, sigma)
    actions = dist.sample()
    old_logp = dist.log_prob(actions).sum(dim=-1, keepdim=True).detach()

    minibatch = {
        "actions": actions,
        "values": torch.zeros(B, 1, device=device),
        "advantages": torch.randn(B, 1, device=device) * 0.1,
        "returns": torch.zeros(B, 1, device=device),
        "actions_log_prob": old_logp,
        "action_mean": mu,
        "action_sigma": sigma,
        "actor_obs": actor_obs,
        "critic_obs": critic_obs,
    }

    out = PPO._compute_ppo_loss(ppo, minibatch)

    for k in ["sonic_recon_loss", "sonic_token_loss", "sonic_cycle_loss"]:
        if k not in out:
            raise SystemExit(f"missing {k} in PPO loss dict")

    aux_total = out["sonic_recon_loss"] + out["sonic_token_loss"] + out["sonic_cycle_loss"]
    base_actor_loss = out["surrogate_loss"] - ppo.config.entropy_coef * out["entropy_loss"]
    expected = base_actor_loss + aux_total

    diff = (out["actor_loss"] - expected).abs().item()
    print(
        {
            "actor_loss": float(out["actor_loss"].detach().cpu()),
            "base_actor_loss": float(base_actor_loss.detach().cpu()),
            "aux_total": float(aux_total.detach().cpu()),
            "abs_diff": diff,
        }
    )

    if not math.isfinite(diff) or diff > 1e-5:
        raise SystemExit(f"actor_loss does not include aux losses as expected (abs_diff={diff})")

    print("smoke_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
