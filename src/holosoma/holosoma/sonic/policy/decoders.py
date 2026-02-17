"""Decoders for SONIC universal policy.

SONIC uses:
- control decoder: produces actions for the robot
- motion decoder: reconstructs motion (auxiliary loss)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from holosoma.utils.safe_torch_import import torch


@dataclass
class ControlDecoderOutputs:
    action: torch.Tensor
    aux: Optional[dict[str, torch.Tensor]] = None


@dataclass
class MotionDecoderOutputs:
    recon: torch.Tensor
    aux: Optional[dict[str, torch.Tensor]] = None


class ControlDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *, tokens: torch.Tensor) -> ControlDecoderOutputs:
        raise NotImplementedError


class MotionDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *, tokens: torch.Tensor) -> MotionDecoderOutputs:
        raise NotImplementedError


class MotionDecoderMLP(MotionDecoder):
    """MLP motion decoder that reconstructs a robot motion command."""

    def __init__(
        self,
        *,
        token_dim: int,
        recon_dim: int,
        hidden_dims: list[int],
    ):
        super().__init__()
        dims = [token_dim, *hidden_dims, recon_dim]
        layers: list[torch.nn.Module] = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers.append(torch.nn.Linear(in_d, out_d))
            if out_d != recon_dim:
                layers.append(torch.nn.ELU())
        self.net = torch.nn.Sequential(*layers)

    def forward(self, *, tokens: torch.Tensor) -> MotionDecoderOutputs:
        return MotionDecoderOutputs(recon=self.net(tokens), aux=None)


class ControlDecoderMLP(ControlDecoder):
    """Minimal MLP control decoder used for PPO wiring (v0)."""

    def __init__(
        self,
        *,
        token_dim: int,
        action_dim: int,
        hidden_dims: list[int],
    ):
        super().__init__()
        dims = [token_dim, *hidden_dims, action_dim]
        layers: list[torch.nn.Module] = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers.append(torch.nn.Linear(in_d, out_d))
            if out_d != action_dim:
                layers.append(torch.nn.ELU())
        self.net = torch.nn.Sequential(*layers)

    def forward(self, *, tokens: torch.Tensor) -> ControlDecoderOutputs:
        return ControlDecoderOutputs(action=self.net(tokens), aux=None)
