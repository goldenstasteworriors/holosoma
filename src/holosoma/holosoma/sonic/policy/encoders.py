"""Encoders for SONIC universal token space.

SONIC describes separate encoders for robot/human/hybrid command modalities.
This file defines the interfaces and default minimal implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from holosoma.utils.safe_torch_import import torch


@dataclass
class EncoderOutputs:
    """Latent representation returned by encoders."""

    z: torch.Tensor  # (B, Dz) or (B, F, Dz)
    aux: dict[str, torch.Tensor] | None = None


class CommandEncoder(Protocol):
    """A command encoder maps command features -> latent z."""

    def __call__(self, command_features: torch.Tensor) -> EncoderOutputs:  # pragma: no cover
        ...


class RobotCommandEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, command_features: torch.Tensor) -> EncoderOutputs:
        raise NotImplementedError


class HumanCommandEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, command_features: torch.Tensor) -> EncoderOutputs:
        raise NotImplementedError


class HybridCommandEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, command_features: torch.Tensor) -> EncoderOutputs:
        raise NotImplementedError


class HumanCommandEncoderMLP(HumanCommandEncoder):
    """MLP encoder for human command features."""

    def __init__(self, *, command_dim: int, latent_dim: int, hidden_dims: list[int]):
        super().__init__()
        dims = [command_dim, *hidden_dims, latent_dim]
        layers: list[torch.nn.Module] = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers.append(torch.nn.Linear(in_d, out_d))
            if out_d != latent_dim:
                layers.append(torch.nn.ELU())
        self.net = torch.nn.Sequential(*layers)

    def forward(self, command_features: torch.Tensor) -> EncoderOutputs:
        return EncoderOutputs(z=self.net(command_features), aux=None)


class HybridCommandEncoderMLP(HybridCommandEncoder):
    """MLP encoder for hybrid command features."""

    def __init__(self, *, command_dim: int, latent_dim: int, hidden_dims: list[int]):
        super().__init__()
        dims = [command_dim, *hidden_dims, latent_dim]
        layers: list[torch.nn.Module] = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers.append(torch.nn.Linear(in_d, out_d))
            if out_d != latent_dim:
                layers.append(torch.nn.ELU())
        self.net = torch.nn.Sequential(*layers)

    def forward(self, command_features: torch.Tensor) -> EncoderOutputs:
        return EncoderOutputs(z=self.net(command_features), aux=None)


class RobotCommandEncoderMLP(RobotCommandEncoder):
    """Minimal MLP encoder used for PPO wiring (v0)."""

    def __init__(self, *, command_dim: int, latent_dim: int, hidden_dims: list[int]):
        super().__init__()
        dims = [command_dim, *hidden_dims, latent_dim]
        layers: list[torch.nn.Module] = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers.append(torch.nn.Linear(in_d, out_d))
            if out_d != latent_dim:
                layers.append(torch.nn.ELU())
        self.net = torch.nn.Sequential(*layers)

    def forward(self, command_features: torch.Tensor) -> EncoderOutputs:
        return EncoderOutputs(z=self.net(command_features), aux=None)
