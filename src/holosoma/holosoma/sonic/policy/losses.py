"""Auxiliary losses for SONIC reproduction.

SONIC total objective (high-level):
    L = L_ppo + L_recon + L_token + L_cycle

This file defines the loss function interfaces and containers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from holosoma.utils.safe_torch_import import torch


@dataclass
class SonicAuxLossWeights:
    recon: float = 1.0
    token: float = 1.0
    cycle: float = 1.0


@dataclass
class SonicAuxLossOutputs:
    total: torch.Tensor
    terms: dict[str, torch.Tensor]


def compute_sonic_aux_losses(
    *,
    recon: Optional[torch.Tensor] = None,
    recon_target: Optional[torch.Tensor] = None,
    tokens: Optional[torch.Tensor] = None,
    tokens_target: Optional[torch.Tensor] = None,
    cycle: Optional[torch.Tensor] = None,
    cycle_target: Optional[torch.Tensor] = None,
    weights: SonicAuxLossWeights = SonicAuxLossWeights(),
) -> SonicAuxLossOutputs:
    """Compute aux losses.

    This is a placeholder implementation that returns zero losses when
    corresponding tensors are missing.
    """

    device = None
    for t in (recon, recon_target, tokens, tokens_target, cycle, cycle_target):
        if isinstance(t, torch.Tensor):
            device = t.device
            break

    if device is None:
        device = torch.device("cpu")

    zero = torch.zeros((), device=device)
    terms: dict[str, torch.Tensor] = {}
    total = zero

    if recon is not None and recon_target is not None:
        loss = torch.mean((recon - recon_target) ** 2)
        terms["aux/recon"] = loss
        total = total + weights.recon * loss

    if tokens is not None and tokens_target is not None:
        loss = torch.mean((tokens - tokens_target) ** 2)
        terms["aux/token"] = loss
        total = total + weights.token * loss

    if cycle is not None and cycle_target is not None:
        loss = torch.mean((cycle - cycle_target) ** 2)
        terms["aux/cycle"] = loss
        total = total + weights.cycle * loss

    if not terms:
        terms["aux/none"] = zero

    return SonicAuxLossOutputs(total=total, terms=terms)
