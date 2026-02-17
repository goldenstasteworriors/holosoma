"""Batch/dataclass definitions for SONIC-style policies.

These are *model-facing* containers (not replay buffers).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from holosoma.sonic.commands.multi_command import SonicCommandWindow
from holosoma.utils.safe_torch_import import torch


@dataclass
class SonicPolicyInputs:
    """Inputs to the universal SONIC policy.

    Expected conventions (to be finalized later):
    - proprio: (B, Dp)
    - command: (B, Dc) or (B, F, Dc)
    """

    proprio: torch.Tensor
    command: SonicCommandWindow
    prev_action: Optional[torch.Tensor] = None


@dataclass
class SonicPolicyOutputs:
    """Outputs from the universal SONIC policy forward pass."""

    action: torch.Tensor
    tokens: Optional[torch.Tensor] = None
    recon: Optional[torch.Tensor] = None
    aux: Optional[dict[str, torch.Tensor]] = None
