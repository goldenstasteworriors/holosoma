"""Multi-command construction utilities.

SONIC tracker needs to support multiple command modalities (robot/human/hybrid)
and optionally future-frame windows.

This file defines the data structures + interfaces only; actual wiring happens
in the SONIC task's command manager term.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from holosoma.utils.safe_torch_import import torch


SonicCommandType = Literal["robot", "human", "hybrid"]


@dataclass(frozen=True)
class SonicWindowSpec:
    """Defines a future window spec.

    - num_frames: number of frames in the window (e.g., 10)
    - dt_s: temporal stride in seconds (e.g., 0.1)
    """

    num_frames: int
    dt_s: float


@dataclass
class SonicCommandWindow:
    """A generic container for (possibly windowed) command features.

    Shapes are intentionally left flexible at this stage.
    """

    command_type: SonicCommandType
    features: torch.Tensor  # (B, F, D) or (B, D)
    mask: Optional[torch.Tensor] = None  # (B, F) optional validity mask

    def flatten(self) -> torch.Tensor:
        """Flatten to (B, -1) for legacy MLP-style policies."""

        if self.features.ndim == 2:
            return self.features
        if self.features.ndim == 3:
            b, f, d = self.features.shape
            return self.features.reshape(b, f * d)
        raise ValueError(f"Unsupported features shape: {tuple(self.features.shape)}")


def build_sonic_command_window(
    *,
    command_type: SonicCommandType,
    robot_features: Optional[torch.Tensor] = None,
    human_features: Optional[torch.Tensor] = None,
    hybrid_features: Optional[torch.Tensor] = None,
) -> SonicCommandWindow:
    """Build a `SonicCommandWindow` from available modality features.

    Interface contract (to be enforced later):
    - When command_type=="robot", robot_features must be provided.
    - When command_type=="human", human_features must be provided.
    - When command_type=="hybrid", hybrid_features must be provided.
    """

    if command_type == "robot":
        if robot_features is None:
            raise ValueError("robot_features is required for command_type='robot'")
        return SonicCommandWindow(command_type=command_type, features=robot_features)
    if command_type == "human":
        if human_features is None:
            raise ValueError("human_features is required for command_type='human'")
        return SonicCommandWindow(command_type=command_type, features=human_features)
    if command_type == "hybrid":
        if hybrid_features is None:
            raise ValueError("hybrid_features is required for command_type='hybrid'")
        return SonicCommandWindow(command_type=command_type, features=hybrid_features)
    raise ValueError(f"Unknown command_type: {command_type}")
