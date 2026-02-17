"""SONIC-specific observation terms.

These terms extend Whole-Body Tracking (WBT) observations to carry the additional
information needed to reproduce SONIC Sec. 3.2 (universal token space):
- command type (robot/human/hybrid)
- synchronized command features for all modalities

Design goals:
- Minimal intrusion: keep WBT managers/reward/termination intact.
- Self-describing tensor layout: include modality dims in a small header so the
  policy-side adapter can parse without hard-coding robot DOF counts.

Layout returned by `motion_command_bundle`:
    [dr, dh, dm, onehot(3), robot_cmd(dr), human_cmd(dh), hybrid_cmd(dm)]

All values are float32; dims are encoded as exact integers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from holosoma.managers.command.terms.sonic import SonicMotionCommand

if TYPE_CHECKING:
    from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager


def _get_sonic_motion_command(env: WholeBodyTrackingManager) -> SonicMotionCommand:
    motion_command = env.command_manager.get_state("motion_command")
    assert motion_command is not None, "motion_command not found in command manager"
    assert isinstance(
        motion_command, SonicMotionCommand
    ), f"Expected SonicMotionCommand, got {type(motion_command)}"
    return motion_command


def motion_command_bundle(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Return SONIC command bundle for universal-policy training."""

    cmd = _get_sonic_motion_command(env)

    robot_cmd = cmd.robot_command
    human_cmd = cmd.human_command
    hybrid_cmd = cmd.hybrid_command

    b = robot_cmd.shape[0]
    dr = robot_cmd.shape[1]
    dh = human_cmd.shape[1]
    dm = hybrid_cmd.shape[1]

    header_dims = torch.tensor([dr, dh, dm], device=env.device, dtype=robot_cmd.dtype).view(1, 3).repeat(b, 1)

    onehot = torch.nn.functional.one_hot(cmd.command_type_id, num_classes=3).to(dtype=robot_cmd.dtype)

    return torch.cat([header_dims, onehot, robot_cmd, human_cmd, hybrid_cmd], dim=-1)
