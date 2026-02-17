"""Universal SONIC policy skeleton.

This module is intentionally NOT wired into PPO yet.
It defines the forward interface we will later integrate into the existing
actor-critic code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from holosoma.sonic.commands.multi_command import SonicCommandType
from holosoma.sonic.data.batch import SonicPolicyInputs, SonicPolicyOutputs
from holosoma.sonic.policy.decoders import ControlDecoder, MotionDecoder
from holosoma.sonic.policy.encoders import HybridCommandEncoder, HumanCommandEncoder, RobotCommandEncoder
from holosoma.sonic.policy.fsq import FSQQuantizer
from holosoma.utils.safe_torch_import import torch


@dataclass
class UniversalPolicyModules:
    robot_encoder: RobotCommandEncoder
    human_encoder: HumanCommandEncoder
    hybrid_encoder: HybridCommandEncoder
    quantizer: FSQQuantizer
    control_decoder: ControlDecoder
    motion_decoder: Optional[MotionDecoder] = None


class UniversalSonicPolicy(torch.nn.Module):
    """Universal policy = encoder -> quantizer -> (control decoder + motion decoder)."""

    def __init__(self, modules: UniversalPolicyModules):
        super().__init__()
        self.robot_encoder = modules.robot_encoder
        self.human_encoder = modules.human_encoder
        self.hybrid_encoder = modules.hybrid_encoder
        self.quantizer = modules.quantizer
        self.control_decoder = modules.control_decoder
        self.motion_decoder = modules.motion_decoder

    def forward(self, inputs: SonicPolicyInputs) -> SonicPolicyOutputs:
        cmd_type: SonicCommandType = inputs.command.command_type
        features = inputs.command.features

        if cmd_type == "robot":
            z = self.robot_encoder(features).z
        elif cmd_type == "human":
            z = self.human_encoder(features).z
        elif cmd_type == "hybrid":
            z = self.hybrid_encoder(features).z
        else:
            raise ValueError(f"Unknown command type: {cmd_type}")

        q = self.quantizer(z)
        ctrl = self.control_decoder(tokens=q.tokens)

        recon = None
        if self.motion_decoder is not None:
            recon = self.motion_decoder(tokens=q.tokens).recon

        return SonicPolicyOutputs(action=ctrl.action, tokens=q.tokens, recon=recon, aux={})
