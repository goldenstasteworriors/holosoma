"""SONIC-specific command terms.

The first milestone for SONIC reproduction is to provide a future-window motion
command (e.g., 10 frames) while keeping the rest of the WBT task intact.

We implement this as a subclass of WBT's MotionCommand so existing observation
terms, reward terms, and termination terms that assert `isinstance(MotionCommand)`
continue to work.
"""

from __future__ import annotations

from typing import Any

from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.utils.rotations import quat_rotate_inverse, yaw_quat
from holosoma.utils.safe_torch_import import torch


class SonicMotionCommand(MotionCommand):
    """Motion command that exposes a future window in the `command` property.

    Baseline behavior (reset/step, relative pose computation, metrics, etc.) is
    inherited from `MotionCommand`.

    The only change is how `command` is produced:
    - WBT: concat([joint_pos_t, joint_vel_t])
    - SONIC v0: concat over future window, then flatten to (B, F * D)
    """

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        params = getattr(cfg, "params", {}) or {}
        self.window_num_frames: int = int(params.get("window_num_frames", 10))
        self.window_dt_s: float = float(params.get("window_dt_s", 0.1))

        # SONIC (Sec. 3.2) also uses a human command stream (smaller dt) and a
        # hybrid stream (upper-body keypoints + lower-body robot motion).
        self.human_window_num_frames: int = int(params.get("human_window_num_frames", self.window_num_frames))
        self.human_window_dt_s: float = float(params.get("human_window_dt_s", 0.02))

        # Which bodies to use as a proxy for "human joints". We default to the
        # tracked bodies since they exist in the motion data and are sufficient
        # to reproduce the universal-token training pipeline.
        self.human_body_names: list[str] | None = params.get("human_body_names")
        self.hybrid_upper_body_names: list[str] = list(
            params.get(
                "hybrid_upper_body_names",
                ["torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link"],
            )
        )

        # Per-env command type sampling probabilities: [robot, human, hybrid].
        probs = params.get("command_type_probs")
        if probs is None:
            probs = [0.34, 0.33, 0.33]
        if not isinstance(probs, (list, tuple)) or len(probs) != 3:
            raise ValueError("command_type_probs must be a list/tuple of length 3")
        self.command_type_probs = torch.tensor(probs, dtype=torch.float32)

        if self.window_num_frames <= 0:
            raise ValueError("window_num_frames must be > 0")
        if self.window_dt_s <= 0.0:
            raise ValueError("window_dt_s must be > 0")
        if self.human_window_num_frames <= 0:
            raise ValueError("human_window_num_frames must be > 0")
        if self.human_window_dt_s <= 0.0:
            raise ValueError("human_window_dt_s must be > 0")

        self._window_stride_steps: int | None = None
        self._window_offsets: torch.Tensor | None = None

        self._human_stride_steps: int | None = None
        self._human_offsets: torch.Tensor | None = None

        self._human_body_indices: torch.Tensor | None = None
        self._hybrid_upper_body_indices: torch.Tensor | None = None

        # 0=robot, 1=human, 2=hybrid
        self.command_type_id: torch.Tensor | None = None

    def setup(self) -> None:
        super().setup()

        # Allocate command type buffer.
        self.command_type_id = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # Resolve body indices used for human/hybrid commands.
        robot_body_names = self._env.simulator._body_list  # type: ignore[attr-defined]

        if self.human_body_names is None:
            self.human_body_names = list(self.motion_cfg.body_names_to_track)

        missing = [bn for bn in self.human_body_names if bn not in robot_body_names]
        if missing:
            raise ValueError(f"human_body_names contains bodies not in simulator: {missing}")
        self._human_body_indices = torch.tensor(
            [robot_body_names.index(bn) for bn in self.human_body_names],
            device=self.device,
            dtype=torch.long,
        )

        missing = [bn for bn in self.hybrid_upper_body_names if bn not in robot_body_names]
        if missing:
            raise ValueError(f"hybrid_upper_body_names contains bodies not in simulator: {missing}")
        self._hybrid_upper_body_indices = torch.tensor(
            [robot_body_names.index(bn) for bn in self.hybrid_upper_body_names],
            device=self.device,
            dtype=torch.long,
        )

        # Map desired seconds stride to motion frames.
        # NOTE: motion.fps is loaded from the motion npz.
        fps = float(self.motion.fps)
        stride = int(round(self.window_dt_s * fps))
        self._window_stride_steps = max(1, stride)

        self._window_offsets = (
            torch.arange(self.window_num_frames, device=self.device, dtype=torch.long) * self._window_stride_steps
        )

        human_stride = int(round(self.human_window_dt_s * fps))
        self._human_stride_steps = max(1, human_stride)
        self._human_offsets = (
            torch.arange(self.human_window_num_frames, device=self.device, dtype=torch.long) * self._human_stride_steps
        )

        # Move probs to device once we know env device.
        self.command_type_probs = self.command_type_probs.to(self.device)

    def reset(self, env_ids: torch.Tensor | None) -> None:
        super().reset(env_ids)

        if self.command_type_id is None:
            raise RuntimeError("SonicMotionCommand.setup() must be called before reset")

        env_ids = self._ensure_index_tensor(env_ids)
        if env_ids.numel() == 0:
            return

        probs = self.command_type_probs / (self.command_type_probs.sum() + 1e-8)
        sampled = torch.multinomial(probs, num_samples=env_ids.numel(), replacement=True)
        self.command_type_id[env_ids] = sampled

    def _get_window_indices(self) -> torch.Tensor:
        if self._window_offsets is None:
            raise RuntimeError("SonicMotionCommand.setup() must be called before using command")

        base = self.time_steps.view(-1, 1)  # (B, 1)
        idx = base + self._window_offsets.view(1, -1)  # (B, F)
        idx = torch.clamp(idx, 0, int(self.motion.time_step_total) - 1)
        return idx

    def _get_human_window_indices(self) -> torch.Tensor:
        if self._human_offsets is None:
            raise RuntimeError("SonicMotionCommand.setup() must be called before using human_command")

        base = self.time_steps.view(-1, 1)
        idx = base + self._human_offsets.view(1, -1)
        idx = torch.clamp(idx, 0, int(self.motion.time_step_total) - 1)
        return idx

    def _body_pos_local_heading(self, *, idx_flat: torch.Tensor, body_indices: torch.Tensor) -> torch.Tensor:
        """Return body positions in the local heading frame, relative to reference body."""

        body_pos_w = self.motion.body_pos_w[idx_flat][:, body_indices]  # (N, B, 3)
        ref_pos_w = self.motion.body_pos_w[idx_flat][:, self.ref_body_index].unsqueeze(1)  # (N, 1, 3)
        vec_w = body_pos_w - ref_pos_w

        ref_quat_xyzw = self.motion.body_quat_w[idx_flat][:, self.ref_body_index]  # (N, 4)
        heading_xyzw = yaw_quat(ref_quat_xyzw, w_last=True)
        heading_rep = heading_xyzw.unsqueeze(1).repeat(1, body_pos_w.shape[1], 1).reshape(-1, 4)
        vec_local = quat_rotate_inverse(heading_rep, vec_w.reshape(-1, 3), w_last=True)
        return vec_local.reshape(body_pos_w.shape[0], body_pos_w.shape[1], 3)

    @property
    def command(self) -> torch.Tensor:
        """Return flattened future-window command.

        Output shape:
            (num_envs, window_num_frames * (2 * num_dofs))
        """

        idx = self._get_window_indices()  # (B, F)
        b, f = idx.shape

        idx_flat = idx.reshape(-1)
        joint_pos = self.motion.joint_pos[idx_flat].reshape(b, f, -1)
        joint_vel = self.motion.joint_vel[idx_flat].reshape(b, f, -1)
        window = torch.cat([joint_pos, joint_vel], dim=-1)  # (B, F, 2J)
        return window.reshape(b, -1)

    @property
    def robot_command(self) -> torch.Tensor:
        return self.command

    @property
    def human_command(self) -> torch.Tensor:
        """Human command proxy: future window of 3D joint positions.

        We derive these from the motion's body positions (retargeted to the robot)
        and express them in the robot's local heading frame, relative to the
        reference body.

        Shape:
            (num_envs, Fh * (Nh * 3))
        """

        # Preferred: use preprocessed human motion stream if present in the motion file.
        # Expected shape in converted npz: (T, J, 3).
        motion_human = getattr(self.motion, "human_motion", None)
        if motion_human is not None:
            idx = self._get_human_window_indices()  # (B, Fh)
            b, f = idx.shape
            idx_flat = idx.reshape(-1)
            human = motion_human[idx_flat]  # (B*Fh, J, 3)
            human = human.reshape(b, f, -1)
            return human.reshape(b, -1)

        if self._human_body_indices is None:
            raise RuntimeError("SonicMotionCommand.setup() must be called before using human_command")

        idx = self._get_human_window_indices()  # (B, Fh)
        b, f = idx.shape
        idx_flat = idx.reshape(-1)

        pos_local = self._body_pos_local_heading(idx_flat=idx_flat, body_indices=self._human_body_indices)
        pos_local = pos_local.reshape(b, f, -1)
        return pos_local.reshape(b, -1)

    @property
    def hybrid_command(self) -> torch.Tensor:
        """Hybrid command: upper-body keypoints (current) + lower-body robot future window."""

        # Preferred: use preprocessed hybrid stream if present in the motion file.
        # Expected shapes in converted npz:
        # - hybrid_motion_upper_body: (T, K, 3)  (typically K=3)
        # - hybrid_motion_lower_body: (T, Dl)    (e.g., lower-body features)
        motion_upper = getattr(self.motion, "hybrid_motion_upper_body", None)
        motion_lower = getattr(self.motion, "hybrid_motion_lower_body", None)
        if motion_upper is not None and motion_lower is not None:
            # upper-body at current timestep
            idx0 = self.time_steps.reshape(-1)
            upper = motion_upper[idx0].reshape(self.num_envs, -1)

            # lower-body over the same future window stride as robot command
            idx = self._get_window_indices()  # (B, F)
            b, f = idx.shape
            idx_flat = idx.reshape(-1)
            lower = motion_lower[idx_flat].reshape(b, f, -1)
            lower = lower.reshape(b, -1)

            return torch.cat([upper, lower], dim=-1)

        if self._hybrid_upper_body_indices is None:
            raise RuntimeError("SonicMotionCommand.setup() must be called before using hybrid_command")

        # Upper-body keypoints at current timestep.
        idx_flat = self.time_steps.reshape(-1)
        upper_local = self._body_pos_local_heading(idx_flat=idx_flat, body_indices=self._hybrid_upper_body_indices)
        upper_local = upper_local.reshape(self.num_envs, -1)

        # Lower-body robot motion window (flattened).
        return torch.cat([upper_local, self.robot_command], dim=-1)
