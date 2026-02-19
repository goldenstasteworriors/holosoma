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
                ["torso_link", "left_rubber_hand_link", "right_rubber_hand_link"],
            )
        )

        # Optional lower-body DOF filter for hybrid command.
        self.hybrid_lower_dof_names: list[str] | None = params.get("hybrid_lower_dof_names")

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
        self._hybrid_lower_dof_indices: torch.Tensor | None = None

        # 0=robot, 1=human, 2=hybrid
        self.command_type_id: torch.Tensor | None = None

    def setup(self) -> None:
        super().setup()

        # Allocate command type buffer.
        self.command_type_id = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # Resolve body indices used for human/hybrid commands.
        robot_body_names = getattr(self._env.simulator, "body_names", None)
        if robot_body_names is None:
            robot_body_names = self._env.simulator._body_list  # type: ignore[attr-defined]

        if self.human_body_names is None:
            self.human_body_names = list(self.motion_cfg.body_names_to_track)

        def _resolve_body_name(name: str) -> str | None:
            if name in robot_body_names:
                return name
            if name.endswith("_link"):
                no_link = name[: -len("_link")]
                if no_link in robot_body_names:
                    return no_link
            with_link = f"{name}_link"
            if with_link in robot_body_names:
                return with_link

            # Common G1 naming differences / missing bodies in exposed lists.
            # If rubber-hand bodies are not present, fall back to wrist links.
            if "left_rubber_hand" in name:
                for candidate in ("left_rubber_hand", "left_wrist_yaw_link"):
                    if candidate in robot_body_names:
                        return candidate
            if "right_rubber_hand" in name:
                for candidate in ("right_rubber_hand", "right_wrist_yaw_link"):
                    if candidate in robot_body_names:
                        return candidate
            return None

        resolved_human = []
        missing_human = []
        for bn in self.human_body_names:
            resolved = _resolve_body_name(bn)
            if resolved is None:
                missing_human.append(bn)
            else:
                resolved_human.append(resolved)

        if missing_human:
            raise ValueError(f"human_body_names contains bodies not in simulator: {missing_human}")
        self.human_body_names = resolved_human
        self._human_body_indices = torch.tensor(
            [robot_body_names.index(bn) for bn in self.human_body_names],
            device=self.device,
            dtype=torch.long,
        )

        resolved_hybrid_upper = []
        missing_hybrid_upper = []
        for bn in self.hybrid_upper_body_names:
            resolved = _resolve_body_name(bn)
            if resolved is None:
                missing_hybrid_upper.append(bn)
            else:
                resolved_hybrid_upper.append(resolved)

        if missing_hybrid_upper:
            raise ValueError(
                f"hybrid_upper_body_names contains bodies not in simulator: {missing_hybrid_upper}"
            )
        self.hybrid_upper_body_names = resolved_hybrid_upper
        self._hybrid_upper_body_indices = torch.tensor(
            [robot_body_names.index(bn) for bn in self.hybrid_upper_body_names],
            device=self.device,
            dtype=torch.long,
        )

        # Resolve DOF indices used for hybrid lower-body command.
        if self.hybrid_lower_dof_names is None:
            robot_cfg = getattr(self._env, "robot_config", None)
            if robot_cfg is not None and getattr(robot_cfg, "lower_dof_names", None) is not None:
                self.hybrid_lower_dof_names = list(robot_cfg.lower_dof_names)

        if self.hybrid_lower_dof_names is not None:
            robot_dof_names = getattr(self._env.simulator, "dof_names", None)
            if robot_dof_names is None:
                raise ValueError("Simulator does not expose dof_names; cannot use hybrid_lower_dof_names")
            missing_dofs = [dn for dn in self.hybrid_lower_dof_names if dn not in robot_dof_names]
            if missing_dofs:
                raise ValueError(f"hybrid_lower_dof_names contains DOFs not in simulator: {missing_dofs}")
            self._hybrid_lower_dof_indices = torch.tensor(
                [robot_dof_names.index(dn) for dn in self.hybrid_lower_dof_names],
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
        max_idx = (self._get_env_motion_time_step_total() - 1).view(-1, 1)
        idx = torch.minimum(idx, max_idx)
        idx = torch.clamp(idx, 0)
        return idx

    def _get_human_window_indices(self) -> torch.Tensor:
        if self._human_offsets is None:
            raise RuntimeError("SonicMotionCommand.setup() must be called before using human_command")

        base = self.time_steps.view(-1, 1)
        idx = base + self._human_offsets.view(1, -1)
        max_idx = (self._get_env_motion_time_step_total() - 1).view(-1, 1)
        idx = torch.minimum(idx, max_idx)
        idx = torch.clamp(idx, 0)
        return idx

    def _pool_attr_available(self, attr: str) -> bool:
        """Return True if `attr` is available (non-None) for the current clip assignment.

        For pool mode, require all clips in the pool to have the attribute so that
        gather operations are well-defined.
        """

        if not self._using_motion_pool():
            return getattr(self.motion, attr, None) is not None

        assert self._motion_pool is not None
        for clip in self._motion_pool:
            if getattr(clip, attr, None) is None:
                return False
        return True

    def _body_pos_local_heading(self, *, idx_flat: torch.Tensor, body_indices: torch.Tensor) -> torch.Tensor:
        """Return body positions in the local heading frame, relative to reference body."""

        idx_full = idx_flat.view(self.num_envs, -1)
        body_pos_w_full = self._gather_from_motion_pool(attr="body_pos_w", idx=idx_full)  # (B, F, Nb, 3)
        body_pos_w_full = body_pos_w_full.reshape(-1, body_pos_w_full.shape[2], 3)  # (B*F, Nb, 3)
        body_pos_w = body_pos_w_full[:, body_indices]  # (B*F, K, 3)
        ref_pos_w = body_pos_w_full[:, self.ref_body_index].unsqueeze(1)  # (B*F, 1, 3)
        vec_w = body_pos_w - ref_pos_w

        body_quat_w_full = self._gather_from_motion_pool(attr="body_quat_w", idx=idx_full)  # (B, F, Nb, 4)
        ref_quat_xyzw = body_quat_w_full[:, :, self.ref_body_index].reshape(-1, 4)  # (B*F, 4)
        heading_xyzw = yaw_quat(ref_quat_xyzw, w_last=True)
        heading_rep = heading_xyzw.unsqueeze(1).repeat(1, body_pos_w.shape[1], 1).reshape(-1, 4)
        vec_local = quat_rotate_inverse(heading_rep, vec_w.reshape(-1, 3), w_last=True)
        return vec_local.reshape(body_pos_w.shape[0], body_pos_w.shape[1], 3)

    def _body_vec_local_heading(self, *, vec_w: torch.Tensor, idx_flat: torch.Tensor) -> torch.Tensor:
        """Rotate world vectors into the local heading frame."""

        idx_full = idx_flat.view(self.num_envs, -1)
        body_quat_w_full = self._gather_from_motion_pool(attr="body_quat_w", idx=idx_full)  # (B, F, Nb, 4)
        ref_quat_xyzw = body_quat_w_full[:, :, self.ref_body_index].reshape(-1, 4)  # (B*F, 4)
        heading_xyzw = yaw_quat(ref_quat_xyzw, w_last=True)
        heading_rep = heading_xyzw.unsqueeze(1).repeat(1, vec_w.shape[1], 1).reshape(-1, 4)
        vec_local = quat_rotate_inverse(heading_rep, vec_w.reshape(-1, 3), w_last=True)
        return vec_local.reshape(vec_w.shape[0], vec_w.shape[1], 3)

    @property
    def command(self) -> torch.Tensor:
        """Return flattened future-window command.

        Output shape:
            (num_envs, window_num_frames * (2 * num_dofs))
        """

        idx = self._get_window_indices()  # (B, F)
        b, f = idx.shape

        joint_pos = self._gather_from_motion_pool(attr="joint_pos", idx=idx)
        joint_vel = self._gather_from_motion_pool(attr="joint_vel", idx=idx)
        window = torch.cat([joint_pos, joint_vel], dim=-1)  # (B, F, 2J)
        return window.reshape(b, -1)

    @property
    def robot_command(self) -> torch.Tensor:
        return self.command

    @property
    def human_command(self) -> torch.Tensor:
        """Human command: future window of (pos_rel_root, vel_rel_root).

        Definition per project spec:
        - human command = each joint/link's 3D position relative to root + velocity (window)

        Shape:
            (num_envs, Fh * (Nh * 6))
        """

        # Preferred: use conversion-derived per-DOF joint anchor signals (root-relative).
        if self._pool_attr_available("sonic_joint_pos_rel_root") and self._pool_attr_available(
            "sonic_joint_lin_vel_rel_root"
        ):
            idx = self._get_human_window_indices()  # (B, Fh)
            b, f = idx.shape
            joint_pos = self._gather_from_motion_pool(attr="sonic_joint_pos_rel_root", idx=idx)
            joint_vel = self._gather_from_motion_pool(attr="sonic_joint_lin_vel_rel_root", idx=idx)
            feat = torch.cat([joint_pos, joint_vel], dim=-1).reshape(b, f, -1)
            return feat.reshape(b, -1)

        # Next best: use conversion-derived sonic_* body signals (root-relative).
        if self._pool_attr_available("sonic_body_pos_rel_root") and self._pool_attr_available(
            "sonic_body_lin_vel_rel_root"
        ):
            if self._human_body_indices is None:
                raise RuntimeError("SonicMotionCommand.setup() must be called before using human_command")
            idx = self._get_human_window_indices()  # (B, Fh)
            b, f = idx.shape
            pos = self._gather_from_motion_pool(attr="sonic_body_pos_rel_root", idx=idx)[:, :, self._human_body_indices]
            vel = self._gather_from_motion_pool(attr="sonic_body_lin_vel_rel_root", idx=idx)[:, :, self._human_body_indices]
            feat = torch.cat([pos, vel], dim=-1).reshape(b, f, -1)
            return feat.reshape(b, -1)

        # Backward-compat: preprocessed human stream if present.
        if self._pool_attr_available("human_motion"):
            idx = self._get_human_window_indices()  # (B, Fh)
            b, f = idx.shape
            pos_w = self._gather_from_motion_pool(attr="human_motion", idx=idx)
            pos = pos_w.reshape(-1, *pos_w.shape[2:])
            if pos.shape[-1] == 3:
                vel = torch.zeros_like(pos)
                feat = torch.cat([pos, vel], dim=-1)
            elif pos.shape[-1] == 6:
                feat = pos
            else:
                raise ValueError(f"Unexpected human_motion last dim: {pos.shape[-1]}")
            feat = feat.reshape(b, f, -1)
            return feat.reshape(b, -1)

        if self._human_body_indices is None:
            raise RuntimeError("SonicMotionCommand.setup() must be called before using human_command")

        idx = self._get_human_window_indices()  # (B, Fh)
        b, f = idx.shape
        idx_flat = idx.reshape(-1)

        # Fallback proxy: local-heading, relative-to-ref body pos + vel.
        idx_full = idx  # (B, Fh)
        body_pos_w_full = self._gather_from_motion_pool(attr="body_pos_w", idx=idx_full)  # (B, Fh, Nb, 3)
        body_pos_w = body_pos_w_full.reshape(-1, body_pos_w_full.shape[2], 3)[:, self._human_body_indices]
        ref_pos_w = body_pos_w_full[:, :, self.ref_body_index].reshape(-1, 3).unsqueeze(1)
        pos_rel = body_pos_w - ref_pos_w
        pos_local = self._body_vec_local_heading(vec_w=pos_rel, idx_flat=idx_flat)

        body_vel_w_full = self._gather_from_motion_pool(attr="body_lin_vel_w", idx=idx_full)
        body_vel_w = body_vel_w_full.reshape(-1, body_vel_w_full.shape[2], 3)[:, self._human_body_indices]
        ref_vel_w = body_vel_w_full[:, :, self.ref_body_index].reshape(-1, 3).unsqueeze(1)
        vel_rel = body_vel_w - ref_vel_w
        vel_local = self._body_vec_local_heading(vec_w=vel_rel, idx_flat=idx_flat)

        feat = torch.cat([pos_local, vel_local], dim=-1).reshape(b, f, -1)
        return feat.reshape(b, -1)

    @property
    def hybrid_command(self) -> torch.Tensor:
        """Hybrid command: upper-body keypoints + lower-body joint angles/vel (window).

        Definition per project spec:
        - upper-body = 3 keypoints (head + two hands): pos_rel_root + vel_rel_root (window)
        - lower-body = joint angles + joint velocities (window)
        """

        # Preferred: conversion-derived sonic_* hybrid upper-body keypoints (root-relative).
        if self._pool_attr_available("sonic_hybrid_upper_pos_rel_root") and self._pool_attr_available(
            "sonic_hybrid_upper_vel_rel_root"
        ):
            idx_u = self._get_human_window_indices()  # (B, Fh)
            b, f = idx_u.shape
            upper_pos = self._gather_from_motion_pool(attr="sonic_hybrid_upper_pos_rel_root", idx=idx_u)
            upper_vel = self._gather_from_motion_pool(attr="sonic_hybrid_upper_vel_rel_root", idx=idx_u)
            upper = torch.cat([upper_pos, upper_vel], dim=-1).reshape(b, f, -1)
            upper = upper.reshape(b, -1)

            lower = self._hybrid_lower_robot_window()
            return torch.cat([upper, lower], dim=-1)

        # Next best: use sonic_* body signals and pick configured upper-body names.
        if self._pool_attr_available("sonic_body_pos_rel_root") and self._pool_attr_available(
            "sonic_body_lin_vel_rel_root"
        ):
            if self._hybrid_upper_body_indices is None:
                raise RuntimeError("SonicMotionCommand.setup() must be called before using hybrid_command")
            idx_u = self._get_human_window_indices()
            b, f = idx_u.shape
            pos = self._gather_from_motion_pool(attr="sonic_body_pos_rel_root", idx=idx_u)[:, :, self._hybrid_upper_body_indices]
            vel = self._gather_from_motion_pool(attr="sonic_body_lin_vel_rel_root", idx=idx_u)[:, :, self._hybrid_upper_body_indices]
            upper = torch.cat([pos, vel], dim=-1).reshape(b, f, -1).reshape(b, -1)
            lower = self._hybrid_lower_robot_window()
            return torch.cat([upper, lower], dim=-1)

        # Backward-compat: preprocessed hybrid stream if present in the motion file.
        if self._pool_attr_available("hybrid_motion_upper_body") and self._pool_attr_available(
            "hybrid_motion_lower_body"
        ):
            idx_u = self._get_human_window_indices()  # (B, Fh)
            b, f = idx_u.shape
            upper_w = self._gather_from_motion_pool(attr="hybrid_motion_upper_body", idx=idx_u)
            upper = upper_w.reshape(-1, *upper_w.shape[2:])
            if upper.shape[-1] == 3:
                upper = torch.cat([upper, torch.zeros_like(upper)], dim=-1)
            upper = upper.reshape(b, f, -1).reshape(b, -1)

            idx_l = self._get_window_indices()
            lower = self._gather_from_motion_pool(attr="hybrid_motion_lower_body", idx=idx_l).reshape(b, -1)
            return torch.cat([upper, lower], dim=-1)

        if self._hybrid_upper_body_indices is None:
            raise RuntimeError("SonicMotionCommand.setup() must be called before using hybrid_command")

        # Fallback proxy: local-heading upper-body (pos+vel window) + lower-body robot window.
        if self._hybrid_upper_body_indices is None:
            raise RuntimeError("SonicMotionCommand.setup() must be called before using hybrid_command")

        idx_u = self._get_human_window_indices()
        b, f = idx_u.shape
        idx_u_flat = idx_u.reshape(-1)

        idx_full = idx_u
        body_pos_w_full = self._gather_from_motion_pool(attr="body_pos_w", idx=idx_full)
        body_pos_w = body_pos_w_full.reshape(-1, body_pos_w_full.shape[2], 3)[:, self._hybrid_upper_body_indices]
        ref_pos_w = body_pos_w_full[:, :, self.ref_body_index].reshape(-1, 3).unsqueeze(1)
        pos_rel = body_pos_w - ref_pos_w
        pos_local = self._body_vec_local_heading(vec_w=pos_rel, idx_flat=idx_u_flat)

        body_vel_w_full = self._gather_from_motion_pool(attr="body_lin_vel_w", idx=idx_full)
        body_vel_w = body_vel_w_full.reshape(-1, body_vel_w_full.shape[2], 3)[:, self._hybrid_upper_body_indices]
        ref_vel_w = body_vel_w_full[:, :, self.ref_body_index].reshape(-1, 3).unsqueeze(1)
        vel_rel = body_vel_w - ref_vel_w
        vel_local = self._body_vec_local_heading(vec_w=vel_rel, idx_flat=idx_u_flat)

        upper = torch.cat([pos_local, vel_local], dim=-1).reshape(b, f, -1).reshape(b, -1)
        lower = self._hybrid_lower_robot_window()
        return torch.cat([upper, lower], dim=-1)

    def _hybrid_lower_robot_window(self) -> torch.Tensor:
        """Return lower-body joint (pos, vel) future window, flattened."""

        idx = self._get_window_indices()  # (B, F)
        b, f = idx.shape

        joint_pos = self._gather_from_motion_pool(attr="joint_pos", idx=idx)
        joint_vel = self._gather_from_motion_pool(attr="joint_vel", idx=idx)

        if self._hybrid_lower_dof_indices is not None:
            joint_pos = joint_pos[:, :, self._hybrid_lower_dof_indices]
            joint_vel = joint_vel[:, :, self._hybrid_lower_dof_indices]

        window = torch.cat([joint_pos, joint_vel], dim=-1)
        return window.reshape(b, -1)
