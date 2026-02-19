"""SONIC tracking command presets for the G1 robot.

Baseline is copied from Whole Body Tracking (WBT) so we can introduce SONIC
tracker changes without touching existing WBT experiments.
"""

import os

from dataclasses import replace

from holosoma.config_types.command import CommandManagerCfg, CommandTermCfg, MotionConfig, NoiseToInitialPoseConfig

init_pose_config = NoiseToInitialPoseConfig(
    overall_noise_scale=1.0,
    dof_pos=0.1,
    root_pos=[0.05, 0.05, 0.01],
    root_rot=[0.1, 0.1, 0.2],
    # Table 2: Root velocity perturbations (external pushes)
    root_lin_vel=[0.5, 0.5, 0.2],
    root_ang_vel=[0.52, 0.52, 0.78],
    object_pos=[0.05, 0.05, 0.0],
)

motion_config = MotionConfig(
    motion_file=os.environ.get(
        "HOLOSOMA_SONIC_MOTION_FILE",
        "holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj.npz",
    ),
    body_names_to_track=[
        "pelvis",
        "left_hip_roll_link",
        "left_knee_link",
        "left_ankle_roll_link",
        "right_hip_roll_link",
        "right_knee_link",
        "right_ankle_roll_link",
        "torso_link",
        "left_shoulder_roll_link",
        "left_elbow_link",
        "left_wrist_yaw_link",
        "right_shoulder_roll_link",
        "right_elbow_link",
        "right_wrist_yaw_link",
    ],
    body_name_ref=["torso_link"],
    # Table 4: bin-based adaptive motion sampling
    use_adaptive_timesteps_sampler=True,
    adaptive_sampling_bin_size_s=1.0,
    adaptive_sampling_failure_rate_cap_beta=200.0,
    adaptive_sampling_blending_alpha=0.1,
    noise_to_initial_pose=init_pose_config,
)

motion_config_w_object = replace(
    motion_config,
    motion_file="holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz",
)

g1_29dof_sonic_command = CommandManagerCfg(
    params={},
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.sonic:SonicMotionCommand",
            params={
                "motion_config": motion_config,
                # SONIC v0: future-window command
                "window_num_frames": 10,
                "window_dt_s": 0.1,
                # SONIC Sec. 3.2: multi-modal command streams
                "human_window_num_frames": 10,
                "human_window_dt_s": 0.02,
                # sample robot/human/hybrid per episode
                "command_type_probs": [0.34, 0.33, 0.33],
                # hybrid upper-body keypoints: head + hands (G1 model has no explicit head body,
                # so torso_link is used as a head proxy)
                "hybrid_upper_body_names": ["torso_link", "left_rubber_hand_link", "right_rubber_hand_link"],
            },
        ),
    },
    reset_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.sonic:SonicMotionCommand",
        )
    },
    step_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.sonic:SonicMotionCommand",
        )
    },
)

g1_29dof_sonic_command_w_object = replace(
    g1_29dof_sonic_command,
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.sonic:SonicMotionCommand",
            params={
                "motion_config": motion_config_w_object,
                "window_num_frames": 10,
                "window_dt_s": 0.1,
            },
        )
    },
)

__all__ = [
    "g1_29dof_sonic_command",
    "g1_29dof_sonic_command_w_object",
]
