from dataclasses import replace

from holosoma.config_types.experiment import ExperimentConfig, NightlyConfig, TrainingConfig
from holosoma.config_values import action, algo, robot, simulator, terrain
from holosoma.config_values.sonic.g1.command import g1_29dof_sonic_command, g1_29dof_sonic_command_w_object
from holosoma.config_values.sonic.g1.curriculum import g1_29dof_sonic_curriculum
from holosoma.config_values.sonic.g1.observation import g1_29dof_sonic_observation, g1_29dof_sonic_observation_w_object
from holosoma.config_values.sonic.g1.randomization import (
    g1_29dof_sonic_randomization,
    g1_29dof_sonic_randomization_w_object,
)
from holosoma.config_values.sonic.g1.reward import g1_29dof_sonic_reward, g1_29dof_sonic_reward_w_object
from holosoma.config_values.sonic.g1.termination import g1_29dof_sonic_termination

g1_29dof_sonic = ExperimentConfig(
    training=TrainingConfig(
        project="SONIC",
        name="g1_29dof_sonic_manager",
        num_envs=8192,
    ),
    env_class="holosoma.envs.sonic.sonic_manager.SonicTrackingManager",
    algo=replace(
        algo.ppo,
        config=replace(
            algo.ppo.config,
            num_learning_iterations=40000,
            save_interval=4000,
            entropy_coef=0.005,
            init_noise_std=1.0,
            init_at_random_ep_len=False,
            use_symmetry=False,
            actor_optimizer=replace(algo.ppo.config.actor_optimizer, weight_decay=0.000),
            critic_optimizer=replace(algo.ppo.config.critic_optimizer, weight_decay=0.000),
            module_dict=replace(
                algo.ppo.config.module_dict,
                actor=replace(
                    algo.ppo.config.module_dict.actor,
                    type="SonicUniversal",
                    # Reuse existing LayerConfig fields:
                    # - hidden_dims: used by encoder+decoder MLPs
                    # - encoder_output_dim: used as latent/token dimension
                    layer_config=replace(
                        algo.ppo.config.module_dict.actor.layer_config,
                        encoder_output_dim=128,
                        sonic_fsq_enabled=True,
                        sonic_fsq_levels=[8, 5, 5, 5],
                        sonic_recon_coef=1.0,
                        sonic_token_coef=1.0,
                        sonic_cycle_coef=1.0,
                    ),
                ),
            ),
        ),
    ),
    simulator=replace(
        simulator.isaacsim,
        config=replace(
            simulator.isaacsim.config,
            sim=replace(
                simulator.isaacsim.config.sim,
                max_episode_length_s=10.0,
            ),
        ),
    ),
    robot=replace(
        robot.g1_29dof,
        control=replace(robot.g1_29dof.control, action_scale=1.0),
        asset=replace(robot.g1_29dof.asset, enable_self_collisions=True),
        init_state=replace(robot.g1_29dof.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    terrain=terrain.terrain_locomotion_plane,
    observation=g1_29dof_sonic_observation,
    action=action.g1_29dof_joint_pos,
    termination=g1_29dof_sonic_termination,
    randomization=g1_29dof_sonic_randomization,
    command=g1_29dof_sonic_command,
    curriculum=g1_29dof_sonic_curriculum,
    reward=g1_29dof_sonic_reward,
    nightly=NightlyConfig(
        iterations=8000,
        metrics={
            "Episode/rew_motion_global_ref_position_error_exp": [0.16, "inf"],
            "Episode/rew_motion_global_ref_orientation_error_exp": [0.20, "inf"],
            "Episode/rew_motion_relative_body_position_error_exp": [0.45, "inf"],
            "Episode/rew_motion_relative_body_orientation_error_exp": [0.30, "inf"],
            "Episode/rew_motion_global_body_lin_vel": [0.30, "inf"],
            "Episode/rew_motion_global_body_ang_vel": [0.02, "inf"],
        },
    ),
)

g1_29dof_sonic_w_object = replace(
    g1_29dof_sonic,
    command=g1_29dof_sonic_command_w_object,
    robot=replace(
        robot.g1_29dof_w_object,
        asset=replace(
            robot.g1_29dof_w_object.asset,
            enable_self_collisions=True,
        ),
        object=replace(
            robot.g1_29dof_w_object.object,
            object_urdf_path="holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf",
        ),
        init_state=replace(robot.g1_29dof_w_object.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    randomization=g1_29dof_sonic_randomization_w_object,
    observation=g1_29dof_sonic_observation_w_object,
    reward=g1_29dof_sonic_reward_w_object,
    simulator=replace(
        simulator.isaacsim,
        config=replace(simulator.isaacsim.config, scene=replace(simulator.isaacsim.config.scene, env_spacing=0.0)),
    ),
)

__all__ = [
    "g1_29dof_sonic",
    "g1_29dof_sonic_w_object",
]

"""
Example 1: Robot only:
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-sonic

Example 2: Robot+Object:
python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-sonic-w-object
"""
