"""Minimal unit test for ONNX export functionality."""

import tempfile
from pathlib import Path

import onnx
import torch
from torch import nn

from holosoma.agents.modules.module_utils import setup_ppo_actor_module
from holosoma.config_types.algo import LayerConfig, ModuleConfig
from holosoma.utils.inference_helpers import export_motion_and_policy_as_onnx, export_policy_as_onnx


class ActorWrapper(nn.Module):
    """Wrapper matching PPO's actor_onnx_wrapper pattern."""

    def __init__(self, actor: nn.Module):
        super().__init__()
        self.actor = actor

    def forward(self, actor_obs: torch.Tensor) -> torch.Tensor:
        return self.actor.act_inference({"actor_obs": actor_obs})


def test_export_policy_as_onnx():
    """Test ONNX export, load, and dimension verification."""
    OBS_DIM, ACT_DIM = 10, 5

    # Minimal config for PPOActor
    module_config = ModuleConfig(
        type="MLP",
        input_dim=["actor_obs"],
        output_dim=[ACT_DIM],
        layer_config=LayerConfig(
            hidden_dims=[64],
            activation="ReLU",
            dropout_prob=0.0,
        ),
        min_noise_std=None,
        min_mean_noise_std=None,
    )

    # Create PPOActor
    actor = setup_ppo_actor_module(
        obs_dim_dict={"actor_obs": OBS_DIM},
        module_config=module_config,
        num_actions=ACT_DIM,
        init_noise_std=0.1,
        device="cpu",
        history_length={"actor_obs": 1},
    )
    wrapper = ActorWrapper(actor)
    wrapper.eval()

    # Export to a temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = str(Path(tmpdir) / "test_policy.onnx")
        example_obs = torch.zeros(1, OBS_DIM)

        export_policy_as_onnx(
            wrapper=wrapper,
            onnx_file_path=onnx_path,
            example_obs_dict={"actor_obs": example_obs},
        )

        # Load and verify
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)

        # Check input/output dims
        assert len(model.graph.input) == 1
        assert len(model.graph.output) == 1

        input_shape = model.graph.input[0].type.tensor_type.shape
        output_shape = model.graph.output[0].type.tensor_type.shape

        assert input_shape.dim[1].dim_value == OBS_DIM
        assert output_shape.dim[1].dim_value == ACT_DIM


class _FakeMotion:
    def __init__(self):
        # Minimal tensors required by _OnnxMotionPolicyExporter
        self.joint_pos = torch.zeros(2, 3)
        self.joint_vel = torch.zeros(2, 3)
        self.body_pos_w = torch.zeros(2, 1, 3)
        self.body_quat_w = torch.zeros(2, 1, 4)
        self.fps = torch.tensor([50])
        self.time_step_total = 2


class _FakeMotionCommand:
    def __init__(self, *, dr: int, dh: int, dm: int):
        self.robot_command = torch.zeros(1, dr)
        self.human_command = torch.zeros(1, dh)
        self.hybrid_command = torch.zeros(1, dm)
        self.motion = _FakeMotion()
        self.ref_body_index = 0


def test_export_motion_and_policy_as_onnx_sonic_uses_model_dims():
    """Regression test: avoid SONIC export mismatch when motion-command dims differ.

    The trained SONIC adapter is lazily built with dr/dh/dm inferred from the
    bundle header seen during training (and stored in weights). During export,
    the env motion-command dims can differ; exporter should prefer the model's
    built dims to avoid matmul shape mismatch.
    """

    # Build a SONIC universal actor.
    # We keep actor_obs dimension minimal: header(6) + dr+dh+dm.
    dr = dh = dm = 100
    obs_dim = 6 + dr + dh + dm
    act_dim = 5

    module_config = ModuleConfig(
        type="SonicUniversal",
        input_dim=["actor_obs"],
        output_dim=[act_dim],
        layer_config=LayerConfig(
            hidden_dims=[128],
            activation="ELU",
            dropout_prob=0.0,
            encoder_output_dim=128,
            sonic_fsq_enabled=True,
            sonic_fsq_levels=[8, 5, 5, 5],
            sonic_recon_coef=1.0,
            sonic_token_coef=1.0,
            sonic_cycle_coef=1.0,
        ),
        min_noise_std=None,
        min_mean_noise_std=None,
    )

    actor = setup_ppo_actor_module(
        obs_dim_dict={"actor_obs": obs_dim},
        module_config=module_config,
        num_actions=act_dim,
        init_noise_std=0.1,
        device="cpu",
        history_length={"actor_obs": 1},
    )

    # Build the SONIC policy modules once with a synthetic bundle.
    obs = torch.zeros(1, obs_dim)
    obs[0, 0] = float(dr)
    obs[0, 1] = float(dh)
    obs[0, 2] = float(dm)
    obs[0, 3] = 1.0  # onehot -> robot
    _ = actor.act_inference({"actor_obs": obs})

    wrapper = ActorWrapper(actor)
    wrapper.eval()

    # Create a fake motion_command that reports different dims.
    # This previously caused matmul mismatch during ONNX export.
    motion_command = _FakeMotionCommand(dr=580, dh=580, dm=580)

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = str(Path(tmpdir) / "test_sonic_motion_policy.onnx")
        export_motion_and_policy_as_onnx(
            actor=wrapper,
            motion_command=motion_command,
            onnx_file_path=onnx_path,
            device="cpu",
        )

        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)


def test_sonic_adapter_load_state_dict_builds_lazy_modules():
    """Regression: loading checkpoint should not fail with unexpected policy.* keys."""

    from holosoma.sonic.policy.ppo_adapter import UniversalSonicPolicyActorAdapter

    dr = dh = dm = 100
    obs_dim = 6 + dr + dh + dm

    class _Wrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.adapter = UniversalSonicPolicyActorAdapter(
                action_dim=5,
                decoder_hidden_dims=[128],
                latent_dim=128,
            )

    wrapper_src = _Wrapper()
    obs = torch.zeros(1, obs_dim)
    obs[0, 0] = float(dr)
    obs[0, 1] = float(dh)
    obs[0, 2] = float(dm)
    obs[0, 3] = 1.0
    _ = wrapper_src.adapter(obs)
    state_dict = wrapper_src.state_dict()

    wrapper_dst = _Wrapper()

    # Should not raise due to unexpected `adapter.policy.*` keys; this exercises
    # recursive loading (adapter._load_from_state_dict).
    wrapper_dst.load_state_dict(state_dict, strict=True)


if __name__ == "__main__":
    test_export_policy_as_onnx()
