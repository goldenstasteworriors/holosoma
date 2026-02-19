"""ONNXRuntime smoke tests for exported policies.

Goal: ensure the exported ONNX is actually runnable via onnxruntime with a
(batch=1) actor_obs input, and that I/O names/shapes are consistent.

This is intentionally minimal and CPU-only.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from holosoma.agents.modules.module_utils import setup_ppo_actor_module
from holosoma.config_types.algo import LayerConfig, ModuleConfig
from holosoma.utils.inference_helpers import export_policy_as_onnx


onnxruntime = pytest.importorskip("onnxruntime")


class ActorWrapper(nn.Module):
    """Wrapper matching PPO's actor_onnx_wrapper pattern."""

    def __init__(self, actor: nn.Module):
        super().__init__()
        self.actor = actor

    def forward(self, actor_obs: torch.Tensor) -> torch.Tensor:
        return self.actor.act_inference({"actor_obs": actor_obs})


def _run_ort_once(*, onnx_path: str, obs: np.ndarray) -> np.ndarray:
    session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    assert len(session.get_inputs()) == 1
    assert len(session.get_outputs()) == 1

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Exporter contract
    assert input_name == "actor_obs"
    assert output_name == "action"

    outputs = session.run(None, {input_name: obs})
    assert isinstance(outputs, list) and len(outputs) == 1
    return outputs[0]


def test_exported_onnx_runs_in_onnxruntime_mlp():
    obs_dim, act_dim = 10, 5

    module_config = ModuleConfig(
        type="MLP",
        input_dim=["actor_obs"],
        output_dim=[act_dim],
        layer_config=LayerConfig(
            hidden_dims=[64],
            activation="ReLU",
            dropout_prob=0.0,
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

    wrapper = ActorWrapper(actor)
    wrapper.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = str(Path(tmpdir) / "policy_mlp.onnx")
        example_obs = torch.zeros(1, obs_dim)
        export_policy_as_onnx(wrapper=wrapper, onnx_file_path=onnx_path, example_obs_dict={"actor_obs": example_obs})

        obs = np.zeros((1, obs_dim), dtype=np.float32)
        out = _run_ort_once(onnx_path=onnx_path, obs=obs)

        assert out.shape == (1, act_dim)
        assert out.dtype == np.float32
        assert np.isfinite(out).all()


def test_exported_onnx_runs_in_onnxruntime_sonic_universal():
    # Small synthetic SONIC bundle dims; in real runs these are data-driven
    # (e.g., dr=F*(2*num_dofs) for robot command).
    dr, dh, dm = 20, 12, 16
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

    # Build lazy SONIC adapter once so export traces a fully-initialized graph.
    obs = torch.zeros(1, obs_dim)
    obs[0, 0] = float(dr)
    obs[0, 1] = float(dh)
    obs[0, 2] = float(dm)
    obs[0, 3] = 1.0  # onehot -> robot
    _ = actor.act_inference({"actor_obs": obs})

    wrapper = ActorWrapper(actor)
    wrapper.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = str(Path(tmpdir) / "policy_sonic.onnx")
        export_policy_as_onnx(wrapper=wrapper, onnx_file_path=onnx_path, example_obs_dict={"actor_obs": obs})

        ort_in = obs.detach().cpu().numpy().astype(np.float32)
        out = _run_ort_once(onnx_path=onnx_path, obs=ort_in)

        assert out.shape == (1, act_dim)
        assert out.dtype == np.float32
        assert np.isfinite(out).all()
