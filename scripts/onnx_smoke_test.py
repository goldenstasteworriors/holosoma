"""Minimal ONNXRuntime smoke test.

Usage (must run via conda env as per project constraint):
  conda run -n hssim python scripts/onnx_smoke_test.py --onnx path/to/model.onnx

This script:
- loads the ONNX model with onnxruntime (CPU)
- constructs a (batch=1) float32 input matching the model's actor_obs dim
- runs one forward pass
- prints output shape and value range

It is intentionally minimal and does not depend on IsaacSim.
"""

from __future__ import annotations

import argparse

import numpy as np


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True, help="Path to exported .onnx model")
    p.add_argument(
        "--obs-dim",
        type=int,
        default=None,
        help="Override actor_obs feature dim if ONNX shape is dynamic/unknown",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    import onnxruntime as ort

    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()

    print("inputs:")
    for i in inputs:
        print(" -", i.name, i.shape, i.type)
    print("outputs:")
    for o in outputs:
        print(" -", o.name, o.shape, o.type)

    input_names = [i.name for i in inputs]

    feed: dict[str, np.ndarray] = {}

    def _infer_feat_dim(shape) -> int | None:
        if shape is None or len(shape) < 2:
            return None
        return int(shape[1]) if isinstance(shape[1], int) else None

    # Case 1: exported policy-only model.
    if input_names == ["actor_obs"]:
        feat_dim = _infer_feat_dim(inputs[0].shape)
        if args.obs_dim is not None:
            feat_dim = int(args.obs_dim)
        if feat_dim is None or feat_dim <= 0:
            raise SystemExit("Could not infer actor_obs dim; pass --obs-dim")
        feed["actor_obs"] = np.zeros((1, feat_dim), dtype=np.float32)

    # Case 2: exported motion+policy model.
    elif set(input_names) == {"obs", "time_step"}:
        obs_in = next(i for i in inputs if i.name == "obs")
        ts_in = next(i for i in inputs if i.name == "time_step")
        feat_dim = _infer_feat_dim(obs_in.shape)
        if args.obs_dim is not None:
            feat_dim = int(args.obs_dim)
        if feat_dim is None or feat_dim <= 0:
            raise SystemExit("Could not infer obs dim; pass --obs-dim")
        feed["obs"] = np.zeros((1, feat_dim), dtype=np.float32)
        # Exporter uses float time_step (see inference_helpers).
        feed["time_step"] = np.zeros((1, 1), dtype=np.float32)

    else:
        raise SystemExit(f"Unsupported input signature: {input_names}")

    y = session.run(None, feed)

    print("\nrun_ok")
    for name, arr in zip([o.name for o in outputs], y, strict=False):
        print(f"{name}: shape={getattr(arr, 'shape', None)} dtype={getattr(arr, 'dtype', None)}")
        if isinstance(arr, np.ndarray) and arr.size > 0 and np.isfinite(arr).any():
            print(f"{name}: min={float(np.min(arr))} max={float(np.max(arr))}")


if __name__ == "__main__":
    main()
