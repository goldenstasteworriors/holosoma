from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import tyro

src_root = Path(__file__).resolve().parents[2]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from holosoma_retargeting.config_types.data_conversion import DataConversionConfig  # noqa: E402
from holosoma_retargeting.data_conversion.convert_data_format_mj import main  # noqa: E402


@dataclass(frozen=True)
class BatchConvertConfig:
    input_dir: str
    output_dir: str

    pattern: str = "*_original.npz"
    output_fps: int = 50

    robot: str = "g1"
    data_format: str = "lafan"
    object_name: str = "ground"

    has_dynamic_object: bool = False
    use_omniretarget_data: bool = False

    no_viewer: bool = True
    realtime: bool = False

    once: bool = True

    overwrite: bool = False


def _build_output_path(output_dir: Path, input_path: Path, *, suffix: str) -> Path:
    # Keep base name but drop trailing "_original" if present.
    name = input_path.stem
    if name.endswith("_original"):
        name = name[: -len("_original")]
    return output_dir / f"{name}{suffix}.npz"


def run(cfg: BatchConvertConfig) -> None:
    input_dir = Path(cfg.input_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = sorted(input_dir.glob(cfg.pattern))
    if not inputs:
        raise SystemExit(f"No input files found: {input_dir}/{cfg.pattern}")

    suffix = f"_mj_fps{cfg.output_fps}"

    for i, in_path in enumerate(inputs, start=1):
        out_path = _build_output_path(output_dir, in_path, suffix=suffix)
        if out_path.exists() and not cfg.overwrite:
            print(f"[{i}/{len(inputs)}] skip (exists): {out_path}")
            continue

        args = DataConversionConfig(
            input_file=str(in_path),
            robot=cfg.robot,
            data_format=cfg.data_format,
            object_name=cfg.object_name,
            input_fps=30,
            output_fps=cfg.output_fps,
            line_range=None,
            has_dynamic_object=cfg.has_dynamic_object,
            output_name=str(out_path),
            once=cfg.once,
            no_viewer=cfg.no_viewer,
            realtime=cfg.realtime,
            use_omniretarget_data=cfg.use_omniretarget_data,
        )

        print(f"[{i}/{len(inputs)}] convert: {in_path.name} -> {out_path.name}")
        main(args)


if __name__ == "__main__":
    run(tyro.cli(BatchConvertConfig))
