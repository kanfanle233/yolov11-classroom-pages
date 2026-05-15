from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def _resolve_repo_root(anchor: Path) -> Path:
    for p in [anchor] + list(anchor.parents):
        if (p / "data").exists() and (p / "scripts").exists():
            return p
    raise RuntimeError(f"Cannot resolve repo root from: {anchor}")


def _resolve_path(repo_root: Path, raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def _parse_extra_args(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError("--extra_args_json must be a JSON object.")
    return obj


def _to_arg_list(extra: Dict[str, Any]) -> List[str]:
    args: List[str] = []
    for key in sorted(extra.keys()):
        val = extra[key]
        args.append(f"--{key}")
        if isinstance(val, bool):
            args.append("1" if val else "0")
        else:
            args.append(str(val))
    return args


def expected_outputs(out_dir: Path) -> List[str]:
    required = [
        "pose_keypoints_v2.jsonl",
        "pose_tracks_smooth.jsonl",
        "transcript.jsonl",
        "event_queries.jsonl",
        "align_multimodal.json",
        "verified_events.jsonl",
        "pipeline_manifest.json",
    ]
    return [str((out_dir / name).resolve()) for name in required]


def build_command(args: argparse.Namespace, repo_root: Path) -> List[str]:
    py = str(_resolve_path(repo_root, args.py))
    script = repo_root / "scripts" / "09_run_pipeline.py"
    if not script.exists():
        raise FileNotFoundError(f"Pipeline script not found: {script}")
    out_dir = str(_resolve_path(repo_root, args.out_dir))

    cmd: List[str] = [
        py,
        str(script),
        "--video",
        str(args.video),
        "--out_dir",
        out_dir,
        "--asr_backend",
        str(args.asr_backend),
        "--pose_model",
        str(args.pose_model),
        "--det_model",
        str(args.det_model),
        "--enable_behavior_det",
        str(int(args.enable_behavior_det)),
        "--behavior_det_model",
        str(args.behavior_det_model),
        "--from_step",
        str(int(args.from_step)),
        "--skip_vis",
        "--export_compat",
        str(int(args.export_compat)),
    ]
    cmd.extend(_to_arg_list(_parse_extra_args(args.extra_args_json)))
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Wrapper for full integration pipeline.")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--asr_backend", type=str, default="whisper")
    parser.add_argument("--pose_model", type=str, default="auto")
    parser.add_argument("--det_model", type=str, default="yolo11x.pt")
    parser.add_argument("--enable_behavior_det", type=int, default=1)
    parser.add_argument("--behavior_det_model", type=str, default="runs/detect/case_yolo_train/weights/best.pt")
    parser.add_argument("--from_step", type=int, default=2)
    parser.add_argument("--export_compat", type=int, default=1)
    parser.add_argument("--extra_args_json", type=str, default="")
    parser.add_argument("--py", type=str, default=sys.executable)
    parser.add_argument("--emit_only", type=int, default=1)
    parser.add_argument("--dry_run", type=int, default=0)
    parser.add_argument("--print_json", type=int, default=1)
    args = parser.parse_args()

    repo_root = _resolve_repo_root(Path(__file__).resolve())
    cmd = build_command(args, repo_root)
    out_dir = _resolve_path(repo_root, args.out_dir)
    payload = {
        "stage": "infer_full_pipeline",
        "command": cmd,
        "output_dir": str(out_dir),
        "expected_outputs": expected_outputs(out_dir),
        "emit_only": int(args.emit_only),
        "dry_run": int(args.dry_run),
    }

    if int(args.print_json) == 1:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(" ".join(cmd))

    if int(args.emit_only) == 1 or int(args.dry_run) == 1:
        return

    ret = subprocess.run(cmd, check=False)
    if ret.returncode != 0:
        raise RuntimeError(f"Integration wrapper failed, exit={ret.returncode}")


if __name__ == "__main__":
    main()
