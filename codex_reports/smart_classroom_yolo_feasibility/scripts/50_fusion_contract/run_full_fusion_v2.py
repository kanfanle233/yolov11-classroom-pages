from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from fusion_utils import resolve_path, resolve_repo_root


def _parse_extra(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError("--extra_args_json must be a JSON object.")
    return obj


def _extend_args(cmd: List[str], args: Dict[str, Any]) -> None:
    for key in sorted(args.keys()):
        cmd.append(f"--{key}")
        val = args[key]
        if isinstance(val, bool):
            cmd.append("1" if val else "0")
        else:
            cmd.append(str(val))


def build_command(args: argparse.Namespace, repo_root: Path) -> List[str]:
    py = str(resolve_path(repo_root, args.py))
    cmd = [
        py,
        str(repo_root / "scripts" / "09_run_pipeline.py"),
        "--video",
        str(args.video),
        "--out_dir",
        str(resolve_path(repo_root, args.out_dir)),
        "--py",
        py,
        "--asr_backend",
        str(args.asr_backend),
        "--pose_model",
        str(args.pose_model),
        "--det_model",
        str(args.det_model),
        "--enable_object_evidence",
        str(int(args.enable_object_evidence)),
        "--enable_behavior_det",
        "1",
        "--behavior_det_model",
        str(args.behavior_det_model),
        "--behavior_action_mode",
        "append",
        "--fusion_contract_v2",
        "1",
        "--semantic_taxonomy",
        str(args.semantic_taxonomy),
        "--from_step",
        str(int(args.from_step)),
        "--export_compat",
        "1",
        "--whisper_model",
        str(args.whisper_model),
        "--whisper_device",
        str(args.whisper_device),
        "--whisper_compute_type",
        str(args.whisper_compute_type),
        "--whisper_beam_size",
        str(int(args.whisper_beam_size)),
        "--whisper_min_avg_logprob",
        str(args.whisper_min_avg_logprob),
        "--whisper_max_no_speech_prob",
        str(args.whisper_max_no_speech_prob),
    ]
    if int(args.skip_vis) == 1:
        cmd.append("--skip_vis")
    _extend_args(cmd, _parse_extra(args.extra_args_json))
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full fusion_contract_v2 classroom pipeline.")
    parser.add_argument("--video", default="data/智慧课堂学生行为数据集/正方视角/001.mp4", type=str)
    parser.add_argument("--out_dir", default="output/codex_reports/run_full_fusion_v2_001/full_integration_001", type=str)
    parser.add_argument("--py", default=sys.executable, type=str)
    parser.add_argument("--asr_backend", default="whisper", type=str)
    parser.add_argument("--pose_model", default="auto", type=str)
    parser.add_argument("--det_model", default="yolo11x.pt", type=str)
    parser.add_argument("--behavior_det_model", default="runs/detect/official_yolo11s_detect_e150_v1/weights/best.pt", type=str)
    parser.add_argument(
        "--semantic_taxonomy",
        default="codex_reports/smart_classroom_yolo_feasibility/profiles/action_semantics_8class.yaml",
        type=str,
    )
    parser.add_argument("--enable_object_evidence", default=1, type=int)
    parser.add_argument("--from_step", default=2, type=int)
    parser.add_argument("--skip_vis", default=0, type=int)
    parser.add_argument("--whisper_model", default="medium", type=str)
    parser.add_argument("--whisper_device", default="cuda", type=str)
    parser.add_argument("--whisper_compute_type", default="float16", type=str)
    parser.add_argument("--whisper_beam_size", default=10, type=int)
    parser.add_argument("--whisper_min_avg_logprob", default="-0.6", type=str)
    parser.add_argument("--whisper_max_no_speech_prob", default="0.65", type=str)
    parser.add_argument("--extra_args_json", default="", type=str)
    parser.add_argument("--emit_only", default=1, type=int)
    parser.add_argument("--dry_run", default=0, type=int)
    parser.add_argument("--print_json", default=1, type=int)
    args = parser.parse_args()

    repo_root = resolve_repo_root(Path(__file__))
    cmd = build_command(args, repo_root)
    payload = {
        "stage": "run_full_fusion_v2",
        "command": cmd,
        "output_dir": str(resolve_path(repo_root, args.out_dir)),
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
        raise RuntimeError(f"fusion v2 pipeline failed, exit={ret.returncode}")


if __name__ == "__main__":
    main()
