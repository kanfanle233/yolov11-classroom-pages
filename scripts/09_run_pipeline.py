import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from paths import PROJECT_ROOT
except Exception:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

from contracts.schemas import SCHEMA_VERSION
from verifier.dataset import build_training_samples, convert_to_contract_samples, save_training_samples


def run_step(py_exe: str, script: Path, args: List[str]) -> None:
    cmd = [py_exe, str(script)] + args
    print("\n" + "=" * 80)
    print(f"[RUN] {script.name} {' '.join(args)}")
    print("=" * 80)
    r = subprocess.run(cmd, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"Step failed: {script.name} (exit={r.returncode})")


def file_is_fresh(output_path: Path, inputs: List[Path], min_bytes: int = 10) -> bool:
    if not output_path.exists() or not output_path.is_file():
        return False
    try:
        if output_path.stat().st_size < min_bytes:
            return False
        out_m = output_path.stat().st_mtime
        for ip in inputs:
            if ip.exists() and ip.stat().st_mtime > out_m:
                return False
        return True
    except Exception:
        return False


def maybe_run(step_id, step_name, outputs, inputs, force, from_step, run_fn, min_bytes=10):
    if step_id < from_step:
        print(f"[SKIP] step{step_id:03d} {step_name}")
        return
    if force:
        print(f"[FORCE] step{step_id:03d} {step_name}")
        run_fn()
        return

    all_fresh = all(file_is_fresh(o, inputs, min_bytes=min_bytes) for o in outputs)
    if all_fresh:
        print(f"[CACHE] step{step_id:03d} {step_name}")
    else:
        print(f"[DO] step{step_id:03d} {step_name}")
        run_fn()


def resolve_model_or_fail(model_arg: str) -> str:
    p = Path(model_arg)
    if p.is_absolute() or len(p.parts) > 1:
        q = p if p.is_absolute() else (PROJECT_ROOT / p).resolve()
        if not q.exists():
            raise FileNotFoundError(f"Model not found: {q}")
        return str(q)

    local = (PROJECT_ROOT / model_arg).resolve()
    if local.exists():
        return str(local)
    cwd_local = (Path.cwd() / model_arg).resolve()
    if cwd_local.exists():
        return str(cwd_local)
    return model_arg


def resolve_video_path(video_arg: str) -> Path:
    candidate = Path(video_arg)
    if candidate.is_absolute():
        return candidate.resolve()
    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (PROJECT_ROOT / candidate).resolve()


def _resolve_out_dir(args, video_path: Path, base_dir: Path) -> Path:
    if args.out_dir:
        return resolve_video_path(args.out_dir)
    if args.output_root:
        root = resolve_video_path(args.output_root)
        view_dir = root / args.view if args.view else root
        case_id = args.case_id or video_path.stem
        if args.video_id:
            vid = args.video_id
        else:
            view_code = args.view_code or args.view or "case"
            vid = f"{view_code}__{case_id}"
        return view_dir / vid
    return (base_dir / "output" / (args.name or video_path.stem)).resolve()


def _auto_pose_model_path(user_arg: str) -> str:
    if user_arg and user_arg.lower() != "auto":
        return resolve_model_or_fail(user_arg)
    candidates = [
        PROJECT_ROOT / "yolo11x-pose.pt",
        PROJECT_ROOT / "yolo11l-pose.pt",
        PROJECT_ROOT / "yolo11m-pose.pt",
        PROJECT_ROOT / "models" / "classroom_yolo_enhanced.pt",
        PROJECT_ROOT / "yolo11s-pose.pt",
        PROJECT_ROOT / "yolo11n-pose.pt",
    ]
    for cand in candidates:
        cand = cand.resolve()
        if cand.exists():
            print(f"[INFO] using pose model: {cand}")
            return str(cand)
    return resolve_model_or_fail("yolo11x-pose.pt")


def _is_placeholder_transcript(path: Path) -> bool:
    if not path.exists():
        return True
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = str(obj.get("text", ""))
                if "[ASR_EMPTY:" in text:
                    return True
                return False
    except Exception:
        return True
    return True


def _iter_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                out.append(row)
    return out


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _write_pipeline_manifest(
    *,
    out_dir: Path,
    case_id: str,
    video_id: str,
    artifacts: Dict[str, Path],
    config_snapshot: Dict[str, Any],
) -> None:
    payload = {
        "case_id": case_id,
        "video_id": video_id,
        "schema_version": SCHEMA_VERSION,
        "artifacts": {k: str(v) for k, v in artifacts.items() if v.exists()},
        "config_snapshot": config_snapshot,
    }
    _write_json(out_dir / "pipeline_manifest.json", payload)


def main() -> None:
    base_dir = PROJECT_ROOT
    scripts_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Formal pipeline (fixed schema): pose -> uq -> actions -> asr -> event queries -> align -> verifier"
    )
    parser.add_argument("--video", type=str, default="data/videos/demo3.mp4")
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--output_root", type=str, default="")
    parser.add_argument("--view", type=str, default="")
    parser.add_argument("--view_code", type=str, default="")
    parser.add_argument("--case_id", type=str, default="")
    parser.add_argument("--video_id", type=str, default="")
    parser.add_argument("--py", type=str, default=sys.executable)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--from_step", type=int, default=1)
    parser.add_argument("--skip_vis", action="store_true")
    parser.add_argument("--export_compat", type=int, default=1)

    parser.add_argument("--pose_model", type=str, default="auto")
    parser.add_argument("--det_model", type=str, default="yolo11x.pt")
    parser.add_argument("--action_model", type=str, default="")
    parser.add_argument("--action_mode", choices=["auto", "slowfast", "rules"], default="auto")
    parser.add_argument("--asr_backend", choices=["auto", "api", "whisper", "openai"], default="auto")
    parser.add_argument("--asr_model", type=str, default="paraformer-realtime-v1")
    parser.add_argument("--asr_lang", type=str, default="zh")
    parser.add_argument("--whisper_model", type=str, default="small")
    parser.add_argument("--whisper_device", type=str, default="cpu")
    parser.add_argument("--whisper_compute_type", type=str, default="int8")
    parser.add_argument("--whisper_beam_size", type=int, default=5)
    parser.add_argument("--whisper_vad_filter", type=int, default=0)
    parser.add_argument("--whisper_condition_on_previous_text", type=int, default=0)
    parser.add_argument("--whisper_retry_on_fail", type=int, default=1)
    parser.add_argument("--whisper_retry_compute_type", type=str, default="int8_float16")
    parser.add_argument("--whisper_fallback_device", type=str, default="cpu")
    parser.add_argument("--whisper_fallback_compute_type", type=str, default="int8")
    parser.add_argument("--openai_asr_model", type=str, default="gpt-4o-mini-transcribe")
    parser.add_argument("--openai_api_key_env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--openai_timeout_sec", type=float, default=180.0)
    parser.add_argument("--openai_max_file_mb", type=float, default=24.5)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--train_verifier", type=int, default=0, help="1=train verifier model before step07")
    parser.add_argument("--verifier_model", type=str, default="", help="path to verifier checkpoint (.pt)")
    parser.add_argument("--eval_target_field", type=str, default="auto")
    parser.add_argument("--calibration_target_field", type=str, default="auto")
    parser.add_argument("--calibration_prob_field", type=str, default="p_match")
    parser.add_argument("--calibration_num_bins", type=int, default=10)
    parser.add_argument("--disable_temperature_scaling", type=int, default=0)

    # Legacy compatibility args (accepted but no longer on default chain).
    parser.add_argument("--fuse_action_obj", type=int, default=0)
    parser.add_argument("--fuse_window", type=float, default=0.8)
    parser.add_argument("--fuse_alpha", type=float, default=0.75)
    parser.add_argument("--fuse_beta", type=float, default=0.25)
    parser.add_argument("--enable_object_evidence", type=int, default=0)
    parser.add_argument("--object_conf", type=float, default=0.10)
    parser.add_argument("--object_iou", type=float, default=0.50)
    parser.add_argument("--object_imgsz", type=int, default=1280)
    parser.add_argument("--object_stride", type=int, default=1)
    parser.add_argument("--object_classes", type=str, default="67,73,63")
    parser.add_argument("--enable_behavior_det", type=int, default=0)
    parser.add_argument("--behavior_det_model", type=str, default="runs/detect/case_yolo_train/weights/best.pt")
    parser.add_argument("--behavior_det_conf", type=float, default=0.25)
    parser.add_argument("--behavior_det_iou", type=float, default=0.50)
    parser.add_argument("--behavior_det_imgsz", type=int, default=832)
    parser.add_argument("--behavior_det_stride", type=int, default=1)
    parser.add_argument("--behavior_det_device", type=str, default="")
    parser.add_argument("--behavior_action_mode", choices=["off", "append", "behavior_only"], default="off")
    parser.add_argument("--behavior_track_iou_thres", type=float, default=0.30)
    parser.add_argument("--behavior_track_max_gap", type=int, default=3)
    parser.add_argument("--behavior_extra_track_offset", type=int, default=100000)
    parser.add_argument("--enable_peer_aware", type=int, default=0)
    parser.add_argument("--peer_radius", type=float, default=0.15)
    parser.add_argument("--interaction_model", choices=["igformer", "legacy"], default="igformer")
    parser.add_argument("--enable_mllm", type=int, default=0)
    parser.add_argument("--interpolate_occluded", type=int, default=0)
    parser.add_argument("--occlusion_conf_thres", type=float, default=0.2)
    parser.add_argument("--track_min_frames", type=int, default=75)
    parser.add_argument("--track_min_frames_ratio", type=float, default=0.0)
    parser.add_argument("--track_min_frames_min", type=int, default=10)
    parser.add_argument("--viz_gap_tol", type=float, default=1.5)
    parser.add_argument("--viz_min_dur", type=float, default=0.1)
    args = parser.parse_args()

    video_path = resolve_video_path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")
    name = args.name if args.name else (args.case_id or video_path.stem)
    out_dir = _resolve_out_dir(args, video_path, base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pose_keypoints = out_dir / "pose_keypoints_v2.jsonl"
    pose_tracks = out_dir / "pose_tracks_smooth.jsonl"
    pose_uq = out_dir / "pose_tracks_smooth_uq.jsonl"
    objects_jsonl = out_dir / "objects.jsonl"
    behavior_det_jsonl = out_dir / "behavior_det.jsonl"
    behavior_actions_jsonl = out_dir / "actions.behavior.jsonl"
    actions_behavior_aug_jsonl = out_dir / "actions.behavior_aug.jsonl"
    actions_raw_jsonl = out_dir / "actions.raw.jsonl"
    actions_jsonl = out_dir / "actions.jsonl"
    transcript_jsonl = out_dir / "transcript.jsonl"
    event_queries_jsonl = out_dir / "event_queries.jsonl"
    aligned_json = out_dir / "align_multimodal.json"
    verified_events_jsonl = out_dir / "verified_events.jsonl"
    verifier_model = Path(args.verifier_model).resolve() if args.verifier_model else (out_dir / "verifier.pt")
    verifier_report_raw = out_dir / "verifier_report.raw.json"
    verifier_samples_raw = out_dir / "verifier_samples.raw.jsonl"
    verifier_samples_train = out_dir / "verifier_samples_train.jsonl"
    verifier_eval_report = out_dir / "verifier_eval_report.json"
    verifier_calibration_report = out_dir / "verifier_calibration_report.json"
    verifier_reliability_diagram = out_dir / "verifier_reliability_diagram.svg"
    per_person_json = out_dir / "per_person_sequences.json"
    timeline_png = out_dir / "timeline_chart.png"
    timeline_json = out_dir / "timeline_chart.json"

    py = args.py
    yolo_cfg_dir = (base_dir / ".ultralytics").resolve()
    yolo_cfg_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(yolo_cfg_dir)
    print(f"[INFO] YOLO_CONFIG_DIR={yolo_cfg_dir}")

    pose_model_abs = _auto_pose_model_path(args.pose_model)
    det_model_abs = resolve_model_or_fail(args.det_model)
    object_evidence_enabled = int(args.enable_object_evidence) == 1 or int(args.fuse_action_obj) == 1
    behavior_det_enabled = int(args.enable_behavior_det) == 1
    behavior_action_mode = str(args.behavior_action_mode).strip().lower()
    behavior_det_model_abs = resolve_model_or_fail(args.behavior_det_model) if behavior_det_enabled else ""

    maybe_run(
        2,
        "pose keypoints",
        [pose_keypoints],
        [video_path],
        args.force,
        args.from_step,
        lambda: run_step(
            py,
            scripts_dir / "02_export_keypoints_jsonl.py",
            [
                "--video",
                str(video_path),
                "--out",
                str(pose_keypoints),
                "--model",
                pose_model_abs,
                "--interpolate_occluded",
                str(int(args.interpolate_occluded)),
                "--occlusion_conf_thres",
                str(float(args.occlusion_conf_thres)),
            ],
        ),
    )

    maybe_run(
        4,
        "track and smooth",
        [pose_tracks],
        [pose_keypoints],
        args.force,
        args.from_step,
        lambda: run_step(
            py,
            scripts_dir / "03_track_and_smooth.py",
            [
                "--in",
                str(pose_keypoints),
                "--video",
                str(video_path),
                "--out",
                str(pose_tracks),
                "--track_min_frames",
                str(int(args.track_min_frames)),
                "--track_min_frames_ratio",
                str(float(args.track_min_frames_ratio)),
                "--track_min_frames_min",
                str(int(args.track_min_frames_min)),
            ],
        ),
    )

    maybe_run(
        35,
        "estimate uncertainty",
        [pose_uq],
        [pose_tracks],
        args.force,
        args.from_step,
        lambda: run_step(
            py,
            scripts_dir / "03c_estimate_track_uncertainty.py",
            ["--in", str(pose_tracks), "--out", str(pose_uq), "--validate", "1"],
        ),
    )

    if object_evidence_enabled:
        maybe_run(
            45,
            "object detection evidence",
            [objects_jsonl],
            [video_path],
            args.force,
            args.from_step,
            lambda: run_step(
                py,
                scripts_dir / "02b_export_objects_jsonl.py",
                [
                    "--video",
                    str(video_path),
                    "--out",
                    str(objects_jsonl),
                    "--model",
                    str(det_model_abs),
                    "--conf",
                    str(float(args.object_conf)),
                    "--iou",
                    str(float(args.object_iou)),
                    "--imgsz",
                    str(int(args.object_imgsz)),
                    "--stride",
                    str(int(args.object_stride)),
                    "--classes",
                    str(args.object_classes),
                ],
            ),
        )

    def _run_actions():
        action_out = actions_raw_jsonl if object_evidence_enabled else actions_jsonl
        cmd = [
            "--video",
            str(video_path),
            "--pose",
            str(pose_tracks),
            "--out",
            str(action_out),
            "--model_mode",
            str(args.action_mode),
        ]
        if args.action_model:
            cmd += ["--model_weight", resolve_model_or_fail(args.action_model)]
        run_step(py, scripts_dir / "05_slowfast_actions.py", cmd)

    maybe_run(
        5,
        "action recognition",
        [actions_raw_jsonl if object_evidence_enabled else actions_jsonl],
        [video_path, pose_tracks],
        args.force,
        args.from_step,
        _run_actions,
    )

    if object_evidence_enabled:
        maybe_run(
            55,
            "fuse actions with object evidence",
            [actions_jsonl],
            [actions_raw_jsonl, objects_jsonl],
            args.force,
            args.from_step,
            lambda: run_step(
                py,
                scripts_dir / "05b_fuse_actions_with_objects.py",
                [
                    "--actions",
                    str(actions_raw_jsonl),
                    "--objects",
                    str(objects_jsonl),
                    "--out",
                    str(actions_jsonl),
                    "--window",
                    str(float(args.fuse_window)),
                    "--alpha",
                    str(float(args.fuse_alpha)),
                    "--beta",
                    str(float(args.fuse_beta)),
                ],
            ),
        )

    if behavior_det_enabled:
        maybe_run(
            46,
            "behavior detection evidence",
            [behavior_det_jsonl],
            [video_path],
            args.force,
            args.from_step,
            lambda: run_step(
                py,
                scripts_dir / "02d_export_behavior_det_jsonl.py",
                [
                    "--video",
                    str(video_path),
                    "--out",
                    str(behavior_det_jsonl),
                    "--model",
                    str(behavior_det_model_abs),
                    "--conf",
                    str(float(args.behavior_det_conf)),
                    "--iou",
                    str(float(args.behavior_det_iou)),
                    "--imgsz",
                    str(int(args.behavior_det_imgsz)),
                    "--stride",
                    str(int(args.behavior_det_stride)),
                    "--device",
                    str(args.behavior_det_device),
                ],
            ),
        )
        maybe_run(
            47,
            "convert behavior det to actions",
            [behavior_actions_jsonl],
            [behavior_det_jsonl],
            args.force,
            args.from_step,
            lambda: run_step(
                py,
                scripts_dir / "05c_behavior_det_to_actions.py",
                [
                    "--in",
                    str(behavior_det_jsonl),
                    "--out",
                    str(behavior_actions_jsonl),
                    "--fps",
                    str(float(args.fps)),
                    "--iou_thres",
                    str(float(args.behavior_track_iou_thres)),
                    "--max_gap_frames",
                    str(int(args.behavior_track_max_gap)),
                ],
            ),
        )

    actions_for_downstream = actions_jsonl
    if behavior_det_enabled and behavior_action_mode == "behavior_only":
        actions_for_downstream = behavior_actions_jsonl
    elif behavior_det_enabled and behavior_action_mode == "append":
        maybe_run(
            58,
            "merge rule actions with behavior actions",
            [actions_behavior_aug_jsonl],
            [actions_jsonl, behavior_actions_jsonl],
            args.force,
            args.from_step,
            lambda: run_step(
                py,
                scripts_dir / "05d_merge_action_sources.py",
                [
                    "--primary",
                    str(actions_jsonl),
                    "--extra",
                    str(behavior_actions_jsonl),
                    "--out",
                    str(actions_behavior_aug_jsonl),
                    "--extra_track_offset",
                    str(int(args.behavior_extra_track_offset)),
                ],
            ),
        )
        actions_for_downstream = actions_behavior_aug_jsonl

    def _run_asr():
        def _run_api() -> None:
            run_step(
                py,
                scripts_dir / "06_api_asr_realtime.py",
                ["--video", str(video_path), "--out_dir", str(out_dir), "--asr_model", str(args.asr_model)],
            )

        def _unlink_transcript_if_exists() -> None:
            try:
                if transcript_jsonl.exists() and transcript_jsonl.is_file():
                    transcript_jsonl.unlink()
            except Exception:
                pass

        def _run_whisper_once(device: str, compute_type: str) -> None:
            _unlink_transcript_if_exists()
            run_step(
                py,
                scripts_dir / "06_asr_whisper_to_jsonl.py",
                [
                    "--video",
                    str(video_path),
                    "--out_dir",
                    str(out_dir),
                    "--lang",
                    str(args.asr_lang),
                    "--model",
                    str(args.whisper_model),
                    "--device",
                    str(device),
                    "--compute_type",
                    str(compute_type),
                    "--beam_size",
                    str(int(args.whisper_beam_size)),
                    "--vad_filter",
                    str(int(args.whisper_vad_filter)),
                    "--condition_on_previous_text",
                    str(int(args.whisper_condition_on_previous_text)),
                ],
            )

        def _run_whisper() -> None:
            primary_device = str(args.whisper_device).strip().lower()
            primary_compute = str(args.whisper_compute_type).strip()
            try:
                _run_whisper_once(primary_device, primary_compute)
                return
            except Exception as e1:
                if int(args.whisper_retry_on_fail) != 1 or primary_device != "cuda":
                    raise
                print(f"[WARN] whisper primary failed on cuda ({e1}); trying recovery chain...")

            retry_compute = str(args.whisper_retry_compute_type).strip()
            if retry_compute and retry_compute != primary_compute:
                try:
                    _run_whisper_once(primary_device, retry_compute)
                    print(f"[INFO] whisper recovered by retry compute_type={retry_compute}")
                    return
                except Exception as e2:
                    print(f"[WARN] whisper retry compute_type={retry_compute} failed ({e2})")

            fallback_device = str(args.whisper_fallback_device).strip().lower()
            if fallback_device:
                fallback_compute = str(args.whisper_fallback_compute_type).strip() or primary_compute
                _run_whisper_once(fallback_device, fallback_compute)
                print(f"[INFO] whisper recovered by fallback device={fallback_device} compute_type={fallback_compute}")
                return

            raise RuntimeError("whisper recovery chain exhausted")

        def _run_openai() -> None:
            run_step(
                py,
                scripts_dir / "06c_asr_openai_to_jsonl.py",
                [
                    "--video",
                    str(video_path),
                    "--out_dir",
                    str(out_dir),
                    "--model",
                    str(args.openai_asr_model),
                    "--lang",
                    str(args.asr_lang),
                    "--api_key_env",
                    str(args.openai_api_key_env),
                    "--timeout_sec",
                    str(float(args.openai_timeout_sec)),
                    "--max_file_mb",
                    str(float(args.openai_max_file_mb)),
                ],
            )

        if args.asr_backend == "api":
            _run_api()
            return
        if args.asr_backend == "whisper":
            _run_whisper()
            return
        if args.asr_backend == "openai":
            _run_openai()
            return

        # auto: try api first, fallback whisper when still placeholder.
        _run_api()
        if _is_placeholder_transcript(transcript_jsonl):
            print("[INFO] ASR placeholder detected, fallback to whisper...")
            _run_whisper()

    maybe_run(
        6,
        "asr transcript",
        [transcript_jsonl],
        [video_path],
        args.force,
        args.from_step,
        _run_asr,
    )

    maybe_run(
        65,
        "extract event queries",
        [event_queries_jsonl],
        [transcript_jsonl],
        args.force,
        args.from_step,
        lambda: run_step(
            py,
            scripts_dir / "06b_event_query_extraction.py",
            ["--transcript", str(transcript_jsonl), "--out", str(event_queries_jsonl), "--validate", "1"],
        ),
    )

    maybe_run(
        66,
        "adaptive multimodal align",
        [aligned_json],
        [event_queries_jsonl, actions_for_downstream, pose_uq],
        args.force,
        args.from_step,
        lambda: run_step(
            py,
            scripts_dir / "xx_align_multimodal.py",
            [
                "--event_queries",
                str(event_queries_jsonl),
                "--actions",
                str(actions_for_downstream),
                "--pose_uq",
                str(pose_uq),
                "--out",
                str(aligned_json),
            ],
        ),
    )

    contract_samples = convert_to_contract_samples(
        build_training_samples(
            event_queries_path=event_queries_jsonl,
            aligned_path=aligned_json,
            actions_path=actions_for_downstream,
        )
    )
    save_training_samples(verifier_samples_train, contract_samples)

    if int(args.train_verifier) == 1:
        maybe_run(
            67,
            "train verifier",
            [verifier_model, verifier_report_raw],
            [event_queries_jsonl, aligned_json, actions_for_downstream],
            args.force,
            args.from_step,
            lambda: run_step(
                py,
                PROJECT_ROOT / "verifier" / "train.py",
                [
                    "--event_queries",
                    str(event_queries_jsonl),
                    "--aligned",
                    str(aligned_json),
                    "--actions",
                    str(actions_for_downstream),
                    "--out_model",
                    str(verifier_model),
                    "--out_report",
                    str(verifier_report_raw),
                    "--out_samples",
                    str(verifier_samples_raw),
                ],
            ),
        )

        if verifier_samples_raw.exists():
            save_training_samples(verifier_samples_train, convert_to_contract_samples(_iter_jsonl(verifier_samples_raw)))

    maybe_run(
        70,
        "dual verification",
        [verified_events_jsonl],
        [event_queries_jsonl, aligned_json, pose_uq, actions_for_downstream, verifier_model],
        args.force,
        args.from_step,
        lambda: run_step(
            py,
            scripts_dir / "07_dual_verification.py",
            [
                "--actions",
                str(actions_for_downstream),
                "--event_queries",
                str(event_queries_jsonl),
                "--pose_uq",
                str(pose_uq),
                "--aligned",
                str(aligned_json),
                "--out",
                str(verified_events_jsonl),
                "--verifier_model",
                str(verifier_model if verifier_model.exists() else ""),
                "--per_person_out",
                str(per_person_json),
                "--validate",
                "1",
            ],
        ),
    )

    maybe_run(
        71,
        "verifier evaluation report",
        [verifier_eval_report],
        [verified_events_jsonl],
        args.force,
        args.from_step,
        lambda: run_step(
            py,
            PROJECT_ROOT / "verifier" / "eval.py",
            [
                "--verified",
                str(verified_events_jsonl),
                "--out",
                str(verifier_eval_report),
                "--split",
                "val",
                "--target_field",
                str(args.eval_target_field),
            ],
        ),
    )

    maybe_run(
        72,
        "verifier calibration report",
        [verifier_calibration_report],
        [verified_events_jsonl],
        args.force,
        args.from_step,
        lambda: run_step(
            py,
            PROJECT_ROOT / "verifier" / "calibration.py",
            [
                "--verified",
                str(verified_events_jsonl),
                "--out",
                str(verifier_calibration_report),
                "--split",
                "val",
                "--target_field",
                str(args.calibration_target_field),
                "--prob_field",
                str(args.calibration_prob_field),
                "--num_bins",
                str(int(args.calibration_num_bins)),
                "--disable_temperature_scaling",
                str(int(args.disable_temperature_scaling)),
                "--diagram_out",
                str(verifier_reliability_diagram),
            ],
        ),
    )

    if not args.skip_vis:
        maybe_run(
            10,
            "timeline visualization",
            [timeline_png, timeline_json],
            [verified_events_jsonl],
            args.force,
            args.from_step,
            lambda: run_step(
                py,
                scripts_dir / "10_visualize_timeline.py",
                [
                    "--src",
                    str(per_person_json),
                    "--verified_src",
                    str(verified_events_jsonl),
                    "--out",
                    str(timeline_png),
                    "--gap_tol",
                    str(args.viz_gap_tol),
                    "--min_dur",
                    str(args.viz_min_dur),
                    "--fps",
                    str(args.fps),
                ],
            ),
        )

    if int(args.export_compat) == 1:
        case_id = args.case_id or name
        if args.video_id:
            video_id = args.video_id
        else:
            view_code = args.view_code or args.view or "case"
            video_id = f"{view_code}__{case_id}"
        _write_pipeline_manifest(
            out_dir=out_dir,
            case_id=case_id,
            video_id=video_id,
            artifacts={
                "pose_keypoints": pose_keypoints,
                "pose_tracks_smooth": pose_tracks,
                "pose_tracks_smooth_uq": pose_uq,
                "objects": objects_jsonl,
                "behavior_det": behavior_det_jsonl,
                "actions_behavior": behavior_actions_jsonl,
                "actions_behavior_aug": actions_behavior_aug_jsonl,
                "actions_raw": actions_raw_jsonl,
                "actions": actions_jsonl,
                "actions_used_for_align": actions_for_downstream,
                "transcript": transcript_jsonl,
                "event_queries": event_queries_jsonl,
                "align_multimodal": aligned_json,
                "verifier_samples_train": verifier_samples_train,
                "verified_events": verified_events_jsonl,
                "verifier_model": verifier_model,
                "verifier_eval_report": verifier_eval_report,
                "verifier_calibration_report": verifier_calibration_report,
                "verifier_reliability_diagram": verifier_reliability_diagram,
                "per_person_sequences": per_person_json,
                "timeline_chart_png": timeline_png,
                "timeline_chart_json": timeline_json,
            },
            config_snapshot={
                "from_step": int(args.from_step),
                "train_verifier": int(args.train_verifier),
                "action_mode": str(args.action_mode),
                "asr_backend": str(args.asr_backend),
                "asr_model": str(args.asr_model),
                "asr_lang": str(args.asr_lang),
                "whisper_model": str(args.whisper_model),
                "whisper_device": str(args.whisper_device),
                "whisper_compute_type": str(args.whisper_compute_type),
                "whisper_beam_size": int(args.whisper_beam_size),
                "whisper_vad_filter": int(args.whisper_vad_filter),
                "whisper_condition_on_previous_text": int(args.whisper_condition_on_previous_text),
                "whisper_retry_on_fail": int(args.whisper_retry_on_fail),
                "whisper_retry_compute_type": str(args.whisper_retry_compute_type),
                "whisper_fallback_device": str(args.whisper_fallback_device),
                "whisper_fallback_compute_type": str(args.whisper_fallback_compute_type),
                "openai_asr_model": str(args.openai_asr_model),
                "openai_api_key_env": str(args.openai_api_key_env),
                "openai_timeout_sec": float(args.openai_timeout_sec),
                "openai_max_file_mb": float(args.openai_max_file_mb),
                "det_model": str(det_model_abs),
                "object_evidence_enabled": bool(object_evidence_enabled),
                "object_conf": float(args.object_conf),
                "object_iou": float(args.object_iou),
                "object_imgsz": int(args.object_imgsz),
                "object_stride": int(args.object_stride),
                "object_classes": str(args.object_classes),
                "behavior_det_enabled": bool(behavior_det_enabled),
                "behavior_det_model": str(behavior_det_model_abs),
                "behavior_det_conf": float(args.behavior_det_conf),
                "behavior_det_iou": float(args.behavior_det_iou),
                "behavior_det_imgsz": int(args.behavior_det_imgsz),
                "behavior_det_stride": int(args.behavior_det_stride),
                "behavior_det_device": str(args.behavior_det_device),
                "behavior_action_mode": str(behavior_action_mode),
                "behavior_track_iou_thres": float(args.behavior_track_iou_thres),
                "behavior_track_max_gap": int(args.behavior_track_max_gap),
                "behavior_extra_track_offset": int(args.behavior_extra_track_offset),
                "actions_used_for_align": str(actions_for_downstream),
                "fuse_window": float(args.fuse_window),
                "fuse_alpha": float(args.fuse_alpha),
                "fuse_beta": float(args.fuse_beta),
                "interaction_model": str(args.interaction_model),
                "enable_mllm": int(args.enable_mllm),
                "track_min_frames": int(args.track_min_frames),
                "track_min_frames_ratio": float(args.track_min_frames_ratio),
                "track_min_frames_min": int(args.track_min_frames_min),
                "eval_target_field": str(args.eval_target_field),
                "calibration_target_field": str(args.calibration_target_field),
                "calibration_prob_field": str(args.calibration_prob_field),
                "calibration_num_bins": int(args.calibration_num_bins),
                "disable_temperature_scaling": int(args.disable_temperature_scaling),
            },
        )

        # Keep legacy aliases for downstream tools that still consume old names.
        if pose_tracks.exists():
            shutil.copy2(pose_tracks, out_dir / f"{case_id}.jsonl")
        if timeline_json.exists():
            shutil.copy2(timeline_json, out_dir / "timeline_viz.json")

    print("\n[DONE] formal pipeline completed")
    print(f"Output dir             : {out_dir}")
    print(f"Pose tracks UQ         : {pose_uq}")
    if object_evidence_enabled:
        print(f"Object evidence        : {objects_jsonl}")
    if behavior_det_enabled:
        print(f"Behavior detections    : {behavior_det_jsonl}")
        print(f"Behavior actions       : {behavior_actions_jsonl}")
        print(f"Behavior action mode   : {behavior_action_mode}")
        print(f"Actions for align      : {actions_for_downstream}")
    print(f"Event queries          : {event_queries_jsonl}")
    print(f"Verified events        : {verified_events_jsonl}")
    print(f"Verifier eval report   : {verifier_eval_report}")
    print(f"Verifier calibration   : {verifier_calibration_report}")
    print(f"Reliability diagram    : {verifier_reliability_diagram}")
    print(f"Verifier model         : {verifier_model if verifier_model.exists() else 'not_used'}")
    print(f"Timeline chart         : {timeline_png}")


if __name__ == "__main__":
    main()
