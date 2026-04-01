import argparse
import json
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

from contracts.schemas import ARTIFACT_VERSION, SCHEMA_VERSION


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
    enhanced = (PROJECT_ROOT / "models" / "classroom_yolo_enhanced.pt").resolve()
    if enhanced.exists():
        print(f"[INFO] using enhanced classroom pose model: {enhanced}")
        return str(enhanced)
    return resolve_model_or_fail("yolo11n-pose.pt")


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


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        return obj
    return {}


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


def _convert_training_samples_to_contract(raw_samples_path: Path, contract_samples_path: Path) -> None:
    rows = _iter_jsonl(raw_samples_path)
    converted: List[Dict[str, Any]] = []
    for row in rows:
        event_id = str(row.get("event_id", row.get("query_id", "")))
        target = int(row.get("target", 0))
        sample_type = str(row.get("sample_type", "semantic_mismatch"))
        converted.append(
            {
                "sample_id": str(row.get("sample_id", "")),
                "event_id": event_id,
                "sample_type": sample_type,
                "query_text": str(row.get("query_text", "")),
                "event_type": str(row.get("event_type", "unknown")),
                "track_id": int(row.get("track_id", -1)),
                "clip_start": float(row.get("clip_start", row.get("window_start", 0.0))),
                "clip_end": float(row.get("clip_end", row.get("window_end", 0.0))),
                "target_label": "match" if target == 1 else "mismatch",
                "negative_kind": "" if sample_type == "positive" else sample_type,
                "provenance": {
                    "source": "verifier.dataset",
                    "legacy_fields": {
                        "overlap": float(row.get("overlap", 0.0)),
                        "action_confidence": float(row.get("action_confidence", 0.0)),
                        "uq_score": float(row.get("uq_score", 0.0)),
                        "text_score": float(row.get("text_score", 0.0)),
                    },
                },
            }
        )
    contract_samples_path.parent.mkdir(parents=True, exist_ok=True)
    with contract_samples_path.open("w", encoding="utf-8") as f:
        for row in converted:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _convert_reports_to_contract(
    raw_report_path: Path,
    eval_report_path: Path,
    calibration_report_path: Path,
) -> None:
    report = _load_json(raw_report_path)
    train_metrics = report.get("train_metrics", {}) if isinstance(report.get("train_metrics"), dict) else {}
    val_metrics = report.get("val_metrics", {}) if isinstance(report.get("val_metrics"), dict) else {}
    runtime_cfg = report.get("runtime_config", {}) if isinstance(report.get("runtime_config"), dict) else {}
    sample_types = report.get("sample_types", {}) if isinstance(report.get("sample_types"), dict) else {}

    counts = {
        "total": int(report.get("num_samples", 0)),
        "train": int(report.get("num_train", 0)),
        "val": int(report.get("num_val", 0)),
        "sample_types": sample_types,
    }
    eval_report = {
        "split": "val",
        "counts": counts,
        "metrics": {
            "train": train_metrics,
            "val": val_metrics,
        },
        "config": runtime_cfg,
        "artifact_version": ARTIFACT_VERSION,
    }
    calibration_report = {
        "split": "val",
        "ece": float(val_metrics.get("ece", 0.0)),
        "brier": float(val_metrics.get("brier", 0.0)),
        "temperature": float(runtime_cfg.get("temperature", 1.0)),
        "bin_stats": [],
        "artifact_version": ARTIFACT_VERSION,
    }
    _write_json(eval_report_path, eval_report)
    _write_json(calibration_report_path, calibration_report)


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
    parser.add_argument("--det_model", type=str, default="yolo11n.pt")
    parser.add_argument("--action_model", type=str, default="")
    parser.add_argument("--action_mode", choices=["auto", "slowfast", "rules"], default="auto")
    parser.add_argument("--asr_backend", choices=["auto", "api", "whisper"], default="auto")
    parser.add_argument("--asr_model", type=str, default="paraformer-realtime-v1")
    parser.add_argument("--asr_lang", type=str, default="zh")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--train_verifier", type=int, default=0, help="1=train verifier model before step07")
    parser.add_argument("--verifier_model", type=str, default="", help="path to verifier checkpoint (.pt)")

    # Legacy compatibility args (accepted but no longer on default chain).
    parser.add_argument("--fuse_action_obj", type=int, default=0)
    parser.add_argument("--fuse_window", type=float, default=0.8)
    parser.add_argument("--fuse_alpha", type=float, default=0.75)
    parser.add_argument("--fuse_beta", type=float, default=0.25)
    parser.add_argument("--enable_peer_aware", type=int, default=0)
    parser.add_argument("--peer_radius", type=float, default=0.15)
    parser.add_argument("--interaction_model", choices=["igformer", "legacy"], default="igformer")
    parser.add_argument("--enable_mllm", type=int, default=0)
    parser.add_argument("--interpolate_occluded", type=int, default=0)
    parser.add_argument("--occlusion_conf_thres", type=float, default=0.2)
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
    per_person_json = out_dir / "per_person_sequences.json"
    timeline_png = out_dir / "timeline_chart.png"
    timeline_json = out_dir / "timeline_chart.json"

    py = args.py
    pose_model_abs = _auto_pose_model_path(args.pose_model)
    _ = resolve_model_or_fail(args.det_model)

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
            ["--in", str(pose_keypoints), "--video", str(video_path), "--out", str(pose_tracks)],
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

    def _run_actions():
        cmd = [
            "--video",
            str(video_path),
            "--pose",
            str(pose_tracks),
            "--out",
            str(actions_jsonl),
            "--model_mode",
            str(args.action_mode),
        ]
        if args.action_model:
            cmd += ["--model_weight", resolve_model_or_fail(args.action_model)]
        run_step(py, scripts_dir / "05_slowfast_actions.py", cmd)

    maybe_run(
        5,
        "action recognition",
        [actions_jsonl],
        [video_path, pose_tracks],
        args.force,
        args.from_step,
        _run_actions,
    )

    def _run_asr():
        if args.asr_backend == "api":
            run_step(
                py,
                scripts_dir / "06_api_asr_realtime.py",
                ["--video", str(video_path), "--out_dir", str(out_dir), "--asr_model", str(args.asr_model)],
            )
            return
        if args.asr_backend == "whisper":
            run_step(
                py,
                scripts_dir / "06_asr_whisper_to_jsonl.py",
                ["--video", str(video_path), "--out_dir", str(out_dir), "--lang", str(args.asr_lang)],
            )
            return

        # auto: try api first, fallback whisper when still placeholder.
        run_step(
            py,
            scripts_dir / "06_api_asr_realtime.py",
            ["--video", str(video_path), "--out_dir", str(out_dir), "--asr_model", str(args.asr_model)],
        )
        if _is_placeholder_transcript(transcript_jsonl):
            print("[INFO] ASR placeholder detected, fallback to whisper...")
            run_step(
                py,
                scripts_dir / "06_asr_whisper_to_jsonl.py",
                ["--video", str(video_path), "--out_dir", str(out_dir), "--lang", str(args.asr_lang)],
            )

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
        [event_queries_jsonl, actions_jsonl, pose_uq],
        args.force,
        args.from_step,
        lambda: run_step(
            py,
            scripts_dir / "xx_align_multimodal.py",
            [
                "--event_queries",
                str(event_queries_jsonl),
                "--actions",
                str(actions_jsonl),
                "--pose_uq",
                str(pose_uq),
                "--out",
                str(aligned_json),
            ],
        ),
    )

    if int(args.train_verifier) == 1:
        maybe_run(
            67,
            "train verifier",
            [verifier_model, verifier_report_raw],
            [event_queries_jsonl, aligned_json, actions_jsonl],
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
                    str(actions_jsonl),
                    "--out_model",
                    str(verifier_model),
                    "--out_report",
                    str(verifier_report_raw),
                    "--out_samples",
                    str(verifier_samples_raw),
                ],
            ),
        )

        if verifier_report_raw.exists():
            _convert_reports_to_contract(
                raw_report_path=verifier_report_raw,
                eval_report_path=verifier_eval_report,
                calibration_report_path=verifier_calibration_report,
            )
        if verifier_samples_raw.exists():
            _convert_training_samples_to_contract(
                raw_samples_path=verifier_samples_raw,
                contract_samples_path=verifier_samples_train,
            )

    maybe_run(
        70,
        "dual verification",
        [verified_events_jsonl],
        [event_queries_jsonl, aligned_json, pose_uq, actions_jsonl, verifier_model],
        args.force,
        args.from_step,
        lambda: run_step(
            py,
            scripts_dir / "07_dual_verification.py",
            [
                "--actions",
                str(actions_jsonl),
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
                "actions": actions_jsonl,
                "transcript": transcript_jsonl,
                "event_queries": event_queries_jsonl,
                "align_multimodal": aligned_json,
                "verifier_samples_train": verifier_samples_train,
                "verified_events": verified_events_jsonl,
                "verifier_model": verifier_model,
                "verifier_eval_report": verifier_eval_report,
                "verifier_calibration_report": verifier_calibration_report,
                "per_person_sequences": per_person_json,
                "timeline_chart_png": timeline_png,
                "timeline_chart_json": timeline_json,
            },
            config_snapshot={
                "from_step": int(args.from_step),
                "train_verifier": int(args.train_verifier),
                "action_mode": str(args.action_mode),
                "asr_backend": str(args.asr_backend),
                "interaction_model": str(args.interaction_model),
                "enable_mllm": int(args.enable_mllm),
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
    print(f"Event queries          : {event_queries_jsonl}")
    print(f"Verified events        : {verified_events_jsonl}")
    print(f"Verifier model         : {verifier_model if verifier_model.exists() else 'not_used'}")
    print(f"Timeline chart         : {timeline_png}")


if __name__ == "__main__":
    main()
