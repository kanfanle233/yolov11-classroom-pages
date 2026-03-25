# scripts/09_run_pipeline.py
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from paths import PROJECT_ROOT
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_step(py_exe: str, script: Path, args: List[str]):
    cmd = [py_exe, str(script)] + args
    print("\n" + "=" * 80)
    print(f"[RUN] {script.name} {' '.join(args)}")
    print("=" * 80)
    r = subprocess.run(cmd, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"Step failed: {script.name} (exit={r.returncode})")


def file_is_fresh(output_path: Path, inputs: List[Path], min_bytes: int = 100) -> bool:
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


def maybe_run(step_id, step_name, outputs, inputs, force, from_step, run_fn, min_bytes=100):
    if step_id < from_step:
        print(f"[SKIP] step{step_id:02d} {step_name}")
        return
    if force:
        print(f"[FORCE] step{step_id:02d} {step_name}")
        run_fn()
        return

    all_fresh = True
    for o in outputs:
        if not file_is_fresh(o, inputs, min_bytes):
            all_fresh = False
            break

    if all_fresh:
        print(f"[CACHE] step{step_id:02d} {step_name} -> outputs fresh")
    else:
        print(f"[DO] step{step_id:02d} {step_name}")
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
    # Allow Ultralytics model aliases, e.g. yolo11n-pose.pt.
    return model_arg


def resolve_video_path(video_arg: str) -> Path:
    candidate = Path(video_arg)
    if candidate.is_absolute():
        return candidate.resolve()
    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (PROJECT_ROOT / candidate).resolve()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _get_video_info(video_path: Path, fps_fallback: float = 25.0):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return fps_fallback, 0, 0, 0
    fps = cap.get(cv2.CAP_PROP_FPS) or fps_fallback
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return float(fps), int(n), int(w), int(h)


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


def _export_compat_bundle(
    *,
    out_dir: Path,
    video_path: Path,
    case_id: str,
    video_id: str,
    view_name: str,
    pose_tracks: Path,
    actions_jsonl: Path,
    overlay_video: Path,
    timeline_png: Path,
    fps_hint: float,
) -> None:
    fps, n_frames, w, h = _get_video_info(video_path, fps_fallback=fps_hint)
    legacy_jsonl = out_dir / f"{case_id}.jsonl"
    legacy_meta = out_dir / f"{case_id}.meta.json"
    legacy_summary = out_dir / f"{case_id}_summary.json"

    if pose_tracks.exists():
        shutil.copy2(pose_tracks, legacy_jsonl)
        _write_json(
            legacy_meta,
            {
                "video_id": video_id,
                "case_id": case_id,
                "view": view_name,
                "video_path": str(video_path),
                "fps": fps,
                "frames": n_frames,
                "width": w,
                "height": h,
                "source_tracks": str(pose_tracks),
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )

    if actions_jsonl.exists():
        beh_jsonl = out_dir / f"{case_id}_behavior.jsonl"
        beh_meta = out_dir / f"{case_id}_behavior.meta.json"
        shutil.copy2(actions_jsonl, beh_jsonl)
        _write_json(
            beh_meta,
            {
                "video_id": video_id,
                "case_id": case_id,
                "view": view_name,
                "video_path": str(video_path),
                "fps": fps,
                "frames": n_frames,
                "labels": [],
                "source_actions": str(actions_jsonl),
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )

    if overlay_video.exists():
        compat_overlay = out_dir / f"{case_id}_overlay.mp4"
        if compat_overlay != overlay_video:
            shutil.copy2(overlay_video, compat_overlay)

    timeline_json = timeline_png.with_suffix(".json")
    if timeline_json.exists():
        timeline_alias = out_dir / "timeline_viz.json"
        if timeline_alias != timeline_json:
            shutil.copy2(timeline_json, timeline_alias)

    if not legacy_summary.exists():
        _write_json(legacy_summary, {"info": "See downstream summary outputs for details."})


def _auto_pose_model_path(user_arg: str) -> str:
    if user_arg and user_arg.lower() != "auto":
        return resolve_model_or_fail(user_arg)

    enhanced = (PROJECT_ROOT / "models" / "classroom_yolo_enhanced.pt").resolve()
    if enhanced.exists():
        print(f"[INFO] Found enhanced classroom model, using: {enhanced}")
        return str(enhanced)

    print(
        "[INFO] Enhanced pose model not found. Falling back to yolo11n-pose.pt. "
        "You can train one via scripts/training/train_classroom_yolo.py"
    )
    return resolve_model_or_fail("yolo11n-pose.pt")


def main():
    base_dir = PROJECT_ROOT
    scripts_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Deep Learning Classroom Behavior Pipeline")
    parser.add_argument("--video", type=str, default="data/videos/demo3.mp4")
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--output_root", type=str, default="")
    parser.add_argument("--view", type=str, default="")
    parser.add_argument("--view_code", type=str, default="")
    parser.add_argument("--case_id", type=str, default="")
    parser.add_argument("--video_id", type=str, default="")
    parser.add_argument("--export_compat", type=int, default=1)
    parser.add_argument("--py", type=str, default=sys.executable)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--from_step", type=int, default=1)
    parser.add_argument("--skip_vis", action="store_true")
    parser.add_argument("--skip_video", action="store_true")

    # Models
    parser.add_argument("--pose_model", type=str, default="auto", help="auto | model path")
    parser.add_argument("--det_model", type=str, default="yolo11n.pt")
    parser.add_argument("--action_model", type=str, default="")
    parser.add_argument("--stgcn_weight", type=str, default="")

    # Behavior fusion
    parser.add_argument("--fuse_action_obj", type=int, default=1)
    parser.add_argument("--fuse_window", type=float, default=0.8)
    parser.add_argument("--fuse_alpha", type=float, default=0.75)
    parser.add_argument("--fuse_beta", type=float, default=0.25)

    # Cross-upgrade switches
    parser.add_argument("--enable_peer_aware", type=int, default=1)
    parser.add_argument("--peer_radius", type=float, default=0.15)
    parser.add_argument("--interaction_model", choices=["igformer", "legacy"], default="igformer")
    parser.add_argument("--dsig_k", type=int, default=3)
    parser.add_argument("--sdig_threshold", type=float, default=0.35)

    parser.add_argument("--enable_mllm", type=int, default=0)
    parser.add_argument("--mllm_model", type=str, default="heuristic")
    parser.add_argument("--mllm_quantize", type=str, default="none")
    parser.add_argument("--mllm_keyframe_interval", type=int, default=30)
    parser.add_argument("--enable_cca", type=int, default=1)
    parser.add_argument("--enable_dqd", type=int, default=1)

    # Export options
    parser.add_argument("--interpolate_occluded", type=int, default=0)
    parser.add_argument("--occlusion_conf_thres", type=float, default=0.2)

    # Misc
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--viz_gap_tol", type=float, default=1.5)
    parser.add_argument("--viz_min_dur", type=float, default=0.1)

    args = parser.parse_args()

    video_path = resolve_video_path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    name = args.name if args.name else (args.case_id or video_path.stem)
    out_dir = _resolve_out_dir(args, video_path, base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Outputs
    pose_keypoints = out_dir / "pose_keypoints_v2.jsonl"
    objects_jsonl = out_dir / "objects.jsonl"
    pose_tracks = out_dir / "pose_tracks_smooth.jsonl"
    pose_demo_video = out_dir / "pose_demo_out.mp4"
    objects_demo_video = out_dir / "objects_demo_out.mp4"

    actions_jsonl = out_dir / "actions.jsonl"
    actions_fused_jsonl = out_dir / "actions_fused.jsonl"
    embeddings_pkl = out_dir / "embeddings.pkl"
    keyframe_dir = out_dir / "keyframes"

    transcript_jsonl = out_dir / "transcript.jsonl"
    per_person_json = out_dir / "per_person_sequences.json"
    mllm_verified_json = out_dir / "mllm_verified_sequences.json"
    group_events_jsonl = out_dir / "group_events.jsonl"

    student_features_json = out_dir / "student_features.json"
    student_projection_json = out_dir / "student_projection.json"

    overlay_video = out_dir / f"{name}_overlay.mp4"
    timeline_png = out_dir / "timeline_chart.png"

    py = args.py
    pose_model_abs = _auto_pose_model_path(args.pose_model)
    det_model_abs = resolve_model_or_fail(args.det_model)

    video_fps, _, video_w, video_h = _get_video_info(video_path, fps_fallback=args.fps)
    fps_str = str(video_fps if video_fps > 0 else args.fps)

    # Step 02
    maybe_run(
        2,
        "Pose Keypoints",
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

    # Step 21
    maybe_run(
        21,
        "Pose Demo Video",
        [pose_demo_video],
        [video_path, pose_keypoints],
        args.force,
        args.from_step,
        lambda: run_step(
            py,
            scripts_dir / "01_pose_video_demo.py",
            ["--video", str(video_path), "--out", str(pose_demo_video), "--model", pose_model_abs],
        ),
    )

    # Step 03
    maybe_run(
        3,
        "Objects Detection",
        [objects_jsonl],
        [video_path],
        args.force,
        args.from_step,
        lambda: run_step(
            py,
            scripts_dir / "02b_export_objects_jsonl.py",
            ["--video", str(video_path), "--out", str(objects_jsonl), "--model", det_model_abs],
        ),
    )

    # Step 09 (object demo, optional side output)
    maybe_run(
        9,
        "Objects Demo Video",
        [objects_demo_video],
        [video_path, objects_jsonl],
        args.force,
        args.from_step,
        lambda: run_step(
            py,
            scripts_dir / "03b_objects_video_demo.py",
            [
                "--video",
                str(video_path),
                "--objects",
                str(objects_jsonl),
                "--out",
                str(objects_demo_video),
                "--mux_audio",
                "1",
                "--conf",
                "0.2",
                "--show_empty",
                "1",
            ],
        ),
    )

    # Step 04
    maybe_run(
        4,
        "Tracking",
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

    # Step 05
    def _run05():
        cmd_args = [
            "--video",
            str(video_path),
            "--pose",
            str(pose_tracks),
            "--out",
            str(actions_jsonl),
            "--emb_out",
            str(embeddings_pkl),
            "--save_keyframes",
            str(int(args.enable_mllm)),
            "--keyframe_dir",
            str(keyframe_dir),
        ]
        if args.action_model:
            cmd_args += ["--model_weight", resolve_model_or_fail(args.action_model)]
        run_step(py, scripts_dir / "05_slowfast_actions.py", cmd_args)

    maybe_run(
        5,
        "SlowFast Behavior Recognition",
        [actions_jsonl, embeddings_pkl],
        [video_path, pose_tracks],
        args.force,
        args.from_step,
        _run05,
    )

    # Step 55
    if int(args.fuse_action_obj) == 1:
        maybe_run(
            55,
            "Action/Object Fusion",
            [actions_fused_jsonl],
            [actions_jsonl, objects_jsonl],
            args.force,
            args.from_step,
            lambda: run_step(
                py,
                scripts_dir / "05b_fuse_actions_with_objects.py",
                [
                    "--actions",
                    str(actions_jsonl),
                    "--objects",
                    str(objects_jsonl),
                    "--out",
                    str(actions_fused_jsonl),
                    "--window",
                    str(args.fuse_window),
                    "--alpha",
                    str(args.fuse_alpha),
                    "--beta",
                    str(args.fuse_beta),
                ],
            ),
        )

    actions_for_downstream = (
        actions_fused_jsonl if int(args.fuse_action_obj) == 1 and actions_fused_jsonl.exists() else actions_jsonl
    )

    # Step 06
    maybe_run(
        6,
        "ASR",
        [transcript_jsonl],
        [video_path],
        args.force,
        args.from_step,
        lambda: run_step(py, scripts_dir / "06_api_asr_realtime.py", ["--video", str(video_path), "--out_dir", str(out_dir)]),
    )

    # Step 07 (+Peer)
    maybe_run(
        7,
        "Dual Verification + Peer Aware",
        [per_person_json],
        [actions_for_downstream, transcript_jsonl, pose_tracks],
        args.force,
        args.from_step,
        lambda: run_step(
            py,
            scripts_dir / "07_dual_verification.py",
            [
                "--actions",
                str(actions_for_downstream),
                "--transcript",
                str(transcript_jsonl),
                "--out",
                str(per_person_json),
                "--fps",
                fps_str,
                "--pose_tracks",
                str(pose_tracks),
                "--enable_peer_aware",
                str(int(args.enable_peer_aware)),
                "--peer_radius",
                str(float(args.peer_radius)),
            ],
        ),
    )

    # Step 14 (optional MLLM)
    if int(args.enable_mllm) == 1:
        maybe_run(
            14,
            "MLLM Semantic Verify",
            [mllm_verified_json],
            [per_person_json, embeddings_pkl, transcript_jsonl, video_path],
            args.force,
            args.from_step,
            lambda: run_step(
                py,
                scripts_dir / "14_mllm_semantic_verify.py",
                [
                    "--per_person",
                    str(per_person_json),
                    "--emb",
                    str(embeddings_pkl),
                    "--transcript",
                    str(transcript_jsonl),
                    "--video",
                    str(video_path),
                    "--out",
                    str(mllm_verified_json),
                    "--mllm_model",
                    str(args.mllm_model),
                    "--quantize",
                    str(args.mllm_quantize),
                    "--keyframe_interval",
                    str(int(args.mllm_keyframe_interval)),
                    "--enable_cca",
                    str(int(args.enable_cca)),
                    "--enable_dqd",
                    str(int(args.enable_dqd)),
                ],
            ),
        )

    # Step 11 (+IGFormer)
    def _run11():
        cmd_args = [
            "--pose",
            str(pose_tracks),
            "--actions",
            str(actions_for_downstream),
            "--out",
            str(group_events_jsonl),
            "--fps",
            fps_str,
            "--width",
            str(int(video_w) if video_w > 0 else 1920),
            "--height",
            str(int(video_h) if video_h > 0 else 1080),
            "--interaction_model",
            args.interaction_model,
            "--dsig_k",
            str(int(args.dsig_k)),
            "--sdig_threshold",
            str(float(args.sdig_threshold)),
        ]
        if args.stgcn_weight:
            cmd_args += ["--model_weight", resolve_model_or_fail(args.stgcn_weight)]
        run_step(py, scripts_dir / "11_group_stgcn.py", cmd_args)

    maybe_run(
        11,
        "Group Interaction Analysis",
        [group_events_jsonl],
        [pose_tracks, actions_for_downstream],
        args.force,
        args.from_step,
        _run11,
    )

    features_src = mllm_verified_json if int(args.enable_mllm) == 1 and mllm_verified_json.exists() else per_person_json

    # Step 12
    maybe_run(
        12,
        "Feature Extraction",
        [student_features_json],
        [features_src],
        args.force,
        args.from_step,
        lambda: run_step(py, scripts_dir / "12_export_features.py", ["--src", str(features_src), "--out", str(student_features_json)]),
    )

    # Step 13
    maybe_run(
        13,
        "Semantic Projection",
        [student_projection_json],
        [embeddings_pkl, student_features_json],
        args.force,
        args.from_step,
        lambda: run_step(
            py,
            scripts_dir / "13_semantic_projection.py",
            [
                "--emb",
                str(embeddings_pkl),
                "--meta",
                str(student_features_json),
                "--out",
                str(student_projection_json),
                "--n_clusters",
                "3",
            ],
        ),
    )

    # Step 08
    if not args.skip_video:
        overlay_inputs = [video_path, actions_for_downstream, transcript_jsonl, group_events_jsonl]
        if int(args.enable_mllm) == 1 and mllm_verified_json.exists():
            overlay_inputs.append(mllm_verified_json)
        maybe_run(
            108,
            "Overlay Video",
            [overlay_video],
            overlay_inputs,
            args.force,
            args.from_step,
            lambda: run_step(
                py,
                scripts_dir / "08_overlay_sequences.py",
                [
                    "--video",
                    str(video_path),
                    "--actions",
                    str(actions_for_downstream),
                    "--transcript",
                    str(transcript_jsonl),
                    "--pose_tracks",
                    str(pose_tracks),
                    "--group_src",
                    str(group_events_jsonl),
                    "--mllm_src",
                    str(mllm_verified_json),
                    "--out_dir",
                    str(out_dir),
                    "--name",
                    name,
                    "--mux_audio",
                    "1",
                    "--out_video",
                    str(overlay_video),
                ],
            ),
        )

    # Step 10
    if not args.skip_vis:
        timeline_inputs = [per_person_json, group_events_jsonl]
        if int(args.enable_mllm) == 1 and mllm_verified_json.exists():
            timeline_inputs.append(mllm_verified_json)
        maybe_run(
            110,
            "Timeline Visualization",
            [timeline_png],
            timeline_inputs,
            args.force,
            args.from_step,
            lambda: run_step(
                py,
                scripts_dir / "10_visualize_timeline.py",
                [
                    "--src",
                    str(per_person_json),
                    "--group_src",
                    str(group_events_jsonl),
                    "--mllm_src",
                    str(mllm_verified_json),
                    "--out",
                    str(timeline_png),
                    "--gap_tol",
                    str(args.viz_gap_tol),
                    "--min_dur",
                    str(args.viz_min_dur),
                    "--fps",
                    fps_str,
                ],
            ),
        )

    if int(args.export_compat) == 1:
        case_id = args.case_id or name
        view_name = args.view or ""
        if args.video_id:
            video_id = args.video_id
        else:
            view_code = args.view_code or args.view or "case"
            video_id = f"{view_code}__{case_id}"
        _export_compat_bundle(
            out_dir=out_dir,
            video_path=video_path,
            case_id=case_id,
            video_id=video_id,
            view_name=view_name,
            pose_tracks=pose_tracks,
            actions_jsonl=actions_for_downstream,
            overlay_video=overlay_video,
            timeline_png=timeline_png,
            fps_hint=args.fps,
        )

    print("\n[DONE] Pipeline completed")
    print(f"Output dir            : {out_dir}")
    print(f"Pose tracks           : {pose_tracks}")
    print(f"Actions downstream    : {actions_for_downstream}")
    print(f"Group events          : {group_events_jsonl}")
    if int(args.enable_mllm) == 1:
        print(f"MLLM verified         : {mllm_verified_json}")
    print(f"Features              : {student_features_json}")
    print(f"Projection            : {student_projection_json}")


if __name__ == "__main__":
    main()
