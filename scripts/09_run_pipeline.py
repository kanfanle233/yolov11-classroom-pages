import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

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
        print(f"[CACHE] step{step_id:02d} {step_name} -> Outputs fresh.")
    else:
        print(f"[DO] step{step_id:02d} {step_name}")
        run_fn()


def resolve_model_or_fail(model_arg: str) -> str:
    p = Path(model_arg)
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Model not found: {p}")
    return str(p)


def main():
    base_dir = PROJECT_ROOT
    scripts_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Deep Learning Classroom Behavior Pipeline")
    parser.add_argument("--video", type=str, default="data/videos/demo3.mp4")
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--py", type=str, default=sys.executable)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--from_step", type=int, default=1)
    parser.add_argument("--skip_vis", action="store_true")
    parser.add_argument("--skip_video", action="store_true")

    # Model weights
    parser.add_argument(
        "--pose_model",
        type=str,
        default="yolo11s-pose.pt",
        help="pose model path; run scripts/training/train_classroom_yolo.py to produce classroom_yolo_enhanced.pt",
    )
    parser.add_argument("--det_model", type=str, default="yolo11s.pt")
    parser.add_argument("--action_model", type=str, default="", help="Optional: SlowFast weights")
    parser.add_argument("--stgcn_weight", type=str, default="", help="Optional: ST-GCN/IGFormer weights")

    # Global params
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--viz_gap_tol", type=float, default=1.5)
    parser.add_argument("--viz_min_dur", type=float, default=0.1)

    # Upgrade switches
    parser.add_argument("--interpolate_occluded", action="store_true")
    parser.add_argument("--enable_peer_aware", type=int, default=0)
    parser.add_argument("--peer_radius", type=float, default=0.15)
    parser.add_argument("--interaction_model", type=str, default="legacy", choices=["legacy", "igformer"])
    parser.add_argument("--dsig_k", type=int, default=3)
    parser.add_argument("--sdig_threshold", type=float, default=0.7)
    parser.add_argument("--enable_mllm", type=int, default=0)
    parser.add_argument("--mllm_model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--mllm_quantize", type=str, default="")
    parser.add_argument("--save_keyframes", type=int, default=0, help="force save keyframes in step05")

    args = parser.parse_args()

    enhanced_pose_model = (base_dir / "models" / "classroom_yolo_enhanced.pt").resolve()
    if args.pose_model == "yolo11s-pose.pt" and enhanced_pose_model.exists():
        args.pose_model = str(enhanced_pose_model)
        print(f"[INFO] Auto-switch pose model -> {args.pose_model}")

    video_path = (base_dir / args.video).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    name = args.name if args.name else video_path.stem
    out_dir = (base_dir / "output" / name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Output paths
    pose_keypoints = out_dir / "pose_keypoints_v2.jsonl"
    pose_demo_video = out_dir / "pose_demo_out.mp4"
    objects_jsonl = out_dir / "objects.jsonl"
    pose_tracks = out_dir / "pose_tracks_smooth.jsonl"
    actions_jsonl = out_dir / "actions.jsonl"
    embeddings_pkl = out_dir / "embeddings.pkl"
    keyframes_dir = out_dir / "keyframes"
    transcript_jsonl = out_dir / "transcript.jsonl"
    per_person_json = out_dir / "per_person_sequences.json"
    mllm_verified_json = out_dir / "mllm_verified_sequences.json"
    overlay_video = out_dir / f"{name}_overlay.mp4"
    objects_demo_video = out_dir / "objects_demo_out.mp4"
    group_events_jsonl = out_dir / "group_events.jsonl"
    student_features_json = out_dir / "student_features.json"
    student_projection_json = out_dir / "student_projection.json"
    timeline_png = out_dir / "timeline_chart.png"

    py = args.py
    pose_model_abs = resolve_model_or_fail(args.pose_model)
    det_model_abs = resolve_model_or_fail(args.det_model)
    fps_str = str(args.fps)

    # Step 02: Pose keypoints
    def _run02():
        cmd = ["--video", str(video_path), "--out", str(pose_keypoints), "--model", pose_model_abs]
        if args.interpolate_occluded:
            cmd.append("--interpolate_occluded")
        run_step(py, scripts_dir / "02_export_keypoints_jsonl.py", cmd)

    maybe_run(2, "Pose Keypoints", [pose_keypoints], [video_path], args.force, args.from_step, _run02)

    # Step 21: Pose demo
    maybe_run(
        21,
        "Pose Demo Video",
        [pose_demo_video],
        [video_path],
        args.force,
        args.from_step,
        lambda: run_step(
            py,
            scripts_dir / "01_pose_video_demo.py",
            ["--video", str(video_path), "--out", str(pose_demo_video), "--model", pose_model_abs],
        ),
    )

    # Step 03: Objects detection
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

    # Step 04: Tracking
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

    # Step 05: SlowFast actions
    def _run05():
        cmd = [
            "--video",
            str(video_path),
            "--pose",
            str(pose_tracks),
            "--out",
            str(actions_jsonl),
            "--emb_out",
            str(embeddings_pkl),
        ]
        if args.action_model:
            cmd += ["--model_weight", resolve_model_or_fail(args.action_model)]
        if int(args.save_keyframes) == 1 or int(args.enable_mllm) == 1:
            cmd += ["--save_keyframes", "1", "--keyframe_dir", str(keyframes_dir)]
        run_step(py, scripts_dir / "05_slowfast_actions.py", cmd)

    maybe_run(
        5,
        "SlowFast Behavior Recognition",
        [actions_jsonl, embeddings_pkl],
        [video_path, pose_tracks],
        args.force,
        args.from_step,
        _run05,
    )

    # Step 06: ASR
    maybe_run(
        6,
        "ASR",
        [transcript_jsonl],
        [video_path],
        args.force,
        args.from_step,
        lambda: run_step(py, scripts_dir / "06_api_asr_realtime.py", ["--video", str(video_path), "--out_dir", str(out_dir)]),
    )

    # Step 07: Merge + peer-aware
    def _run07():
        cmd = [
            "--actions",
            str(actions_jsonl),
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
            str(args.peer_radius),
        ]
        run_step(py, scripts_dir / "07_dual_verification.py", cmd)

    maybe_run(
        7,
        "Dual Verification (+Peer Aware)",
        [per_person_json],
        [actions_jsonl, transcript_jsonl, pose_tracks],
        args.force,
        args.from_step,
        _run07,
    )

    # Step 14: MLLM semantic verify (optional)
    if int(args.enable_mllm) == 1:
        def _run14():
            run_step(
                py,
                scripts_dir / "14_mllm_semantic_verify.py",
                [
                    "--per_person",
                    str(per_person_json),
                    "--emb",
                    str(embeddings_pkl),
                    "--transcript",
                    str(transcript_jsonl),
                    "--keyframes",
                    str(keyframes_dir),
                    "--out",
                    str(mllm_verified_json),
                    "--mllm_model",
                    str(args.mllm_model),
                    "--quantize",
                    str(args.mllm_quantize),
                    "--fps",
                    fps_str,
                ],
            )

        maybe_run(
            14,
            "MLLM Semantic Verify",
            [mllm_verified_json],
            [per_person_json, embeddings_pkl, transcript_jsonl],
            args.force,
            args.from_step,
            _run14,
        )

    sequence_src = mllm_verified_json if (int(args.enable_mllm) == 1 and mllm_verified_json.exists()) else per_person_json

    # Step 11: Group interaction
    def _run11():
        cmd = [
            "--pose",
            str(pose_tracks),
            "--out",
            str(group_events_jsonl),
            "--actions",
            str(actions_jsonl),
            "--interaction_model",
            str(args.interaction_model),
            "--dsig_k",
            str(args.dsig_k),
            "--sdig_threshold",
            str(args.sdig_threshold),
            "--fps",
            fps_str,
        ]
        if args.stgcn_weight:
            cmd += ["--model_weight", resolve_model_or_fail(args.stgcn_weight)]
        run_step(py, scripts_dir / "11_group_stgcn.py", cmd)

    maybe_run(
        11,
        "Group Interactions (IGFormer/Legacy)",
        [group_events_jsonl],
        [pose_tracks, actions_jsonl],
        args.force,
        args.from_step,
        _run11,
    )

    # Step 12: features
    maybe_run(
        12,
        "Feature Extraction",
        [student_features_json],
        [sequence_src],
        args.force,
        args.from_step,
        lambda: run_step(py, scripts_dir / "12_export_features.py", ["--src", str(sequence_src), "--out", str(student_features_json), "--fps", fps_str]),
    )

    # Step 13: projection
    maybe_run(
        13,
        "Semantic Projection (UMAP)",
        [student_projection_json],
        [embeddings_pkl, student_features_json],
        args.force,
        args.from_step,
        lambda: run_step(
            py,
            scripts_dir / "13_semantic_projection.py",
            ["--emb", str(embeddings_pkl), "--meta", str(student_features_json), "--out", str(student_projection_json), "--n_clusters", "3"],
        ),
    )

    # Step 08: overlay video
    if not args.skip_video:
        def _run08():
            cmd = [
                "--video",
                str(video_path),
                "--actions",
                str(actions_jsonl),
                "--transcript",
                str(transcript_jsonl),
                "--out_dir",
                str(out_dir),
                "--name",
                name,
                "--mux_audio",
                "1",
                "--out_video",
                str(overlay_video),
                "--group_events",
                str(group_events_jsonl),
                "--pose_tracks",
                str(pose_tracks),
            ]
            if sequence_src.exists():
                cmd += ["--mllm_seq", str(sequence_src)]
            run_step(py, scripts_dir / "08_overlay_sequences.py", cmd)

        maybe_run(
            8,
            "Overlay Video",
            [overlay_video],
            [video_path, actions_jsonl, transcript_jsonl, group_events_jsonl],
            args.force,
            args.from_step,
            _run08,
        )

    # Step 09: objects demo
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

    # Step 10: timeline
    if not args.skip_vis:
        maybe_run(
            10,
            "Generate Timeline Chart",
            [timeline_png],
            [sequence_src, group_events_jsonl],
            args.force,
            args.from_step,
            lambda: run_step(
                py,
                scripts_dir / "10_visualize_timeline.py",
                [
                    "--src",
                    str(sequence_src),
                    "--out",
                    str(timeline_png),
                    "--group_src",
                    str(group_events_jsonl),
                    "--mllm_src",
                    str(sequence_src),
                    "--gap_tol",
                    str(args.viz_gap_tol),
                    "--min_dur",
                    str(args.viz_min_dur),
                    "--fps",
                    fps_str,
                ],
            ),
        )

    print(f"\n[DONE] All steps completed. Results in: {out_dir}")
    print(f"[OUT] Pose Demo Video     : {pose_demo_video}")
    print(f"[OUT] Group Events        : {group_events_jsonl}")
    print(f"[OUT] Features            : {student_features_json}")
    print(f"[OUT] Semantic Projection : {student_projection_json}")
    if int(args.enable_mllm) == 1:
        print(f"[OUT] MLLM Verified Seq   : {mllm_verified_json}")


if __name__ == "__main__":
    main()
