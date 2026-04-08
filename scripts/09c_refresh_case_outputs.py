import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contracts.schemas import SCHEMA_VERSION
from verifier.dataset import build_training_samples, convert_to_contract_samples, save_training_samples


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FORMAL_KEEP = {
    "actions.jsonl",
    "align_multimodal.json",
    "event_queries.jsonl",
    "per_person_sequences.json",
    "pipeline_manifest.json",
    "pose_keypoints_v2.jsonl",
    "pose_tracks_smooth.jsonl",
    "pose_tracks_smooth_uq.jsonl",
    "timeline_chart.json",
    "timeline_chart.png",
    "transcript.jsonl",
    "verified_events.jsonl",
    "verifier.pt",
    "verifier_samples_train.jsonl",
    "verifier_eval_report.json",
    "verifier_calibration_report.json",
    "verifier_reliability_diagram.svg",
}

LEGACY_NAMES = {
    "align.json",
    "aligned_event_queries.jsonl",
    "contracts_report.json",
    "demo1_overlay.mp4",
    "embeddings.pkl",
    "group_events.jsonl",
    "objects.jsonl",
    "objects_demo_out.mp4",
    "objects_demo_out_noaudio.mp4",
    "pose_demo_out.mp4",
    "pose_tracks_smooth_kpts.jsonl",
    "static_projection.json",
    "student_features.json",
    "student_projection.json",
    "timeline_viz.json",
    "verifier_report.json",
    "verifier_report.raw.json",
    "verifier_samples.jsonl",
    "verifier_samples.raw.jsonl",
}

LEGACY_SUFFIXES = (
    "_behavior.jsonl",
    "_behavior.meta.json",
    "_behavior_overlay.mp4",
    "_behavior_overlay_mp4v.mp4",
    "_summary.json",
    ".meta.json",
    ".wav",
)


def run_step(py_exe: str, script: Path, args: List[str]) -> None:
    cmd = [py_exe, str(script)] + args
    print(f"[RUN] {script.name} {' '.join(args)}")
    completed = subprocess.run(cmd, check=False, cwd=str(PROJECT_ROOT))
    if completed.returncode != 0:
        raise RuntimeError(f"step failed: {script.name} exit={completed.returncode}")


def iter_case_dirs(case_roots: Iterable[str]) -> List[Path]:
    case_dirs: List[Path] = []
    for raw in case_roots:
        path = Path(raw)
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        if path.is_dir():
            case_dirs.append(path)
    return case_dirs


def should_remove(path: Path) -> bool:
    if path.name in LEGACY_NAMES:
        return True
    if any(path.name.endswith(suffix) for suffix in LEGACY_SUFFIXES):
        return True
    if path.suffix == ".jsonl" and path.name not in FORMAL_KEEP:
        stem = path.stem
        if stem.isdigit() or stem.lower().startswith("demo"):
            return True
    return False


def cleanup_legacy_outputs(case_dir: Path) -> List[Path]:
    removed: List[Path] = []
    for path in sorted(case_dir.iterdir()):
        if not path.is_file():
            continue
        if should_remove(path):
            path.unlink(missing_ok=True)
            removed.append(path)
    return removed


def write_manifest(case_dir: Path, py_exe: str, cleanup_legacy: bool, rerun_uq: bool) -> None:
    artifact_map = {
        "actions": "actions.jsonl",
        "align_multimodal": "align_multimodal.json",
        "event_queries": "event_queries.jsonl",
        "per_person_sequences": "per_person_sequences.json",
        "pipeline_manifest": "pipeline_manifest.json",
        "pose_keypoints": "pose_keypoints_v2.jsonl",
        "pose_tracks_smooth": "pose_tracks_smooth.jsonl",
        "pose_tracks_smooth_uq": "pose_tracks_smooth_uq.jsonl",
        "timeline_chart_json": "timeline_chart.json",
        "timeline_chart_png": "timeline_chart.png",
        "transcript": "transcript.jsonl",
        "verified_events": "verified_events.jsonl",
        "verifier_model": "verifier.pt",
        "verifier_samples_train": "verifier_samples_train.jsonl",
        "verifier_eval_report": "verifier_eval_report.json",
        "verifier_calibration_report": "verifier_calibration_report.json",
        "verifier_reliability_diagram": "verifier_reliability_diagram.svg",
    }
    artifacts: Dict[str, str] = {}
    for key, name in artifact_map.items():
        path = case_dir / name
        if path.exists():
            artifacts[key] = str(path)
    payload = {
        "case_id": case_dir.name,
        "video_id": case_dir.name,
        "schema_version": SCHEMA_VERSION,
        "artifacts": artifacts,
        "config_snapshot": {
            "refresh_existing_case": True,
            "cleanup_legacy": bool(cleanup_legacy),
            "rerun_uq": bool(rerun_uq),
            "python_interpreter": py_exe,
            "generated_by": "scripts/09c_refresh_case_outputs.py",
        },
    }
    (case_dir / "pipeline_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def refresh_case(case_dir: Path, py_exe: str, cleanup_legacy: bool, rerun_uq: bool) -> None:
    transcript = case_dir / "transcript.jsonl"
    actions = case_dir / "actions.jsonl"
    pose_tracks = case_dir / "pose_tracks_smooth.jsonl"
    pose_uq = case_dir / "pose_tracks_smooth_uq.jsonl"
    event_queries = case_dir / "event_queries.jsonl"
    aligned = case_dir / "align_multimodal.json"
    verified = case_dir / "verified_events.jsonl"
    verifier_samples_train = case_dir / "verifier_samples_train.jsonl"
    per_person = case_dir / "per_person_sequences.json"
    verifier_model = case_dir / "verifier.pt"
    eval_report = case_dir / "verifier_eval_report.json"
    calibration_report = case_dir / "verifier_calibration_report.json"
    reliability_diagram = case_dir / "verifier_reliability_diagram.svg"
    timeline_png = case_dir / "timeline_chart.png"

    if cleanup_legacy:
        removed = cleanup_legacy_outputs(case_dir)
        print(f"[CLEAN] {case_dir} removed={len(removed)}")

    if rerun_uq and pose_tracks.exists():
        run_step(
            py_exe,
            PROJECT_ROOT / "scripts" / "03c_estimate_track_uncertainty.py",
            ["--in", str(pose_tracks), "--out", str(pose_uq), "--validate", "1"],
        )

    run_step(
        py_exe,
        PROJECT_ROOT / "scripts" / "06b_event_query_extraction.py",
        ["--transcript", str(transcript), "--out", str(event_queries), "--validate", "1"],
    )
    run_step(
        py_exe,
        PROJECT_ROOT / "scripts" / "xx_align_multimodal.py",
        ["--event_queries", str(event_queries), "--actions", str(actions), "--pose_uq", str(pose_uq), "--out", str(aligned)],
    )
    save_training_samples(
        verifier_samples_train,
        convert_to_contract_samples(
            build_training_samples(
                event_queries_path=event_queries,
                aligned_path=aligned,
                actions_path=actions,
            )
        ),
    )

    step07_args = [
        "--actions",
        str(actions),
        "--event_queries",
        str(event_queries),
        "--pose_uq",
        str(pose_uq),
        "--aligned",
        str(aligned),
        "--out",
        str(verified),
        "--per_person_out",
        str(per_person),
        "--validate",
        "1",
    ]
    if verifier_model.exists():
        step07_args.extend(["--verifier_model", str(verifier_model)])
    run_step(py_exe, PROJECT_ROOT / "scripts" / "07_dual_verification.py", step07_args)

    run_step(
        py_exe,
        PROJECT_ROOT / "verifier" / "eval.py",
        ["--verified", str(verified), "--out", str(eval_report), "--split", "val", "--target_field", "auto"],
    )
    run_step(
        py_exe,
        PROJECT_ROOT / "verifier" / "calibration.py",
        [
            "--verified",
            str(verified),
            "--out",
            str(calibration_report),
            "--split",
            "val",
            "--target_field",
            "auto",
            "--prob_field",
            "p_match",
            "--num_bins",
            "10",
            "--diagram_out",
            str(reliability_diagram),
        ],
    )
    run_step(
        py_exe,
        PROJECT_ROOT / "scripts" / "10_visualize_timeline.py",
        ["--src", str(per_person), "--verified_src", str(verified), "--out", str(timeline_png)],
    )
    write_manifest(case_dir, py_exe, cleanup_legacy=cleanup_legacy, rerun_uq=rerun_uq)
    print(f"[DONE] refreshed {case_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh formal contract outputs from existing case directories.")
    parser.add_argument("--case_dir", action="append", required=True, help="case output directory; may be provided multiple times")
    parser.add_argument("--py", default=sys.executable, type=str)
    parser.add_argument("--cleanup_legacy", default=1, type=int)
    parser.add_argument("--rerun_uq", default=1, type=int)
    args = parser.parse_args()

    case_dirs = iter_case_dirs(args.case_dir)
    if not case_dirs:
        raise SystemExit("no valid case directories provided")

    for case_dir in case_dirs:
        refresh_case(
            case_dir=case_dir,
            py_exe=str(args.py),
            cleanup_legacy=bool(int(args.cleanup_legacy)),
            rerun_uq=bool(int(args.rerun_uq)),
        )


if __name__ == "__main__":
    main()
