from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from fusion_utils import read_jsonl, resolve_path, resolve_repo_root, semantic_coverage, write_json


def _file_state(path: Path) -> Dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
    }


def _load_json(path: Path) -> Any:
    if not path.exists() or path.stat().st_size <= 0:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _count_people_from_pose(rows: List[Dict[str, Any]]) -> int:
    track_ids: set[int] = set()
    for row in rows:
        people = row.get("people", row.get("persons", []))
        if isinstance(people, dict):
            people = list(people.values())
        if not isinstance(people, list):
            continue
        for person in people:
            if not isinstance(person, dict):
                continue
            tid = person.get("track_id")
            try:
                track_ids.add(int(tid))
            except Exception:
                continue
    return len(track_ids)


def _count_behavior_items(rows: List[Dict[str, Any]]) -> int:
    total = 0
    for row in rows:
        items = row.get("behaviors", row.get("behavior_detections", []))
        if isinstance(items, list):
            total += len([x for x in items if isinstance(x, dict)])
    return total


def _align_stats(path: Path) -> Dict[str, int]:
    obj = _load_json(path)
    events = obj if isinstance(obj, list) else []
    without = 0
    candidates = 0
    for row in events:
        if not isinstance(row, dict):
            continue
        cand = row.get("candidates", [])
        if not isinstance(cand, list) or len(cand) == 0:
            without += 1
        else:
            candidates += len(cand)
    return {"events": len(events), "events_without_candidates": without, "total_candidates": candidates}


def _timeline_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size <= 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _student_count(path: Path) -> int:
    obj = _load_json(path)
    if not isinstance(obj, dict):
        return 0
    students = obj.get("students", [])
    return len(students) if isinstance(students, list) else 0


def _missing_required_csv_fields(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    required = [
        "student_id",
        "track_id",
        "start_time",
        "end_time",
        "behavior_code",
        "semantic_label_zh",
        "semantic_label_en",
        "confidence",
        "source",
    ]
    bad: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        missing = [key for key in required if key not in row or str(row.get(key, "")).strip() == ""]
        if missing:
            bad.append({"index": idx, "missing_fields": missing, "row": row})
    return bad


def _track_id_threshold_stats(rows: List[Dict[str, Any]], *, threshold: int) -> Dict[str, Any]:
    ids: set[int] = set()
    count = 0
    examples: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            tid = int(row.get("track_id"))
        except Exception:
            continue
        if tid < int(threshold):
            continue
        count += 1
        ids.add(tid)
        if len(examples) < 10:
            examples.append(
                {
                    "track_id": tid,
                    "semantic_id": row.get("semantic_id", ""),
                    "source": row.get("source", ""),
                    "start_time": row.get("start_time"),
                    "end_time": row.get("end_time"),
                }
            )
    return {"rows": count, "track_ids": len(ids), "examples": examples}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the paper pipeline contract from pose to timeline.")
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--report", default="", type=str)
    parser.add_argument("--strict", default=1, type=int)
    parser.add_argument("--require_timeline", default=1, type=int)
    parser.add_argument("--track_backend", choices=["pose", "behavior", "hybrid"], default="pose")
    args = parser.parse_args()

    repo_root = resolve_repo_root(Path(__file__))
    output_dir = resolve_path(repo_root, args.output_dir)
    report_path = resolve_path(repo_root, args.report) if args.report else output_dir / "pipeline_contract_v2_report.json"

    paths = {
        "pose_keypoints": output_dir / "pose_keypoints_v2.jsonl",
        "pose_tracks": output_dir / "pose_tracks_smooth.jsonl",
        "student_tracks": output_dir / "student_tracks.jsonl",
        "behavior_det_semantic": output_dir / "behavior_det.semantic.jsonl",
        "actions_fusion_v2": output_dir / "actions.fusion_v2.jsonl",
        "event_queries_fusion_v2": output_dir / "event_queries.fusion_v2.jsonl",
        "align_multimodal": output_dir / "align_multimodal.json",
        "verified_events": output_dir / "verified_events.jsonl",
        "per_person_sequences": output_dir / "per_person_sequences.json",
        "timeline_chart": output_dir / "timeline_chart.png",
        "timeline_json": output_dir / "timeline_chart.json",
        "timeline_students": output_dir / "timeline_students.csv",
        "student_id_map": output_dir / "student_id_map.json",
        "asr_quality_report": output_dir / "asr_quality_report.json",
        "pipeline_manifest": output_dir / "pipeline_manifest.json",
    }
    files = {name: _file_state(path) for name, path in paths.items()}

    errors: List[str] = []
    warnings: List[str] = []
    required = [
        "pose_keypoints",
        "pose_tracks",
        "behavior_det_semantic",
        "actions_fusion_v2",
        "event_queries_fusion_v2",
        "align_multimodal",
        "verified_events",
        "per_person_sequences",
    ]
    if int(args.require_timeline) == 1:
        required.extend(["timeline_chart", "timeline_json", "timeline_students", "student_id_map"])
    if str(args.track_backend).strip().lower() in {"behavior", "hybrid"}:
        required.append("student_tracks")

    for key in required:
        state = files[key]
        if not state["exists"]:
            errors.append(f"missing required artifact: {key} -> {state['path']}")
        elif int(state["bytes"]) <= 0:
            errors.append(f"empty required artifact: {key} -> {state['path']}")

    pose_rows = read_jsonl(paths["pose_keypoints"])
    track_rows = read_jsonl(paths["pose_tracks"])
    student_track_rows = read_jsonl(paths["student_tracks"])
    behavior_rows = read_jsonl(paths["behavior_det_semantic"])
    actions = read_jsonl(paths["actions_fusion_v2"])
    event_queries = read_jsonl(paths["event_queries_fusion_v2"])
    verified = read_jsonl(paths["verified_events"])
    timeline_rows = _timeline_csv_rows(paths["timeline_students"])

    action_total, action_valid, action_invalid = semantic_coverage(actions)
    if len(pose_rows) <= 0:
        errors.append("pose_keypoints_v2.jsonl has zero rows")
    if len(track_rows) <= 0:
        errors.append("pose_tracks_smooth.jsonl has zero rows")
    if _count_people_from_pose(track_rows) <= 0:
        errors.append("pose_tracks_smooth.jsonl has zero tracked students")
    if str(args.track_backend).strip().lower() in {"behavior", "hybrid"} and _count_people_from_pose(student_track_rows) <= 0:
        errors.append("student_tracks.jsonl has zero tracked students")
    if _count_behavior_items(behavior_rows) <= 0:
        errors.append("behavior_det.semantic.jsonl has zero behavior detections")
    if action_total <= 0:
        errors.append("actions.fusion_v2.jsonl has zero rows")
    if action_invalid:
        errors.append(f"actions.fusion_v2.jsonl semantic coverage failed: {len(action_invalid)} rows")
    if len(event_queries) <= 0:
        errors.append("event_queries.fusion_v2.jsonl has zero rows")
    align = _align_stats(paths["align_multimodal"])
    if align["events"] <= 0:
        errors.append("align_multimodal.json has zero events")
    if align["events_without_candidates"] > 0:
        errors.append(f"align_multimodal.json has events without candidates: {align['events_without_candidates']}")
    if len(verified) <= 0:
        errors.append("verified_events.jsonl has zero rows")

    csv_bad = _missing_required_csv_fields(timeline_rows) if int(args.require_timeline) == 1 else []
    if int(args.require_timeline) == 1:
        if len(timeline_rows) <= 0:
            errors.append("timeline_students.csv has zero rows")
        if _student_count(paths["student_id_map"]) <= 0:
            errors.append("student_id_map.json has zero students")
        if csv_bad:
            errors.append(f"timeline_students.csv has rows missing required fields: {len(csv_bad)}")

    unlinked_behavior_tracks = _track_id_threshold_stats(actions, threshold=200000)
    track_backend = str(args.track_backend).strip().lower()
    tracked_students = _count_people_from_pose(track_rows)
    student_track_students = _count_people_from_pose(student_track_rows)
    if track_backend == "hybrid":
        if student_track_students != tracked_students:
            errors.append(
                f"hybrid student identity mismatch: student_tracks has {student_track_students} students, pose_tracks has {tracked_students}"
            )
        if int(unlinked_behavior_tracks["rows"]) > 0:
            errors.append(
                f"hybrid actions.fusion_v2.jsonl contains unlinked behavior track ids >= 200000: {unlinked_behavior_tracks['rows']} rows"
            )

    asr_report = _load_json(paths["asr_quality_report"])
    if not isinstance(asr_report, dict):
        warnings.append("asr_quality_report.json missing; ASR quality gate could not be audited")
    elif asr_report.get("status") != "ok":
        warnings.append(f"ASR quality status is {asr_report.get('status')}; visual fallback should be used")

    result = {
        "stage": "check_pipeline_contract_v2",
        "output_dir": str(output_dir),
        "files": files,
        "counts": {
            "track_backend": str(args.track_backend),
            "pose_rows": len(pose_rows),
            "pose_track_rows": len(track_rows),
            "tracked_students": tracked_students,
            "student_track_rows": len(student_track_rows),
            "student_track_students": student_track_students,
            "behavior_items": _count_behavior_items(behavior_rows),
            "actions_fusion_v2": action_total,
            "actions_fusion_v2_semantic_valid": action_valid,
            "actions_fusion_v2_unlinked_rows_ge_200000": int(unlinked_behavior_tracks["rows"]),
            "actions_fusion_v2_unlinked_track_ids_ge_200000": int(unlinked_behavior_tracks["track_ids"]),
            "event_queries_fusion_v2": len(event_queries),
            "align_events": align["events"],
            "align_total_candidates": align["total_candidates"],
            "verified_events": len(verified),
            "student_count": _student_count(paths["student_id_map"]),
            "timeline_student_rows": len(timeline_rows),
        },
        "asr_quality": asr_report if isinstance(asr_report, dict) else {},
        "action_semantic_invalid": action_invalid[:20],
        "action_unlinked_examples": unlinked_behavior_tracks["examples"],
        "timeline_csv_invalid": csv_bad[:20],
        "warnings": warnings,
        "errors": errors,
        "status": "ok" if not errors else "failed",
    }
    write_json(report_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if int(args.strict) == 1 and result["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
