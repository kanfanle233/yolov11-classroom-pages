from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from fusion_utils import bbox_iou, clamp01, read_jsonl, resolve_path, resolve_repo_root, safe_float, write_json, write_jsonl


def _semantic_key(det: Dict[str, Any]) -> str:
    return str(det.get("semantic_id", det.get("action", det.get("label", "")))).strip().lower()


def _merge_close(rows: List[Dict[str, Any]], gap_tol: float) -> List[Dict[str, Any]]:
    if not rows:
        return []
    rows = sorted(rows, key=lambda r: (int(r["track_id"]), float(r["start_time"]), float(r["end_time"]), str(r["semantic_id"])))
    out: List[Dict[str, Any]] = [dict(rows[0])]
    for row in rows[1:]:
        last = out[-1]
        same_track = int(row["track_id"]) == int(last["track_id"])
        same_semantic = str(row["semantic_id"]) == str(last["semantic_id"])
        close = float(row["start_time"]) <= float(last["end_time"]) + gap_tol
        if same_track and same_semantic and close:
            last["end_time"] = max(float(last["end_time"]), float(row["end_time"]))
            last["end_frame"] = max(int(last["end_frame"]), int(row["end_frame"]))
            last["duration"] = round(float(last["end_time"]) - float(last["start_time"]), 6)
            last["conf"] = round((float(last["conf"]) + float(row["conf"])) * 0.5, 6)
            last["frame_count"] = int(last.get("frame_count", 1)) + int(row.get("frame_count", 1))
            if row.get("bbox"):
                last["bbox"] = row["bbox"]
        else:
            out.append(dict(row))
    return out


def _is_student_tracks(rows: List[Dict[str, Any]]) -> bool:
    for row in rows[:20]:
        persons = row.get("persons", row.get("people", None))
        if isinstance(persons, list):
            return True
    return False


def _convert_student_track_rows(
    rows: List[Dict[str, Any]],
    *,
    fps: float,
    merge_gap_sec: float,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    action_rows: List[Dict[str, Any]] = []
    skipped = 0
    unmatched_skipped = 0
    for row in sorted(rows, key=lambda x: int(x.get("frame", 0))):
        frame = int(row.get("frame", 0))
        t = safe_float(row.get("t", frame / fps), frame / fps)
        persons = row.get("persons", row.get("people", []))
        if isinstance(persons, dict):
            persons = list(persons.values())
        if not isinstance(persons, list):
            continue
        for person in persons:
            if not isinstance(person, dict):
                skipped += 1
                continue
            try:
                track_id = int(person.get("track_id"))
            except Exception:
                skipped += 1
                continue
            match_status = str(person.get("behavior_match_status", "")).strip().lower()
            if match_status and match_status != "matched":
                unmatched_skipped += 1
                continue
            semantic_id = _semantic_key(person)
            behavior_code = str(person.get("behavior_code", "")).strip().lower()
            bbox = person.get("bbox", [])
            behavior_bbox = person.get("behavior_bbox", [])
            if not semantic_id or not behavior_code:
                skipped += 1
                continue
            if not (isinstance(bbox, list) and len(bbox) == 4):
                bbox = []
            if not (isinstance(behavior_bbox, list) and len(behavior_bbox) == 4):
                behavior_bbox = []

            action_rows.append(
                {
                    "track_id": int(track_id),
                    "behavior_track_id": person.get("behavior_track_id"),
                    "linked_pose_track_id": person.get("linked_pose_track_id"),
                    "track_link_score": person.get("track_link_score"),
                    "track_link_source": person.get("track_link_source"),
                    "behavior_match_status": person.get("behavior_match_status"),
                    "behavior_match_score": person.get("behavior_match_score"),
                    "behavior_bbox": [float(v) for v in behavior_bbox] if behavior_bbox else [],
                    "action": semantic_id,
                    "raw_action": str(person.get("raw_action", person.get("action", ""))).strip().lower(),
                    "behavior_code": behavior_code,
                    "behavior_label_zh": str(person.get("behavior_label_zh", "")).strip(),
                    "behavior_label_en": str(person.get("behavior_label_en", "")).strip(),
                    "semantic_id": semantic_id,
                    "semantic_label_zh": str(person.get("semantic_label_zh", "")).strip(),
                    "semantic_label_en": str(person.get("semantic_label_en", "")).strip(),
                    "taxonomy_version": str(person.get("taxonomy_version", row.get("taxonomy_version", ""))).strip(),
                    "action_code": int(person.get("action_code", 0) or 0),
                    "conf": clamp01(safe_float(person.get("conf", person.get("track_conf", 0.5)), 0.5)),
                    "start_time": float(t),
                    "end_time": float(t + (1.0 / fps)),
                    "start_frame": int(frame),
                    "end_frame": int(frame + 1),
                    "frame": int(frame),
                    "t": float(t),
                    "bbox": [float(v) for v in bbox] if bbox else [],
                    "source": str(person.get("source", "behavior_student_tracker")).strip() or "behavior_student_tracker",
                    "frame_count": 1,
                }
            )

    merged = _merge_close(action_rows, gap_tol=max(0.01, merge_gap_sec))
    return merged, {
        "input_rows": len(rows),
        "input_kind": "student_tracks",
        "raw_action_rows": len(action_rows),
        "merged_rows": len(merged),
        "skipped": skipped,
        "unmatched_skipped": unmatched_skipped,
    }


def convert_rows(
    rows: List[Dict[str, Any]],
    *,
    fps: float,
    iou_thres: float,
    max_gap_frames: int,
    merge_gap_sec: float,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if _is_student_tracks(rows):
        return _convert_student_track_rows(rows, fps=fps, merge_gap_sec=merge_gap_sec)

    active: Dict[int, Tuple[int, List[float]]] = {}
    next_tid = 1
    action_rows: List[Dict[str, Any]] = []
    skipped = 0

    for row in sorted(rows, key=lambda x: int(x.get("frame", 0))):
        frame = int(row.get("frame", 0))
        t = safe_float(row.get("t", frame / fps), frame / fps)
        behaviors = row.get("behaviors", [])
        if not isinstance(behaviors, list):
            continue

        used_track_ids: set[int] = set()
        for det in behaviors:
            if not isinstance(det, dict):
                skipped += 1
                continue
            bbox = det.get("bbox", [])
            if not (isinstance(bbox, list) and len(bbox) == 4):
                skipped += 1
                continue
            semantic_id = _semantic_key(det)
            behavior_code = str(det.get("behavior_code", "")).strip().lower()
            if not semantic_id or not behavior_code:
                skipped += 1
                continue

            best_tid = -1
            best_iou = -1.0
            for tid, (last_frame, last_bbox) in active.items():
                if tid in used_track_ids:
                    continue
                if frame - last_frame > max_gap_frames:
                    continue
                score = bbox_iou([float(v) for v in bbox], last_bbox)
                if score >= iou_thres and score > best_iou:
                    best_iou = score
                    best_tid = tid

            if best_tid < 0:
                best_tid = next_tid
                next_tid += 1

            active[best_tid] = (frame, [float(v) for v in bbox])
            used_track_ids.add(best_tid)

            action_rows.append(
                {
                    "track_id": int(best_tid),
                    "action": semantic_id,
                    "raw_action": str(det.get("raw_action", det.get("action", ""))).strip().lower(),
                    "behavior_code": behavior_code,
                    "behavior_label_zh": str(det.get("behavior_label_zh", "")).strip(),
                    "behavior_label_en": str(det.get("behavior_label_en", "")).strip(),
                    "semantic_id": semantic_id,
                    "semantic_label_zh": str(det.get("semantic_label_zh", "")).strip(),
                    "semantic_label_en": str(det.get("semantic_label_en", "")).strip(),
                    "taxonomy_version": str(det.get("taxonomy_version", "")).strip(),
                    "action_code": int(det.get("action_code", 0) or 0),
                    "conf": clamp01(safe_float(det.get("conf", 0.5), 0.5)),
                    "start_time": float(t),
                    "end_time": float(t + (1.0 / fps)),
                    "start_frame": int(frame),
                    "end_frame": int(frame + 1),
                    "frame": int(frame),
                    "t": float(t),
                    "bbox": [float(v) for v in bbox],
                    "source": "behavior_det_v2",
                    "frame_count": 1,
                }
            )

        stale_tids = [tid for tid, (last_frame, _) in active.items() if frame - last_frame > (max_gap_frames * 3)]
        for tid in stale_tids:
            active.pop(tid, None)

    merged = _merge_close(action_rows, gap_tol=max(0.01, merge_gap_sec))
    return merged, {
        "input_rows": len(rows),
        "input_kind": "behavior_det_semantic",
        "raw_action_rows": len(action_rows),
        "merged_rows": len(merged),
        "skipped": skipped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert semantic behavior detections to semantic action rows.")
    parser.add_argument("--in", dest="in_path", required=True, type=str)
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--fps", default=25.0, type=float)
    parser.add_argument("--iou_thres", default=0.30, type=float)
    parser.add_argument("--max_gap_frames", default=3, type=int)
    parser.add_argument("--merge_gap_sec", default=0.22, type=float)
    parser.add_argument("--report", default="", type=str)
    args = parser.parse_args()

    repo_root = resolve_repo_root(Path(__file__))
    in_path = resolve_path(repo_root, args.in_path)
    out_path = resolve_path(repo_root, args.out)
    report_path = resolve_path(repo_root, args.report) if args.report else out_path.with_suffix(".report.json")

    if not in_path.exists():
        raise FileNotFoundError(f"semantic behavior input not found: {in_path}")
    rows = read_jsonl(in_path)
    actions, stats = convert_rows(
        rows,
        fps=float(args.fps) if float(args.fps) > 0 else 25.0,
        iou_thres=max(0.0, min(1.0, float(args.iou_thres))),
        max_gap_frames=max(1, int(args.max_gap_frames)),
        merge_gap_sec=max(0.01, float(args.merge_gap_sec)),
    )
    written = write_jsonl(out_path, actions)
    report = {"stage": "behavior_det_to_actions_v2", "input": str(in_path), "output": str(out_path), "rows_written": written, "stats": stats}
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
