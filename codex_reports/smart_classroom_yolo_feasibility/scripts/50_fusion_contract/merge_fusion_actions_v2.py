from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from fusion_utils import (
    REQUIRED_ACTION_FIELDS,
    Taxonomy,
    clamp01,
    load_taxonomy,
    read_jsonl,
    resolve_path,
    resolve_repo_root,
    safe_float,
    semantic_coverage,
    write_json,
    write_jsonl,
)


def _time_of(row: Dict[str, Any]) -> float:
    return safe_float(row.get("t", row.get("start_time", row.get("start", 0.0))), 0.0)


def _object_index(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    indexed: List[Dict[str, Any]] = []
    for row in rows:
        t = _time_of(row)
        for obj in row.get("objects", []) if isinstance(row.get("objects"), list) else []:
            if not isinstance(obj, dict):
                continue
            indexed.append(
                {
                    "t": t,
                    "object_type": str(obj.get("object_type", obj.get("name", ""))).strip().lower(),
                    "conf": clamp01(safe_float(obj.get("conf", 0.0), 0.0)),
                    "bbox": obj.get("bbox", []),
                    "support_actions": obj.get("support_actions", {}) if isinstance(obj.get("support_actions"), dict) else {},
                }
            )
    return indexed


def _object_evidence(objects: List[Dict[str, Any]], *, t: float, semantic_id: str, window: float) -> Dict[str, Any]:
    nearby = [obj for obj in objects if abs(float(obj["t"]) - float(t)) <= window]
    top_objects = sorted(nearby, key=lambda x: float(x.get("conf", 0.0)), reverse=True)[:5]
    support = 0.0
    for obj in nearby:
        priors = obj.get("support_actions", {})
        if isinstance(priors, dict) and semantic_id in priors:
            support = max(support, clamp01(float(obj.get("conf", 0.0)) * safe_float(priors.get(semantic_id), 0.0)))
    return {
        "support_score": round(support, 6),
        "objects": [
            {
                "object_type": obj.get("object_type", ""),
                "conf": round(float(obj.get("conf", 0.0)), 6),
                "support": obj.get("support_actions", {}),
            }
            for obj in top_objects
        ],
    }


def _normalize_action(row: Dict[str, Any], taxonomy: Taxonomy, *, source: str, track_offset: int = 0) -> Dict[str, Any]:
    entry, semantic_source = taxonomy.resolve(
        [
            row.get("behavior_code"),
            row.get("semantic_id"),
            row.get("action"),
            row.get("label"),
            row.get("event_type"),
        ]
    )
    st = safe_float(row.get("start_time", row.get("start", row.get("t", 0.0))), 0.0)
    ed = safe_float(row.get("end_time", row.get("end", st + 0.2)), st + 0.2)
    if ed < st:
        st, ed = ed, st
    if ed <= st:
        ed = st + 0.2
    try:
        tid = int(row.get("track_id", 0))
    except Exception:
        tid = 0
    tid += int(track_offset)

    out = dict(row)
    raw_action = str(row.get("raw_action", row.get("action", ""))).strip().lower()
    taxonomy.apply(out, entry)
    out.update(
        {
            "track_id": tid,
            "action": entry["semantic_id"],
            "raw_action": raw_action,
            "conf": clamp01(safe_float(row.get("conf", row.get("confidence", 0.5)), 0.5)),
            "start_time": float(st),
            "end_time": float(ed),
            "start_frame": int(row.get("start_frame", max(0, int(round(st * 25.0))))),
            "end_frame": int(row.get("end_frame", max(0, int(round(ed * 25.0))))),
            "frame": int(row.get("frame", max(0, int(round(st * 25.0))))),
            "t": safe_float(row.get("t", st), st),
            "source": source,
            "semantic_source": semantic_source,
        }
    )
    return out


def _merge_close(rows: List[Dict[str, Any]], gap_tol: float) -> List[Dict[str, Any]]:
    if not rows:
        return []
    rows = sorted(rows, key=lambda r: (int(r["track_id"]), float(r["start_time"]), float(r["end_time"]), str(r["semantic_id"])))
    out: List[Dict[str, Any]] = [dict(rows[0])]
    for row in rows[1:]:
        last = out[-1]
        same_track = int(row["track_id"]) == int(last["track_id"])
        same_semantic = str(row["semantic_id"]) == str(last["semantic_id"])
        same_source = str(row.get("source", "")) == str(last.get("source", ""))
        close = float(row["start_time"]) <= float(last["end_time"]) + gap_tol
        if same_track and same_semantic and same_source and close:
            last["end_time"] = max(float(last["end_time"]), float(row["end_time"]))
            last["end_frame"] = max(int(last["end_frame"]), int(row["end_frame"]))
            last["duration"] = round(float(last["end_time"]) - float(last["start_time"]), 6)
            last["conf"] = round((float(last["conf"]) + float(row["conf"])) * 0.5, 6)
        else:
            out.append(dict(row))
    return out


def merge_actions(
    *,
    primary_rows: List[Dict[str, Any]],
    behavior_rows: List[Dict[str, Any]],
    object_rows: List[Dict[str, Any]],
    taxonomy: Taxonomy,
    behavior_track_offset: int,
    object_window: float,
    object_beta: float,
    merge_gap_sec: float,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for row in primary_rows:
        normalized.append(_normalize_action(row, taxonomy, source=str(row.get("source", "pose_action")), track_offset=0))
    for row in behavior_rows:
        normalized.append(_normalize_action(row, taxonomy, source="behavior_det_v2", track_offset=behavior_track_offset))

    objects = _object_index(object_rows)
    object_supported_rows = 0
    for row in normalized:
        evidence = _object_evidence(objects, t=safe_float(row.get("t", row.get("start_time", 0.0))), semantic_id=str(row["semantic_id"]), window=object_window)
        row["object_evidence"] = evidence
        support = safe_float(evidence.get("support_score"), 0.0)
        if support > 0:
            object_supported_rows += 1
            row["conf"] = clamp01((1.0 - object_beta) * safe_float(row.get("conf", 0.5), 0.5) + object_beta * support)
    merged = _merge_close(normalized, gap_tol=max(0.01, merge_gap_sec))

    total, valid, invalid = semantic_coverage(merged)
    stats = {
        "primary_rows": len(primary_rows),
        "behavior_rows": len(behavior_rows),
        "object_rows": len(object_rows),
        "object_items": len(objects),
        "object_supported_rows": object_supported_rows,
        "merged_rows": len(merged),
        "semantic_total": total,
        "semantic_valid": valid,
        "semantic_invalid": len(invalid),
        "required_fields": REQUIRED_ACTION_FIELDS,
    }
    if invalid:
        stats["invalid_examples"] = invalid[:10]
    return merged, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge pose/rule, object, and semantic behavior actions into actions.fusion_v2.jsonl.")
    parser.add_argument("--actions", required=True, type=str)
    parser.add_argument("--behavior_actions", default="", type=str)
    parser.add_argument("--objects", default="", type=str)
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--taxonomy", required=True, type=str)
    parser.add_argument("--report", default="", type=str)
    parser.add_argument("--behavior_track_offset", default=100000, type=int)
    parser.add_argument("--behavior_track_mode", choices=["offset", "linked"], default="offset")
    parser.add_argument("--object_window", default=0.8, type=float)
    parser.add_argument("--object_beta", default=0.20, type=float)
    parser.add_argument("--merge_gap_sec", default=0.22, type=float)
    parser.add_argument("--strict", default=1, type=int)
    args = parser.parse_args()

    repo_root = resolve_repo_root(Path(__file__))
    actions_path = resolve_path(repo_root, args.actions)
    behavior_path = resolve_path(repo_root, args.behavior_actions) if args.behavior_actions else None
    objects_path = resolve_path(repo_root, args.objects) if args.objects else None
    out_path = resolve_path(repo_root, args.out)
    taxonomy_path = resolve_path(repo_root, args.taxonomy)
    report_path = resolve_path(repo_root, args.report) if args.report else out_path.with_suffix(".report.json")

    if not actions_path.exists():
        raise FileNotFoundError(f"primary actions not found: {actions_path}")
    taxonomy = load_taxonomy(taxonomy_path)
    primary_rows = read_jsonl(actions_path)
    behavior_rows = read_jsonl(behavior_path) if behavior_path and behavior_path.exists() else []
    object_rows = read_jsonl(objects_path) if objects_path and objects_path.exists() else []
    behavior_track_offset = 0 if str(args.behavior_track_mode).strip().lower() == "linked" else int(args.behavior_track_offset)
    merged, stats = merge_actions(
        primary_rows=primary_rows,
        behavior_rows=behavior_rows,
        object_rows=object_rows,
        taxonomy=taxonomy,
        behavior_track_offset=behavior_track_offset,
        object_window=max(0.0, float(args.object_window)),
        object_beta=max(0.0, min(1.0, float(args.object_beta))),
        merge_gap_sec=max(0.01, float(args.merge_gap_sec)),
    )
    stats["behavior_track_mode"] = str(args.behavior_track_mode)
    stats["behavior_track_offset_applied"] = int(behavior_track_offset)
    written = write_jsonl(out_path, merged)
    status = "ok" if stats["semantic_invalid"] == 0 and written > 0 else "failed"
    report = {"stage": "merge_fusion_actions_v2", "output": str(out_path), "rows_written": written, "stats": stats, "status": status}
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if int(args.strict) == 1 and status != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
