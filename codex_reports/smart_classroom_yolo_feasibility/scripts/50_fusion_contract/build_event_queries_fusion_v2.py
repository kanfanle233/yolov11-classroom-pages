from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from fusion_utils import read_jsonl, resolve_path, resolve_repo_root, safe_float, write_json, write_jsonl

REPO_ROOT = resolve_repo_root(Path(__file__))
sys.path.insert(0, str(REPO_ROOT))

from contracts.schemas import SCHEMA_VERSION, validate_event_query_record, validate_jsonl_file  # noqa: E402


def _is_asr_empty(rows: List[Dict[str, Any]]) -> bool:
    if not rows:
        return True
    text = " ".join(str(r.get("query_text", "")) + " " + str(r.get("trigger_text", "")) for r in rows).lower()
    return "[asr_empty" in text or "asr_empty" in text


def _is_asr_placeholder_row(row: Dict[str, Any]) -> bool:
    text = (str(row.get("query_text", "")) + " " + str(row.get("trigger_text", ""))).lower()
    return "[asr_empty" in text or "asr_empty" in text


def _make_visual_query(idx: int, row: Dict[str, Any]) -> Dict[str, Any]:
    st = safe_float(row.get("start_time", row.get("t", 0.0)), 0.0)
    ed = safe_float(row.get("end_time", st + 0.2), st + 0.2)
    if ed < st:
        st, ed = ed, st
    if ed <= st:
        ed = st + 0.2
    semantic_id = str(row.get("semantic_id", row.get("action", "unknown"))).strip().lower() or "unknown"
    label_en = str(row.get("semantic_label_en", semantic_id)).strip() or semantic_id
    label_zh = str(row.get("semantic_label_zh", "")).strip()
    t_center = 0.5 * (st + ed)
    confidence = max(0.1, min(0.95, safe_float(row.get("conf", 0.5), 0.5)))
    return {
        "event_id": f"v_{idx:06d}",
        "query_id": f"v_{idx:06d}",
        "schema_version": SCHEMA_VERSION,
        "event_type": semantic_id,
        "query_text": label_en,
        "trigger_text": f"visual_fallback:{label_en}" + (f"/{label_zh}" if label_zh else ""),
        "trigger_words": [semantic_id],
        "timestamp": round(t_center, 3),
        "t_center": round(t_center, 3),
        "start": round(st, 3),
        "end": round(ed, 3),
        "confidence": round(confidence, 4),
        "source": "visual_fallback",
        "track_id": row.get("track_id"),
        "behavior_code": row.get("behavior_code", ""),
        "semantic_id": semantic_id,
        "semantic_label_zh": label_zh,
        "semantic_label_en": label_en,
        "taxonomy_version": row.get("taxonomy_version", ""),
    }


def _visual_fallback_queries(actions: List[Dict[str, Any]], topk: int, min_conf: float) -> List[Dict[str, Any]]:
    candidates = []
    seen = set()
    for row in actions:
        conf = safe_float(row.get("conf", 0.0), 0.0)
        if conf < min_conf:
            continue
        semantic_id = str(row.get("semantic_id", row.get("action", ""))).strip().lower()
        track_id = row.get("track_id")
        bucket = int(safe_float(row.get("start_time", row.get("t", 0.0)), 0.0) // 2)
        key = (track_id, semantic_id, bucket)
        if key in seen:
            continue
        seen.add(key)
        duration = max(0.2, safe_float(row.get("end_time", 0.0), 0.0) - safe_float(row.get("start_time", 0.0), 0.0))
        score = conf * min(3.0, duration)
        candidates.append((score, row))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [_make_visual_query(i, row) for i, (_, row) in enumerate(candidates[: max(1, topk)])]


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge ASR event queries with visual fallback queries for fusion v2.")
    parser.add_argument("--event_queries", required=True, type=str)
    parser.add_argument("--actions", required=True, type=str)
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--visual_out", default="", type=str)
    parser.add_argument("--report", default="", type=str)
    parser.add_argument("--min_asr_queries", default=2, type=int)
    parser.add_argument("--visual_topk", default=12, type=int)
    parser.add_argument("--visual_min_conf", default=0.35, type=float)
    parser.add_argument("--validate", default=1, type=int)
    args = parser.parse_args()

    repo_root = REPO_ROOT
    event_path = resolve_path(repo_root, args.event_queries)
    actions_path = resolve_path(repo_root, args.actions)
    out_path = resolve_path(repo_root, args.out)
    visual_out = resolve_path(repo_root, args.visual_out) if args.visual_out else out_path.with_name("event_queries.visual_fallback.jsonl")
    report_path = resolve_path(repo_root, args.report) if args.report else out_path.with_suffix(".report.json")

    asr_rows = read_jsonl(event_path)
    actions = read_jsonl(actions_path)
    asr_empty = _is_asr_empty(asr_rows)
    needs_visual = asr_empty or len(asr_rows) < max(0, int(args.min_asr_queries))
    visual_rows = _visual_fallback_queries(actions, topk=int(args.visual_topk), min_conf=float(args.visual_min_conf)) if needs_visual else []
    asr_rows_used = [row for row in asr_rows if not _is_asr_placeholder_row(row)] if asr_empty else list(asr_rows)
    merged = asr_rows_used + visual_rows
    write_jsonl(visual_out, visual_rows)
    written = write_jsonl(out_path, merged)

    if int(args.validate) == 1:
        ok, _, errors = validate_jsonl_file(out_path, validate_event_query_record)
        if not ok:
            raise ValueError(f"invalid fusion event query: {errors[0] if errors else 'unknown schema error'}")

    report = {
        "stage": "build_event_queries_fusion_v2",
        "event_queries": str(event_path),
        "actions": str(actions_path),
        "output": str(out_path),
        "visual_output": str(visual_out),
        "asr_rows": len(asr_rows),
        "asr_rows_used": len(asr_rows_used),
        "action_rows": len(actions),
        "asr_empty": asr_empty,
        "visual_fallback_rows": len(visual_rows),
        "rows_written": written,
        "status": "ok" if written > 0 else "failed",
    }
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if written <= 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
