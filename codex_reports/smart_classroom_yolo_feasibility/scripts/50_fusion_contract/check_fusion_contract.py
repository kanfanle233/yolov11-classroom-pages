from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from fusion_utils import REQUIRED_ACTION_FIELDS, load_json, read_jsonl, resolve_path, resolve_repo_root, semantic_coverage, write_json


def _file_state(path: Path) -> Dict[str, Any]:
    return {"path": str(path), "exists": path.exists(), "bytes": path.stat().st_size if path.exists() and path.is_file() else 0}


def _align_stats(path: Path) -> Dict[str, Any]:
    if not path.exists() or path.stat().st_size <= 0:
        return {"events": 0, "events_without_candidates": 0}
    obj = load_json(path)
    events = obj if isinstance(obj, list) else []
    no_candidates = 0
    total_candidates = 0
    for row in events:
        candidates = row.get("candidates", []) if isinstance(row, dict) else []
        if not isinstance(candidates, list) or len(candidates) == 0:
            no_candidates += 1
        else:
            total_candidates += len(candidates)
    return {"events": len(events), "events_without_candidates": no_candidates, "total_candidates": total_candidates}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate fusion_contract_v2 outputs.")
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--report", default="", type=str)
    parser.add_argument("--strict", default=1, type=int)
    args = parser.parse_args()

    repo_root = resolve_repo_root(Path(__file__))
    output_dir = resolve_path(repo_root, args.output_dir)
    report_path = resolve_path(repo_root, args.report) if args.report else output_dir / "fusion_contract_report.json"

    paths = {
        "actions_fusion_v2": output_dir / "actions.fusion_v2.jsonl",
        "event_queries_fusion_v2": output_dir / "event_queries.fusion_v2.jsonl",
        "event_queries_visual_fallback": output_dir / "event_queries.visual_fallback.jsonl",
        "align_multimodal": output_dir / "align_multimodal.json",
        "verified_events": output_dir / "verified_events.jsonl",
        "transcript": output_dir / "transcript.jsonl",
        "behavior_det_semantic": output_dir / "behavior_det.semantic.jsonl",
        "actions_behavior_semantic": output_dir / "actions.behavior.semantic.jsonl",
        "objects_semantic": output_dir / "objects.semantic.jsonl",
    }
    files = {name: _file_state(path) for name, path in paths.items()}

    hard_errors: List[str] = []
    warnings: List[str] = []
    for required in ["actions_fusion_v2", "event_queries_fusion_v2", "align_multimodal", "verified_events"]:
        state = files[required]
        if not state["exists"]:
            hard_errors.append(f"missing required file: {state['path']}")
        elif int(state["bytes"]) <= 0:
            hard_errors.append(f"empty required file: {state['path']}")

    actions = read_jsonl(paths["actions_fusion_v2"])
    event_queries = read_jsonl(paths["event_queries_fusion_v2"])
    verified = read_jsonl(paths["verified_events"])
    total, valid, invalid = semantic_coverage(actions)
    if total <= 0:
        hard_errors.append("actions.fusion_v2.jsonl has zero rows")
    if invalid:
        hard_errors.append(f"semantic coverage failed for {len(invalid)} action rows")

    align = _align_stats(paths["align_multimodal"])
    if align["events"] > 0 and align["events_without_candidates"] > 0 and total > 0:
        hard_errors.append(f"align events without candidates: {align['events_without_candidates']}")
    if verified and len(verified) <= 1:
        transcript_text = "\n".join(str(r.get("text", "")) for r in read_jsonl(paths["transcript"])).lower()
        if "asr_empty" in transcript_text:
            warnings.append("verified_events has <=1 row and transcript is ASR_EMPTY; visual fallback should be inspected")
    if len(verified) <= 0:
        hard_errors.append("verified_events.jsonl has zero rows")

    result = {
        "stage": "check_fusion_contract",
        "output_dir": str(output_dir),
        "required_action_fields": REQUIRED_ACTION_FIELDS,
        "files": files,
        "counts": {
            "actions_fusion_v2": total,
            "actions_fusion_v2_semantic_valid": valid,
            "event_queries_fusion_v2": len(event_queries),
            "verified_events": len(verified),
            "align_events": align["events"],
            "align_total_candidates": align.get("total_candidates", 0),
        },
        "semantic_invalid": invalid[:50],
        "warnings": warnings,
        "errors": hard_errors,
        "status": "ok" if not hard_errors else "failed",
    }
    write_json(report_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if int(args.strict) == 1 and result["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
