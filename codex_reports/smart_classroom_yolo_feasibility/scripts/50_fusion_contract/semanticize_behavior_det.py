from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from fusion_utils import load_taxonomy, read_jsonl, resolve_path, resolve_repo_root, write_json, write_jsonl


def semanticize_rows(rows: List[Dict[str, Any]], taxonomy) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    out_rows: List[Dict[str, Any]] = []
    stats = {
        "rows": len(rows),
        "behavior_items": 0,
        "fallback_items": 0,
        "empty_behavior_rows": 0,
        "fallback_examples": [],
    }
    for row_idx, row in enumerate(rows):
        out_row = dict(row)
        behaviors = out_row.get("behaviors", [])
        if not isinstance(behaviors, list):
            behaviors = []
        if not behaviors:
            stats["empty_behavior_rows"] += 1
        out_behaviors: List[Dict[str, Any]] = []
        for item_idx, item in enumerate(behaviors):
            if not isinstance(item, dict):
                continue
            det = dict(item)
            entry, source = taxonomy.resolve(
                [
                    det.get("behavior_code"),
                    det.get("label"),
                    det.get("semantic_id"),
                    det.get("action"),
                    det.get("name"),
                ]
            )
            raw_action = str(det.get("action", "")).strip().lower()
            det.setdefault("raw_action", raw_action)
            taxonomy.apply(det, entry)
            det["action"] = entry["semantic_id"]
            det["label"] = str(det.get("label", entry["behavior_code"])).strip().lower() or entry["behavior_code"]
            det["semantic_source"] = source
            stats["behavior_items"] += 1
            if source == "fallback":
                stats["fallback_items"] += 1
                if len(stats["fallback_examples"]) < 20:
                    stats["fallback_examples"].append({"row": row_idx, "item": item_idx, "raw": item})
            out_behaviors.append(det)
        out_row["behaviors"] = out_behaviors
        out_row["taxonomy_version"] = taxonomy.version
        out_rows.append(out_row)
    return out_rows, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Add canonical 8-class semantics to behavior_det.jsonl.")
    parser.add_argument("--in", dest="in_path", required=True, type=str)
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--taxonomy", required=True, type=str)
    parser.add_argument("--report", default="", type=str)
    parser.add_argument("--strict", default=1, type=int)
    args = parser.parse_args()

    repo_root = resolve_repo_root(Path(__file__))
    in_path = resolve_path(repo_root, args.in_path)
    out_path = resolve_path(repo_root, args.out)
    taxonomy_path = resolve_path(repo_root, args.taxonomy)
    report_path = resolve_path(repo_root, args.report) if args.report else out_path.with_suffix(".report.json")

    if not in_path.exists():
        raise FileNotFoundError(f"behavior_det input not found: {in_path}")
    taxonomy = load_taxonomy(taxonomy_path)
    rows = read_jsonl(in_path)
    semantic_rows, stats = semanticize_rows(rows, taxonomy)
    written = write_jsonl(out_path, semantic_rows)

    report = {
        "stage": "semanticize_behavior_det",
        "input": str(in_path),
        "output": str(out_path),
        "taxonomy": str(taxonomy_path),
        "taxonomy_version": taxonomy.version,
        "rows_written": written,
        "stats": stats,
        "status": "ok" if (stats["fallback_items"] == 0 or int(args.strict) != 1) else "warning_fallback",
    }
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
