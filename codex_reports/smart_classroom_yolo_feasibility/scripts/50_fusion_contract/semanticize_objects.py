from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from fusion_utils import normalize_object_name, object_support_actions, read_jsonl, resolve_path, resolve_repo_root, write_json, write_jsonl


def semanticize_objects(rows: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    out_rows: List[Dict[str, Any]] = []
    object_items = 0
    supported_items = 0
    for row in rows:
        out_row = dict(row)
        objects = out_row.get("objects", [])
        if not isinstance(objects, list):
            objects = []
        out_objects: List[Dict[str, Any]] = []
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            item = dict(obj)
            raw_name = item.get("name", item.get("label", ""))
            object_type = normalize_object_name(raw_name)
            support_actions = object_support_actions(object_type)
            item["raw_name"] = str(raw_name)
            item["object_type"] = object_type
            item["support_actions"] = support_actions
            item["source"] = str(item.get("source", "object_det"))
            object_items += 1
            if support_actions:
                supported_items += 1
            out_objects.append(item)
        out_row["objects"] = out_objects
        out_rows.append(out_row)
    return out_rows, {"rows": len(rows), "object_items": object_items, "supported_items": supported_items}


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize classroom object detections for fusion evidence.")
    parser.add_argument("--in", dest="in_path", required=True, type=str)
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--report", default="", type=str)
    args = parser.parse_args()

    repo_root = resolve_repo_root(Path(__file__))
    in_path = resolve_path(repo_root, args.in_path)
    out_path = resolve_path(repo_root, args.out)
    report_path = resolve_path(repo_root, args.report) if args.report else out_path.with_suffix(".report.json")

    rows = read_jsonl(in_path)
    semantic_rows, stats = semanticize_objects(rows)
    written = write_jsonl(out_path, semantic_rows)
    report = {"stage": "semanticize_objects", "input": str(in_path), "output": str(out_path), "rows_written": written, "stats": stats}
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
