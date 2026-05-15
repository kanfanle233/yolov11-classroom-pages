from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_REQUIRED = [
    "behavior_det.semantic.jsonl",
    "actions.behavior.semantic.jsonl",
    "actions.behavior_aug.semantic.jsonl",
    "align_multimodal.semantic.json",
    "verified_events.semantic.jsonl",
    "semantics_manifest.json",
]

REQUIRED_FIELDS = [
    "behavior_code",
    "behavior_label_zh",
    "behavior_label_en",
    "semantic_id",
    "semantic_label_zh",
    "semantic_label_en",
    "taxonomy_version",
]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _check_item(item: Dict[str, Any], valid_codes: set[str], context: str, invalid: List[Dict[str, Any]]) -> None:
    missing = [k for k in REQUIRED_FIELDS if not str(item.get(k, "")).strip()]
    code = str(item.get("behavior_code", "")).strip().lower()
    if missing or code not in valid_codes:
        invalid.append(
            {
                "context": context,
                "missing_fields": missing,
                "behavior_code": code,
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate semantic bridge outputs.")
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--required", action="append", default=[])
    parser.add_argument("--report", default="", type=str)
    parser.add_argument("--strict", default=1, type=int)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    required = list(args.required or []) or list(DEFAULT_REQUIRED)

    present: List[str] = []
    missing: List[str] = []
    empty: List[str] = []
    for rel in required:
        p = (output_dir / rel).resolve()
        if not p.exists():
            missing.append(str(p))
            continue
        if p.is_file() and p.stat().st_size <= 0:
            empty.append(str(p))
            continue
        present.append(str(p))

    invalid: List[Dict[str, Any]] = []
    valid_codes = set()
    manifest_path = output_dir / "semantics_manifest.json"
    if manifest_path.exists():
        manifest = _load_json(manifest_path)
        files = manifest.get("files", {})
        if isinstance(files, dict):
            taxonomy_path = manifest.get("taxonomy_path")
            if isinstance(taxonomy_path, str) and taxonomy_path.strip():
                tp = Path(taxonomy_path).resolve()
                if tp.exists():
                    taxonomy = _load_json(tp)
                    classes = taxonomy.get("classes", [])
                    if isinstance(classes, list):
                        for c in classes:
                            if isinstance(c, dict):
                                code = str(c.get("behavior_code", "")).strip().lower()
                                if code:
                                    valid_codes.add(code)
    if not valid_codes:
        valid_codes = {"tt", "dx", "dk", "zt", "xt", "js", "zl", "jz"}

    behavior_path = output_dir / "behavior_det.semantic.jsonl"
    if behavior_path.exists():
        for i, row in enumerate(_load_jsonl(behavior_path)):
            behaviors = row.get("behaviors", [])
            if not isinstance(behaviors, list):
                continue
            for j, item in enumerate(behaviors):
                if isinstance(item, dict):
                    _check_item(item, valid_codes, f"behavior_det[{i}].behaviors[{j}]", invalid)

    for name in ["actions.behavior.semantic.jsonl", "actions.behavior_aug.semantic.jsonl", "verified_events.semantic.jsonl"]:
        p = output_dir / name
        if not p.exists():
            continue
        for i, row in enumerate(_load_jsonl(p)):
            _check_item(row, valid_codes, f"{name}[{i}]", invalid)

    align_path = output_dir / "align_multimodal.semantic.json"
    if align_path.exists():
        align_rows = _load_json(align_path)
        if isinstance(align_rows, list):
            for i, row in enumerate(align_rows):
                if not isinstance(row, dict):
                    continue
                candidates = row.get("candidates", [])
                if not isinstance(candidates, list):
                    continue
                for j, item in enumerate(candidates):
                    if isinstance(item, dict):
                        _check_item(item, valid_codes, f"align[{i}].candidates[{j}]", invalid)

    result = {
        "output_dir": str(output_dir),
        "required_count": len(required),
        "present_count": len(present),
        "missing_count": len(missing),
        "empty_count": len(empty),
        "semantic_invalid_count": len(invalid),
        "status": "ok" if not missing and not empty and not invalid else "failed",
        "present": present,
        "missing": missing,
        "empty": empty,
        "semantic_invalid": invalid[:50],
    }

    if args.report:
        rp = Path(args.report).resolve()
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if int(args.strict) == 1 and result["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
