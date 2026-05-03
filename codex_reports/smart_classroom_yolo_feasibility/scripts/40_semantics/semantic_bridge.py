from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REQUIRED_SOURCE_FILES = [
    "behavior_det.jsonl",
    "actions.behavior.jsonl",
    "actions.behavior_aug.jsonl",
    "align_multimodal.json",
    "verified_events.jsonl",
]

OPTIONAL_SOURCE_FILES = [
    "per_person_sequences.json",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _resolve_repo_root(anchor: Path) -> Path:
    for candidate in [anchor] + list(anchor.parents):
        if (candidate / "data").exists() and (candidate / "scripts").exists():
            return candidate.resolve()
    raise RuntimeError(f"Cannot resolve repo root from: {anchor}")


def _resolve_path(repo_root: Path, raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
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


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _normalize_token(value: Any) -> str:
    token = str(value or "").strip().lower()
    token = " ".join(token.split())
    return token


def _load_mapping_file(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Taxonomy file is empty: {path}")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError("Taxonomy is not JSON and PyYAML is unavailable.") from exc
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("Taxonomy root must be an object.")
    return data


class Taxonomy:
    def __init__(self, raw: Dict[str, Any], *, display_language: str, mode: str) -> None:
        classes = raw.get("classes")
        if not isinstance(classes, list) or not classes:
            raise ValueError("taxonomy.classes must be a non-empty list")
        self.version = str(raw.get("taxonomy_version", "taxonomy_v1")).strip() or "taxonomy_v1"
        self.display_language = str(display_language).strip() or str(raw.get("display_language", "bilingual"))
        self.mode = str(mode).strip() or str(raw.get("mode", "compatible_enrich"))

        self.by_code: Dict[str, Dict[str, str]] = {}
        self.by_semantic: Dict[str, Dict[str, str]] = {}
        self.by_alias: Dict[str, Dict[str, str]] = {}

        for item in classes:
            if not isinstance(item, dict):
                continue
            behavior_code = _normalize_token(item.get("behavior_code"))
            semantic_id = _normalize_token(item.get("semantic_id"))
            label_zh = str(item.get("label_zh", "")).strip()
            label_en = str(item.get("label_en", "")).strip()
            semantic_label_zh = str(item.get("semantic_label_zh", label_zh)).strip()
            semantic_label_en = str(item.get("semantic_label_en", semantic_id or label_en)).strip()
            if not behavior_code or not semantic_id or not label_zh or not label_en:
                raise ValueError(f"Invalid taxonomy class entry: {item}")
            entry = {
                "behavior_code": behavior_code,
                "behavior_label_zh": label_zh,
                "behavior_label_en": label_en,
                "semantic_id": semantic_id,
                "semantic_label_zh": semantic_label_zh,
                "semantic_label_en": semantic_label_en,
            }
            self.by_code[behavior_code] = entry
            self.by_semantic[semantic_id] = entry

            aliases = item.get("aliases", [])
            if not isinstance(aliases, list):
                aliases = []
            alias_tokens = set(
                [
                    behavior_code,
                    semantic_id,
                    _normalize_token(label_zh),
                    _normalize_token(label_en),
                    _normalize_token(semantic_label_zh),
                    _normalize_token(semantic_label_en),
                ]
            )
            for alias in aliases:
                alias_tokens.add(_normalize_token(alias))
            for alias in alias_tokens:
                if alias:
                    self.by_alias[alias] = entry

        fallback_code = _normalize_token(raw.get("fallback_behavior_code", "tt"))
        self.fallback = self.by_code.get(fallback_code)
        if self.fallback is None:
            self.fallback = next(iter(self.by_code.values()))

    @property
    def valid_codes(self) -> set[str]:
        return set(self.by_code.keys())

    def resolve(self, hints: List[Any]) -> Tuple[Dict[str, str], str, str]:
        for hint in hints:
            token = _normalize_token(hint)
            if not token:
                continue
            if token in self.by_code:
                return self.by_code[token], "behavior_code", token
            if token in self.by_semantic:
                return self.by_semantic[token], "semantic_id", token
            if token in self.by_alias:
                return self.by_alias[token], "alias", token
        return self.fallback, "fallback", ""

    def apply(self, obj: Dict[str, Any], entry: Dict[str, str]) -> None:
        obj["behavior_code"] = entry["behavior_code"]
        obj["behavior_label_zh"] = entry["behavior_label_zh"]
        obj["behavior_label_en"] = entry["behavior_label_en"]
        obj["semantic_id"] = entry["semantic_id"]
        obj["semantic_label_zh"] = entry["semantic_label_zh"]
        obj["semantic_label_en"] = entry["semantic_label_en"]
        obj["taxonomy_version"] = self.version


def _new_file_stats(source: Path, target: Path) -> Dict[str, Any]:
    return {
        "source": str(source),
        "target": str(target),
        "source_exists": source.exists(),
        "rows_or_blocks": 0,
        "items": 0,
        "fallback_items": 0,
    }


def _append_anomaly(
    anomalies: List[Dict[str, Any]],
    *,
    file_tag: str,
    row_index: int,
    field_path: str,
    hints: List[Any],
    reason: str,
) -> None:
    anomalies.append(
        {
            "file": file_tag,
            "row_index": row_index,
            "field_path": field_path,
            "hints": [str(x) for x in hints if str(x).strip()],
            "reason": reason,
        }
    )


def _enrich_behavior_det(
    source: Path,
    target: Path,
    taxonomy: Taxonomy,
    anomalies: List[Dict[str, Any]],
) -> Dict[str, Any]:
    stats = _new_file_stats(source, target)
    if not source.exists():
        return stats

    rows = _load_jsonl(source)
    out_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        out_row = copy.deepcopy(row)
        behaviors = out_row.get("behaviors", [])
        if not isinstance(behaviors, list):
            behaviors = []
            out_row["behaviors"] = behaviors
        for bidx, behavior in enumerate(behaviors):
            if not isinstance(behavior, dict):
                continue
            hints = [
                behavior.get("behavior_code"),
                behavior.get("label"),
                behavior.get("semantic_id"),
                behavior.get("action"),
            ]
            entry, source_type, _ = taxonomy.resolve(hints)
            taxonomy.apply(behavior, entry)
            if source_type == "fallback":
                stats["fallback_items"] += 1
                _append_anomaly(
                    anomalies,
                    file_tag="behavior_det",
                    row_index=idx,
                    field_path=f"behaviors[{bidx}]",
                    hints=hints,
                    reason="fallback_mapping",
                )
            stats["items"] += 1
        out_row["taxonomy_version"] = taxonomy.version
        out_rows.append(out_row)

    stats["rows_or_blocks"] = _write_jsonl(target, out_rows)
    return stats


def _enrich_actions_jsonl(
    source: Path,
    target: Path,
    taxonomy: Taxonomy,
    anomalies: List[Dict[str, Any]],
    file_tag: str,
) -> Dict[str, Any]:
    stats = _new_file_stats(source, target)
    if not source.exists():
        return stats

    rows = _load_jsonl(source)
    out_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        out_row = copy.deepcopy(row)
        hints = [
            out_row.get("behavior_code"),
            out_row.get("label"),
            out_row.get("semantic_id"),
            out_row.get("action"),
            out_row.get("event_type"),
        ]
        entry, source_type, _ = taxonomy.resolve(hints)
        taxonomy.apply(out_row, entry)
        if source_type == "fallback":
            stats["fallback_items"] += 1
            _append_anomaly(
                anomalies,
                file_tag=file_tag,
                row_index=idx,
                field_path=".",
                hints=hints,
                reason="fallback_mapping",
            )
        out_rows.append(out_row)
        stats["items"] += 1

    stats["rows_or_blocks"] = _write_jsonl(target, out_rows)
    return stats


def _enrich_align_json(
    source: Path,
    target: Path,
    taxonomy: Taxonomy,
    anomalies: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, str]]]:
    stats = _new_file_stats(source, target)
    event_index: Dict[str, Dict[str, str]] = {}
    if not source.exists():
        return stats, event_index

    rows = _load_json(source)
    if not isinstance(rows, list):
        raise ValueError(f"align file must be list: {source}")

    out_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        out_row = copy.deepcopy(row)
        event_id = str(out_row.get("event_id", out_row.get("query_id", "")))
        candidates = out_row.get("candidates", [])
        if not isinstance(candidates, list):
            candidates = []
            out_row["candidates"] = candidates
        first_entry: Optional[Dict[str, str]] = None
        for cidx, cand in enumerate(candidates):
            if not isinstance(cand, dict):
                continue
            hints = [
                cand.get("behavior_code"),
                cand.get("label"),
                cand.get("semantic_id"),
                cand.get("action"),
                out_row.get("event_type"),
            ]
            entry, source_type, _ = taxonomy.resolve(hints)
            taxonomy.apply(cand, entry)
            if first_entry is None:
                first_entry = entry
            if source_type == "fallback":
                stats["fallback_items"] += 1
                _append_anomaly(
                    anomalies,
                    file_tag="align_multimodal",
                    row_index=idx,
                    field_path=f"candidates[{cidx}]",
                    hints=hints,
                    reason="fallback_mapping",
                )
            stats["items"] += 1
        if first_entry is None:
            hints = [out_row.get("behavior_code"), out_row.get("semantic_id"), out_row.get("event_type")]
            entry, _, _ = taxonomy.resolve(hints)
            first_entry = entry
        taxonomy.apply(out_row, first_entry)
        if event_id:
            event_index[event_id] = first_entry
        out_rows.append(out_row)

    _write_json(target, out_rows)
    stats["rows_or_blocks"] = len(out_rows)
    return stats, event_index


def _enrich_verified_jsonl(
    source: Path,
    target: Path,
    taxonomy: Taxonomy,
    event_index: Dict[str, Dict[str, str]],
    anomalies: List[Dict[str, Any]],
) -> Dict[str, Any]:
    stats = _new_file_stats(source, target)
    if not source.exists():
        return stats

    rows = _load_jsonl(source)
    out_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        out_row = copy.deepcopy(row)
        event_id = str(out_row.get("event_id", out_row.get("query_id", "")))
        hints = [
            out_row.get("behavior_code"),
            out_row.get("semantic_id"),
            out_row.get("event_type"),
            out_row.get("action"),
        ]
        entry, source_type, _ = taxonomy.resolve(hints)
        if source_type == "fallback" and event_id in event_index:
            entry = event_index[event_id]
            source_type = "align_top_candidate"
        taxonomy.apply(out_row, entry)
        out_row["semantic_source"] = source_type
        if source_type == "fallback":
            stats["fallback_items"] += 1
            _append_anomaly(
                anomalies,
                file_tag="verified_events",
                row_index=idx,
                field_path=".",
                hints=hints,
                reason="fallback_mapping",
            )
        stats["items"] += 1
        out_rows.append(out_row)

    stats["rows_or_blocks"] = _write_jsonl(target, out_rows)
    return stats


def _enrich_per_person_json(
    source: Path,
    target: Path,
    taxonomy: Taxonomy,
    anomalies: List[Dict[str, Any]],
) -> Dict[str, Any]:
    stats = _new_file_stats(source, target)
    if not source.exists():
        return stats

    payload = _load_json(source)
    if not isinstance(payload, dict):
        raise ValueError(f"per_person payload must be object: {source}")
    out_payload = copy.deepcopy(payload)
    people = out_payload.get("people", [])
    if not isinstance(people, list):
        people = []
        out_payload["people"] = people

    for pidx, person in enumerate(people):
        if not isinstance(person, dict):
            continue
        visual = person.get("visual_sequence", [])
        if not isinstance(visual, list):
            continue
        for vidx, item in enumerate(visual):
            if not isinstance(item, dict):
                continue
            hints = [
                item.get("behavior_code"),
                item.get("label"),
                item.get("semantic_id"),
                item.get("action"),
            ]
            entry, source_type, _ = taxonomy.resolve(hints)
            taxonomy.apply(item, entry)
            if source_type == "fallback":
                stats["fallback_items"] += 1
                _append_anomaly(
                    anomalies,
                    file_tag="per_person_sequences",
                    row_index=pidx,
                    field_path=f"visual_sequence[{vidx}]",
                    hints=hints,
                    reason="fallback_mapping",
                )
            stats["items"] += 1

    meta = out_payload.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
        out_payload["meta"] = meta
    meta["taxonomy_version"] = taxonomy.version
    meta["semantic_mode"] = taxonomy.mode
    meta["semantic_display_language"] = taxonomy.display_language

    _write_json(target, out_payload)
    stats["rows_or_blocks"] = len(people)
    return stats


def _validate_required_fields(
    *,
    output_dir: Path,
    taxonomy: Taxonomy,
    strict: bool,
) -> Dict[str, Any]:
    required_keys = [
        "behavior_code",
        "behavior_label_zh",
        "behavior_label_en",
        "semantic_id",
        "semantic_label_zh",
        "semantic_label_en",
        "taxonomy_version",
    ]
    checks: Dict[str, Any] = {
        "required_keys": required_keys,
        "invalid_records": [],
    }

    def _validate_item(item: Dict[str, Any], context: str) -> None:
        missing = [k for k in required_keys if not str(item.get(k, "")).strip()]
        code = _normalize_token(item.get("behavior_code"))
        if missing or code not in taxonomy.valid_codes:
            checks["invalid_records"].append(
                {
                    "context": context,
                    "missing_keys": missing,
                    "behavior_code": code,
                }
            )

    behavior_rows = _load_jsonl(output_dir / "behavior_det.semantic.jsonl")
    for i, row in enumerate(behavior_rows):
        for j, item in enumerate(row.get("behaviors", []) if isinstance(row.get("behaviors"), list) else []):
            if isinstance(item, dict):
                _validate_item(item, f"behavior_det[{i}].behaviors[{j}]")

    for name in ["actions.behavior.semantic.jsonl", "actions.behavior_aug.semantic.jsonl", "verified_events.semantic.jsonl"]:
        for i, row in enumerate(_load_jsonl(output_dir / name)):
            _validate_item(row, f"{name}[{i}]")

    align_rows = _load_json(output_dir / "align_multimodal.semantic.json")
    if isinstance(align_rows, list):
        for i, row in enumerate(align_rows):
            if not isinstance(row, dict):
                continue
            candidates = row.get("candidates", [])
            if not isinstance(candidates, list):
                continue
            for j, item in enumerate(candidates):
                if isinstance(item, dict):
                    _validate_item(item, f"align_multimodal[{i}].candidates[{j}]")

    checks["invalid_count"] = len(checks["invalid_records"])
    checks["status"] = "ok" if checks["invalid_count"] == 0 else "failed"
    if strict and checks["invalid_count"] > 0:
        first = checks["invalid_records"][0]
        raise RuntimeError(f"Semantic field validation failed: {first}")
    return checks


def _expected_outputs(output_dir: Path) -> List[str]:
    names = [
        "behavior_det.semantic.jsonl",
        "actions.behavior.semantic.jsonl",
        "actions.behavior_aug.semantic.jsonl",
        "align_multimodal.semantic.json",
        "verified_events.semantic.jsonl",
        "semantics_manifest.json",
    ]
    return [str((output_dir / name).resolve()) for name in names]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build semantic bridge artifacts from pipeline outputs.")
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument(
        "--taxonomy",
        default="codex_reports/smart_classroom_yolo_feasibility/profiles/action_semantics_8class.yaml",
        type=str,
    )
    parser.add_argument("--display_language", default="bilingual", type=str)
    parser.add_argument("--mode", default="compatible_enrich", type=str)
    parser.add_argument("--strict", default=1, type=int)
    parser.add_argument("--max_anomalies", default=200, type=int)
    parser.add_argument("--emit_only", type=int, default=1)
    parser.add_argument("--dry_run", type=int, default=0)
    parser.add_argument("--print_json", type=int, default=1)
    args = parser.parse_args()

    repo_root = _resolve_repo_root(Path(__file__).resolve())
    output_dir = _resolve_path(repo_root, args.output_dir)
    taxonomy_path = _resolve_path(repo_root, args.taxonomy)

    payload = {
        "stage": "semantic_bridge",
        "output_dir": str(output_dir),
        "taxonomy": str(taxonomy_path),
        "display_language": str(args.display_language),
        "mode": str(args.mode),
        "strict": int(args.strict),
        "expected_outputs": _expected_outputs(output_dir),
        "emit_only": int(args.emit_only),
        "dry_run": int(args.dry_run),
    }
    if int(args.print_json) == 1:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(
            " ".join(
                [
                    "semantic_bridge",
                    "--output_dir",
                    str(output_dir),
                    "--taxonomy",
                    str(taxonomy_path),
                ]
            )
        )

    if int(args.emit_only) == 1 or int(args.dry_run) == 1:
        return

    if not output_dir.exists():
        raise FileNotFoundError(f"output_dir not found: {output_dir}")
    if not taxonomy_path.exists():
        raise FileNotFoundError(f"taxonomy file not found: {taxonomy_path}")

    taxonomy = Taxonomy(_load_mapping_file(taxonomy_path), display_language=args.display_language, mode=args.mode)
    anomalies: List[Dict[str, Any]] = []
    files: Dict[str, Dict[str, Any]] = {}

    missing_required = []
    for name in REQUIRED_SOURCE_FILES:
        if not (output_dir / name).exists():
            missing_required.append(name)
    if missing_required and int(args.strict) == 1:
        raise FileNotFoundError(f"Missing required source files for semantic bridge: {missing_required}")

    files["behavior_det"] = _enrich_behavior_det(
        output_dir / "behavior_det.jsonl",
        output_dir / "behavior_det.semantic.jsonl",
        taxonomy,
        anomalies,
    )
    files["actions_behavior"] = _enrich_actions_jsonl(
        output_dir / "actions.behavior.jsonl",
        output_dir / "actions.behavior.semantic.jsonl",
        taxonomy,
        anomalies,
        "actions_behavior",
    )
    files["actions_behavior_aug"] = _enrich_actions_jsonl(
        output_dir / "actions.behavior_aug.jsonl",
        output_dir / "actions.behavior_aug.semantic.jsonl",
        taxonomy,
        anomalies,
        "actions_behavior_aug",
    )
    align_stats, event_index = _enrich_align_json(
        output_dir / "align_multimodal.json",
        output_dir / "align_multimodal.semantic.json",
        taxonomy,
        anomalies,
    )
    files["align_multimodal"] = align_stats
    files["verified_events"] = _enrich_verified_jsonl(
        output_dir / "verified_events.jsonl",
        output_dir / "verified_events.semantic.jsonl",
        taxonomy,
        event_index,
        anomalies,
    )
    files["per_person_sequences"] = _enrich_per_person_json(
        output_dir / "per_person_sequences.json",
        output_dir / "per_person_sequences.semantic.json",
        taxonomy,
        anomalies,
    )

    validation = _validate_required_fields(
        output_dir=output_dir,
        taxonomy=taxonomy,
        strict=bool(int(args.strict)),
    )

    total_items = sum(int(v.get("items", 0)) for v in files.values())
    fallback_items = sum(int(v.get("fallback_items", 0)) for v in files.values())

    manifest = {
        "stage": "semantic_bridge",
        "generated_at": _now_iso(),
        "output_dir": str(output_dir),
        "taxonomy_path": str(taxonomy_path),
        "taxonomy_version": taxonomy.version,
        "mode": taxonomy.mode,
        "display_language": taxonomy.display_language,
        "required_sources": REQUIRED_SOURCE_FILES,
        "optional_sources": OPTIONAL_SOURCE_FILES,
        "missing_required_sources": missing_required,
        "files": files,
        "summary": {
            "total_items": total_items,
            "fallback_items": fallback_items,
            "fallback_ratio": (round(float(fallback_items) / float(total_items), 6) if total_items > 0 else 0.0),
        },
        "validation": validation,
        "anomaly_count": len(anomalies),
        "anomalies": anomalies[: max(1, int(args.max_anomalies))],
        "status": "ok",
    }
    if missing_required and int(args.strict) != 1:
        manifest["status"] = "warning_missing_sources"
    if validation.get("status") != "ok":
        manifest["status"] = "failed_validation"

    _write_json(output_dir / "semantics_manifest.json", manifest)
    print(f"[DONE] semantic bridge outputs: {output_dir}")
    print(f"[INFO] taxonomy_version: {taxonomy.version}")
    print(f"[INFO] fallback_items: {fallback_items}/{total_items}")
    print(f"[INFO] validation_status: {validation.get('status')}")


if __name__ == "__main__":
    main()
